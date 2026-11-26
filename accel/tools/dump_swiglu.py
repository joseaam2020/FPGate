"""
Dump SwiGLU test vectors (read-only: nothing is recomputed or regenerated).

Takes the intermediates the e1_r7 package already stores for image 0
(vectors/stages_img_00.npz: fc1g, fc1x and swiglu outputs of blocks 0 and 11) and the
block's SiLU table / requant constants from manifest.json + tables.bin, and writes a
tile-sized slice as .hex files under accel/vectors/<block>_swiglu/:

  gi.hex     fc1g output (INT8)          R x C
  ui.hex     fc1x output (INT8)          R x C
  h.hex      expected swiglu_i (INT16)   R x C
  silu.hex   SiLU table, 255 x INT16     (index = gi + 127)
  params.txt R C M0 SHIFT

  docker run --rm -v "<repo>/model:/workspace" -v "<repo>/accel:/accel" -w /workspace \
      model-torch:latest /accel/tools/dump_swiglu.py --pkg quant/release/e1_r7 --out /accel/vectors
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/quant")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "quant"))

import export_check as ec  # noqa: E402


def hexfile(path, arr, bits):
    mask = (1 << bits) - 1
    nib = (bits + 3) // 4
    with open(path, "w") as f:
        for v in np.asarray(arr).reshape(-1).tolist():
            f.write(f"{int(v) & mask:0{nib}x}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--blocks", default="0,11")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=32)
    ap.add_argument("--r0", type=int, default=0)
    ap.add_argument("--c0", type=int, default=0)
    args = ap.parse_args()

    pkg = ec.Pkg(args.pkg, "cpu")
    st = np.load(pkg.root / pkg.m["vectors"][0]["stages_full"])
    R, C, r0, c0 = args.rows, args.cols, args.r0, args.c0
    rs, cs = slice(r0, r0 + R), slice(c0, c0 + C)

    for b in [int(x) for x in args.blocks.split(",")]:
        B = pkg.m["blocks"][b]
        gi, ui, h = (st[f"b{b}.{n}"][rs, cs] for n in ("fc1g_w", "fc1x_w", "swiglu_i"))
        silu = pkg.get("tables.bin", B["silu_table"], dtype=None).numpy()
        M0, sh = B["swiglu_requant"]["M0"], B["swiglu_requant"]["shift"]

        # self-check with the model formula (export_check.run): h = sat((silu[gi+127]*ui*M0 + 2^(sh-1)) >> sh)
        prod = silu.astype(np.int64)[gi.astype(np.int64) + 127] * ui.astype(np.int64)
        chk = np.clip((prod * M0 + (1 << (sh - 1))) >> sh, -32767, 32767)
        assert (chk == h).all(), "formula does not match the stored swiglu_i"

        out = Path(args.out) / f"b{b}_swiglu"
        out.mkdir(parents=True, exist_ok=True)
        hexfile(out / "gi.hex", gi, 8)
        hexfile(out / "ui.hex", ui, 8)
        hexfile(out / "h.hex", h, 16)
        hexfile(out / "silu.hex", silu, 16)
        (out / "params.txt").write_text(f"{R} {C} {M0} {sh}\n")
        print(f"block {b}: R={R} C={C} M0={M0} shift={sh} silu={silu.shape} "
              f"|h|max={np.abs(h).max()} saturated={(np.abs(h) == 32767).mean():.4f}")


if __name__ == "__main__":
    main()

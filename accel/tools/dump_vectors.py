"""
Dump real test vectors of the e1_r7 package for the accelerator testbenches.

No new model: it runs model/quant/export_check.run (bit-exact with the 20 reference
vectors) on image 0 and intercepts golden.int_matmul_acc / golden.requantize, which are
exactly what the GEMM array and the epilogue implement. For each requested layer it
writes one tile-sized slice as .hex files (one two's complement value per line):

  x.hex      activations   R x K   (in_bits wide, signed)
  w.hex      weights       C x K   (INT8)
  acc.hex    x . w^T       R x C   (no bias, 32 bit)
  bias.hex   bias          C       (32 bit)
  accb.hex   acc + bias    R x C   (epilogue input, 32 bit)
  m0.hex     M0            C       (32 bit)
  shift.hex  shift         C       (8 bit)
  y.hex      expected out  R x C   (out_bits wide, saturated)
  params.txt R C K IN_BITS OUT_BITS

Run it in the torch container with accel/ mounted (see accel/tools/README or DECISIONES.md):
  docker compose run --rm -v <repo>/accel:/accel torch \
      python /accel/tools/dump_vectors.py --pkg quant/release/e1_r7 --out /accel/vectors
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/workspace/quant")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "model" / "quant"))

import export_check as ec  # noqa: E402
import golden as gd  # noqa: E402

BLOCK_LAYERS = ["q_w", "k_w", "v_w", "proj_w", "fc1g_w", "fc1x_w", "fc2_w"]


class Done(Exception):
    pass


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
    ap.add_argument("--image", type=int, default=0)
    ap.add_argument("--layers", default="b0.q_w,b0.v_w,b0.fc1g_w,b0.fc2_w,b1.fc2_w")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--cols", type=int, default=32)
    ap.add_argument("--r0", type=int, default=0)
    ap.add_argument("--c0", type=int, default=0)
    args = ap.parse_args()

    want = set(args.layers.split(","))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pkg = ec.Pkg(args.pkg, dev)
    v = pkg.m["vectors"][args.image]
    img = np.fromfile(pkg.root / v["input"], dtype="<f4").reshape(v["input_shape"])

    # call order inside export_check.run
    mm_names = ["patch_embed"] + [f"b{i}.{n}" for i in range(len(pkg.m["blocks"])) for n in BLOCK_LAYERS] + ["head"]
    rq_names = [f"b{i}.{n}" for i in range(len(pkg.m["blocks"])) for n in BLOCK_LAYERS]
    cap, cnt = {}, {"mm": 0, "rq": 0}

    orig_mm, orig_rq = gd.int_matmul_acc, gd.requantize

    def mm(xq, wq):
        acc = orig_mm(xq, wq)
        name = mm_names[cnt["mm"]]
        cnt["mm"] += 1
        if name in want:
            cap.setdefault(name, {}).update(x=xq.cpu().numpy(), w=wq.cpu().numpy(), acc=acc.cpu().numpy())
        return acc

    def rq(acc, M0, shift, bits):
        y = orig_rq(acc, M0, shift, bits)
        name = rq_names[cnt["rq"]]
        cnt["rq"] += 1
        if name in want:
            cap[name].update(accb=acc.cpu().numpy(), m0=M0.cpu().numpy(), shift=shift.cpu().numpy(),
                             y=y.cpu().numpy(), out_bits=bits)
            if all(n in cap and "y" in cap[n] for n in want):
                raise Done
        return y

    gd.int_matmul_acc, gd.requantize = mm, rq
    try:
        with torch.inference_mode():
            ec.run(pkg, torch.from_numpy(img).to(dev))
    except Done:
        pass
    finally:
        gd.int_matmul_acc, gd.requantize = orig_mm, orig_rq

    r0, c0, R, C = args.r0, args.c0, args.rows, args.cols
    for name in sorted(want):
        d = cap[name]
        K = d["x"].shape[1]
        in_bits = pkg.m["layers"][name]["in_bits"]
        out_bits = d["out_bits"]
        rs, cs = slice(r0, r0 + R), slice(c0, c0 + C)
        acc, accb = d["acc"][rs, cs], d["accb"][rs, cs]
        bias = (accb - acc)[0]
        assert (accb - acc == bias[None, :]).all()
        # self-check of the slice: epilogue input -> expected output
        y_chk = gd.requantize(torch.from_numpy(accb), torch.from_numpy(d["m0"][cs]),
                              torch.from_numpy(d["shift"][cs]), out_bits).numpy()
        assert (y_chk == d["y"][rs, cs]).all()

        out = Path(args.out) / name.replace(".", "_")
        out.mkdir(parents=True, exist_ok=True)
        hexfile(out / "x.hex", d["x"][rs], 16 if in_bits > 8 else 8)
        hexfile(out / "w.hex", d["w"][cs], 8)
        hexfile(out / "acc.hex", acc, 32)
        hexfile(out / "bias.hex", bias, 32)
        hexfile(out / "accb.hex", accb, 32)
        hexfile(out / "m0.hex", d["m0"][cs], 32)
        hexfile(out / "shift.hex", d["shift"][cs], 8)
        hexfile(out / "y.hex", d["y"][rs, cs], 16 if out_bits > 8 else 8)
        (out / "params.txt").write_text(f"{R} {C} {K} {in_bits} {out_bits}\n")
        print(f"{name}: R={R} C={C} K={K} in_bits={in_bits} out_bits={out_bits} "
              f"|acc|max={np.abs(accb).max()} saturated={(np.abs(d['y'][rs, cs]) == (2**(out_bits-1)-1)).mean():.4f}")


if __name__ == "__main__":
    main()

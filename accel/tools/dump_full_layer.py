"""
Dump WHOLE layers of the e1_r7 package (all 1025 tokens, all output channels) in the
binary formats the accelerator reads from DDR3 (see DECISIONES.md, "H2"), plus the
expected output of the model. Read-only on the package: it runs export_check.run on
image 0 and intercepts golden.int_matmul_acc / golden.requantize, exactly like
dump_vectors.py, but keeps the full tensors.

Per layer, in <out>/full_<layer>/ :
  mem_a.bin   activations, tile-major  [subtile of 16 rows][plane][k][16 bytes]
  mem_w.bin   per tile of 16 channels: 256 bytes of constants, then K words of 16 bytes
  y.bin       expected output, row-major [M][N], int16 (or int8) little endian
  params.txt  M N K IN_BITS OUT_BITS SPP

  docker run --rm -v "<repo>/model:/workspace" -v "<repo>/accel:/accel" -w /workspace \
      model-torch:latest /accel/tools/dump_full_layer.py --pkg quant/release/e1_r7 \
      --out /accel/vectors --layers b0.q_w,b0.fc2_w
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


def pack_a(x, in_bits):
    """x (M,K) int -> tile-major bytes"""
    M, K = x.shape
    subs = (M + 15) // 16
    xp = np.zeros((subs * 16, K), dtype=np.int64)
    xp[:M] = x
    t = xp.reshape(subs, 16, K).transpose(0, 2, 1)            # (subs, K, 16)
    if in_bits > 8:
        hi = ((t >> 8) & 0xFF).astype(np.uint8)
        lo = (t & 0xFF).astype(np.uint8)
        return np.stack([hi, lo], axis=1).tobytes()           # (subs, 2, K, 16)
    return (t & 0xFF).astype(np.uint8).tobytes()


def pack_w(w, bias, m0, shift):
    """w (N,K) int8, per-channel constants -> per-tile blocks"""
    N, K = w.shape
    out = bytearray()
    for t in range(N // 16):
        c = np.zeros((16, 16), dtype=np.uint8)
        for j in range(16):
            n = t * 16 + j
            c[j, 0:4] = np.frombuffer(np.int32(bias[n]).tobytes(), dtype=np.uint8)
            c[j, 4:8] = np.frombuffer(np.int32(m0[n]).tobytes(), dtype=np.uint8)
            c[j, 8] = np.uint8(shift[n] & 0xFF)
        out += c.tobytes()
        out += (w[t * 16:(t + 1) * 16].T.astype(np.int64) & 0xFF).astype(np.uint8).tobytes()   # (K,16)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", type=int, default=0)
    ap.add_argument("--layers", default="b0.q_w,b0.fc2_w")
    args = ap.parse_args()

    want = set(args.layers.split(","))
    pkg = ec.Pkg(args.pkg, "cpu")
    v = pkg.m["vectors"][args.image]
    img = np.fromfile(pkg.root / v["input"], dtype="<f4").reshape(v["input_shape"])

    mm_names = ["patch_embed"] + [f"b{i}.{n}" for i in range(len(pkg.m["blocks"])) for n in BLOCK_LAYERS] + ["head"]
    rq_names = [f"b{i}.{n}" for i in range(len(pkg.m["blocks"])) for n in BLOCK_LAYERS]
    cap, cnt = {}, {"mm": 0, "rq": 0}
    orig_mm, orig_rq = gd.int_matmul_acc, gd.requantize

    def mm(xq, wq):
        acc = orig_mm(xq, wq)
        name = mm_names[cnt["mm"]]
        cnt["mm"] += 1
        if name in want:
            cap.setdefault(name, {}).update(x=xq.numpy(), w=wq.numpy(), acc=acc.numpy())
        return acc

    def rq(acc, M0, shift, bits):
        y = orig_rq(acc, M0, shift, bits)
        name = rq_names[cnt["rq"]]
        cnt["rq"] += 1
        if name in want:
            cap[name].update(accb=acc.numpy(), m0=M0.numpy(), shift=shift.numpy(), y=y.numpy(), out_bits=bits)
            if all(n in cap and "y" in cap[n] for n in want):
                raise Done
        return y

    gd.int_matmul_acc, gd.requantize = mm, rq
    try:
        with torch.inference_mode():
            ec.run(pkg, torch.from_numpy(img))
    except Done:
        pass
    finally:
        gd.int_matmul_acc, gd.requantize = orig_mm, orig_rq

    for name in sorted(want):
        d = cap[name]
        x, w, acc, accb = d["x"], d["w"], d["acc"], d["accb"]
        M, K = x.shape
        N = w.shape[0]
        in_bits = pkg.m["layers"][name]["in_bits"]
        out_bits = d["out_bits"]
        bias = (accb - acc)[0]
        assert (accb - acc == bias[None, :]).all()
        assert N % 16 == 0 and K % 2 == 0
        # the in_bits=16 layers must really fit 16 bits signed; 8-bit layers 8 bits
        lim = 1 << (in_bits - 1)
        assert x.min() >= -lim and x.max() < lim
        spp = 4 if in_bits > 8 else 16                         # 64 / 256 rows per panel (A RAM = 256 KiB)
        o = Path(args.out) / ("full_" + name.replace(".", "_"))
        o.mkdir(parents=True, exist_ok=True)
        (o / "mem_a.bin").write_bytes(pack_a(x, in_bits))
        (o / "mem_w.bin").write_bytes(pack_w(w, bias, d["m0"], d["shift"]))
        y = d["y"].astype("<i2" if out_bits > 8 else "i1")
        (o / "y.bin").write_bytes(y.tobytes())
        (o / "params.txt").write_text(f"{M} {N} {K} {in_bits} {out_bits} {spp}\n")
        print(f"{name}: M={M} N={N} K={K} in={in_bits} out={out_bits} spp={spp} "
              f"A={len(pack_a(x, in_bits))} B, W={N // 16 * (256 + 16 * K)} B, Y={y.nbytes} B")


if __name__ == "__main__":
    main()

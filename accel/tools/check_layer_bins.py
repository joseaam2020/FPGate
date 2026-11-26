"""
Sanity check of a packed layer directory (mem_a.bin, mem_w.bin, y.bin, params.txt): decode the
DDR3 images exactly as the accelerator reads them, recompute the layer in integer arithmetic
(int64 accumulator, M0/shift requantization, saturation) and compare with y.bin.
This validates the file formats and the dump; it does not exercise the hardware.

  python check_layer_bins.py --dir accel/vectors/full_b0_q_w [--rows 64]
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--rows", type=int, default=0, help="only check the first N rows (0 = all)")
    args = ap.parse_args()
    d = Path(args.dir)
    M, N, K, ib, ob, spp = [int(t) for t in (d / "params.txt").read_text().split()[:6]]
    planes = 2 if ib > 8 else 1
    subs = (M + 15) // 16
    nt = N // 16

    a = np.frombuffer((d / "mem_a.bin").read_bytes(), dtype=np.uint8)
    assert a.size == subs * planes * K * 16, "mem_a.bin size"
    a = a.reshape(subs, planes, K, 16)
    if planes == 2:
        v = (a[:, 0].astype(np.int64) << 8) | a[:, 1].astype(np.int64)
        v = np.where(v >= 32768, v - 65536, v)
    else:
        v = a[:, 0].astype(np.int64)
        v = np.where(v >= 128, v - 256, v)
    x = v.transpose(0, 2, 1).reshape(subs * 16, K)[:M]          # (M, K)

    wbuf = np.frombuffer((d / "mem_w.bin").read_bytes(), dtype=np.uint8)
    tile = 256 + 16 * K
    assert wbuf.size == nt * tile, "mem_w.bin size"
    w = np.zeros((N, K), dtype=np.int64)
    bias = np.zeros(N, dtype=np.int64)
    m0 = np.zeros(N, dtype=np.int64)
    sh = np.zeros(N, dtype=np.int64)
    for t in range(nt):
        blk = wbuf[t * tile:(t + 1) * tile]
        c = blk[:256].reshape(16, 16)
        for j in range(16):
            n = t * 16 + j
            bias[n] = int(np.frombuffer(c[j, 0:4].tobytes(), dtype="<i4")[0])
            m0[n] = int(np.frombuffer(c[j, 4:8].tobytes(), dtype="<i4")[0])
            sh[n] = int(c[j, 8])
        wt = blk[256:].reshape(K, 16).astype(np.int64)
        wt = np.where(wt >= 128, wt - 256, wt)
        w[t * 16:(t + 1) * 16] = wt.T

    y_exp = np.frombuffer((d / "y.bin").read_bytes(), dtype="<i2" if ob > 8 else "i1").reshape(M, N)
    R = args.rows or M
    acc = x[:R] @ w.T + bias[None, :]
    assert np.abs(acc).max() < 2**31, "accumulator overflow"
    prod = acc * m0[None, :]
    y = (prod + (np.int64(1) << (sh[None, :] - 1))) >> sh[None, :]
    qm = 32767 if ob > 8 else 127
    y = np.clip(y, -qm, qm)
    bad = int((y != y_exp[:R]).sum())
    print(f"{d.name}: M={M} N={N} K={K} in={ib} out={ob}: {R * N} outputs, {bad} mismatches, "
          f"saturated {float((np.abs(y) == qm).mean()):.4f}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

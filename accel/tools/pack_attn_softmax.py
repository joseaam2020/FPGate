#!/usr/bin/env python3
"""Pack the real scores of one head for the softmax unit testbench (tb/tb_attn_softmax.sv).

  python accel/tools/pack_attn_softmax.py accel/vectors/attn_b0_h0 --rows 32

Reads s.bin / p.bin / params.txt written by `fpgate-accel --dump-attn` and writes, next to them:
  sm_s.hex       the first --rows rows of S as the engine's RAW32 output would leave them in DDR3:
                 1040 INT32 per row (130 beats of 8), the 15 padding columns after the 1025 real ones filled
                 with a huge value that would dominate the maximum if the unit did not mask them
  sm_p.hex       the expected probabilities, one 16-bit hex value per line, [rows][1025]
  sm_params.txt  rows cols stride_bytes sm_m0 sm_sh
"""
import argparse
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rows", type=int, default=32)
    a = ap.parse_args()
    d = Path(a.dir)
    T, hd, m0, sh, av_m0, av_sh = [int(x) for x in (d / "params.txt").read_text().split()]
    s_all = struct.unpack(f"<{T * T}i", (d / "s.bin").read_bytes())
    p_all = struct.unpack(f"<{T * T}H", (d / "p.bin").read_bytes())
    NCOL = 1040
    POISON = 0x7FFF0000
    lines = []
    for r in range(a.rows):
        row = list(s_all[r * T:(r + 1) * T]) + [POISON] * (NCOL - T)
        for b in range(NCOL // 8):
            lines.append("".join(f"{row[8 * b + l] & 0xFFFFFFFF:08x}" for l in reversed(range(8))))
    (d / "sm_s.hex").write_text("\n".join(lines) + "\n")
    (d / "sm_p.hex").write_text("\n".join(f"{p_all[r * T + c]:04x}" for r in range(a.rows) for c in range(T)) + "\n")
    (d / "sm_params.txt").write_text(f"{a.rows} {T} {NCOL * 4} {m0} {sh}\n")
    print(f"{d.name}: {a.rows} rows x {T} cols, M0 = {m0}, shift = {sh}: {len(lines)} beats")


if __name__ == "__main__":
    main()

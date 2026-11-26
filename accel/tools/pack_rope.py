#!/usr/bin/env python3
"""Extract the RoPE tables of a release package for tb/tb_outfmt.sv and check that they are separable.

  python accel/tools/pack_rope.py model/quant/release/e1_r7 [--cos-off 2624 --sin-off 133696] [--frac 14 --pre 7]

tables.bin holds cos and sin as [1024][64] int16 (Q1.14); row t-1 belongs to token t = 1 + 32*y + x. The hardware
(rtl/rope_unit.sv) keeps four 512 x 16 RAMs instead: d < 32 depends on y only, d >= 32 on x only, and the values of
the pair (2i, 2i+1) are identical. This script asserts exactly that on the real tables and writes, in
accel/vectors/rope/:
  cos.hex, sin.hex   the full tables, one 4-digit hex value per line, [1024][64] row-major
  params.txt         frac pre
"""
import argparse
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("--cos-off", type=int, default=2624)
    ap.add_argument("--sin-off", type=int, default=133696)
    ap.add_argument("--frac", type=int, default=14)
    ap.add_argument("--pre", type=int, default=7)
    a = ap.parse_args()
    raw = (Path(a.pkg) / "tables.bin").read_bytes()
    n = 1024 * 64
    cos = struct.unpack_from(f"<{n}h", raw, a.cos_off)
    sin = struct.unpack_from(f"<{n}h", raw, a.sin_off)
    for name, tab in (("cos", cos), ("sin", sin)):
        for u in range(1024):
            y, x = u >> 5, u & 31
            for d in range(64):
                v = tab[u * 64 + d]
                src = (y * 32 + 0) if d < 32 else (0 * 32 + x)           # y-only half, x-only half
                assert v == tab[src * 64 + d], (name, u, d)
                if d & 1:
                    assert v == tab[u * 64 + d - 1], (name, u, d, "pair")
    out = Path(__file__).resolve().parent.parent / "vectors" / "rope"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cos.hex").write_text("\n".join(f"{v & 0xFFFF:04x}" for v in cos) + "\n")
    (out / "sin.hex").write_text("\n".join(f"{v & 0xFFFF:04x}" for v in sin) + "\n")
    (out / "params.txt").write_text(f"{a.frac} {a.pre}\n")
    print("separable: yes; hardware tables 4 x 512 x 16 bit = 4096 bytes")


if __name__ == "__main__":
    main()

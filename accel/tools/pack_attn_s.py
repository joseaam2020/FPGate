#!/usr/bin/env python3
"""Pack a slice of the real attention scores problem S = Q.K^T (one head) into the DDR3 formats of the
accelerator, for the RAW32 testbench (tb/tb_raw32.sv).

  python accel/tools/pack_attn_s.py accel/vectors/attn_b0_h0 --rows 32 --keys 48

Reads q8.bin, k8.bin, s.bin written by `fpgate-accel --dump-attn` (INT8 head slices [1025][64] and the
INT32 scores [1025][1025]) and writes, next to them:
  raw32_a.hex   the first --rows query rows, activations tile-major (K = 64, one plane)
  raw32_w.hex   the first --keys keys as weight tiles of 16 channels (256 bytes of constants, then K words)
  raw32_s.hex   the expected scores [rows][keys], one 32-bit hex value per line
  raw32_params.txt   rows keys 64
One 256-bit beat per line, byte 31 first (same as pack_layer.py).
"""
import argparse
import struct
from pathlib import Path


def beats(words):
    assert len(words) % 2 == 0
    return [(words[i] + words[i + 1])[::-1].hex() for i in range(0, len(words), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--keys", type=int, default=48)
    a = ap.parse_args()
    d = Path(a.dir)
    T, HD = [int(x) for x in (d / "params.txt").read_text().split()[:2]]
    q = (d / "q8.bin").read_bytes()
    k = (d / "k8.bin").read_bytes()
    s_all = struct.unpack(f"<{T * T}i", (d / "s.bin").read_bytes())
    assert a.keys % 16 == 0

    def sb(buf, r, c):
        v = buf[r * HD + c]
        return v & 0xFF

    subs = (a.rows + 15) // 16
    a_words = []
    for s in range(subs):
        for kk in range(HD):
            a_words.append(bytes(sb(q, s * 16 + i, kk) if s * 16 + i < a.rows else 0 for i in range(16)))
    w_words = []
    for t in range(a.keys // 16):
        w_words += [bytes(16)] * 16                       # epilogue constants: unused in raw mode
        for kk in range(HD):
            w_words.append(bytes(sb(k, t * 16 + j, kk) for j in range(16)))
    (d / "raw32_a.hex").write_text("\n".join(beats(a_words)) + "\n")
    (d / "raw32_w.hex").write_text("\n".join(beats(w_words)) + "\n")
    ref = [s_all[r * T + c] & 0xFFFFFFFF for r in range(a.rows) for c in range(a.keys)]
    (d / "raw32_s.hex").write_text("\n".join(f"{v:08x}" for v in ref) + "\n")
    (d / "raw32_params.txt").write_text(f"{a.rows} {a.keys} {HD}\n")
    print(f"{d.name}: {a.rows} rows x {a.keys} keys, K = {HD}: A {len(a_words) // 2} beats, W {len(w_words) // 2} beats, "
          f"S range {min(s_all[r * T + c] for r in range(a.rows) for c in range(a.keys))} .. "
          f"{max(s_all[r * T + c] for r in range(a.rows) for c in range(a.keys))}")


if __name__ == "__main__":
    main()

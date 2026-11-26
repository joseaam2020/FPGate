"""
Pack one dumped layer slice (accel/vectors/<layer>/, written by dump_vectors.py from the
e1_r7 package) into the DDR3 formats of the accelerator (see DECISIONES.md, "H2"):

  mem_a.hex   activations, tile-major: [subtile of 16 rows][plane][k][16 bytes]
              (16-bit inputs: plane 0 = high byte, plane 1 = low byte)
  mem_w.hex   per tile of 16 output channels: 8 beats of epilogue constants
              (16 entries x {bias i32, M0 i32, shift u8, pad}) then K words of 16 bytes
              (byte j = weight of channel j)

One 256-bit beat per line, 64 hex digits, byte 31 first. Only the standard library is
used; the model package is only read through the already dumped slices.

  python pack_layer.py --dir accel/vectors/b0_q_w [--check]
"""

import argparse
import struct
from pathlib import Path


def read_hex(path, bits):
    mask, sign = (1 << bits) - 1, 1 << (bits - 1)
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            v = int(line, 16) & mask
            out.append(v - (1 << bits) if v & sign else v)
    return out


def beats_to_lines(words16):
    """list of 16-byte words (bytes objects, byte 0 first) -> beats of 2 words"""
    assert len(words16) % 2 == 0, "K must be even"
    lines = []
    for i in range(0, len(words16), 2):
        beat = words16[i] + words16[i + 1]           # byte 0..15 then 16..31
        lines.append(beat[::-1].hex())               # byte 31 first
    return lines


def pack(d):
    R, C, K, ib, ob = [int(t) for t in (d / "params.txt").read_text().split()]
    assert C % 16 == 0 and K % 2 == 0
    xb = 16 if ib > 8 else 8
    x = read_hex(d / "x.hex", xb)
    w = read_hex(d / "w.hex", 8)
    bias = read_hex(d / "bias.hex", 32)
    m0 = read_hex(d / "m0.hex", 32)
    sh = read_hex(d / "shift.hex", 8)
    assert len(x) == R * K and len(w) == C * K

    planes = 2 if xb == 16 else 1
    subs = (R + 15) // 16
    a_words = []
    for s in range(subs):
        for p in range(planes):
            for k in range(K):
                word = bytearray(16)
                for i in range(16):
                    row = s * 16 + i
                    v = x[row * K + k] if row < R else 0
                    word[i] = ((v >> 8) & 0xFF) if (planes == 2 and p == 0) else (v & 0xFF)
                a_words.append(bytes(word))

    w_words = []
    for t in range(C // 16):
        for j in range(16):
            n = t * 16 + j
            e = bytearray(16)
            e[0:4] = (bias[n] & 0xFFFFFFFF).to_bytes(4, "little")
            e[4:8] = (m0[n] & 0xFFFFFFFF).to_bytes(4, "little")
            e[8] = sh[n] & 0xFF
            w_words.append(bytes(e))
        for k in range(K):
            w_words.append(bytes((w[(t * 16 + j) * K + k] & 0xFF) for j in range(16)))
    return (R, C, K, ib, ob), a_words, w_words, (x, w, bias, m0, sh)


def check(d, meta, a_lines, w_lines, raw):
    """Re-read the packed text and compare with the source values."""
    R, C, K, ib, ob = meta
    x, w, bias, m0, sh = raw
    planes = 2 if ib > 8 else 1

    def words(lines):
        out = []
        for ln in lines:
            b = bytes.fromhex(ln)[::-1]
            out += [b[:16], b[16:]]
        return out

    aw, ww = words(a_lines), words(w_lines)
    for s in range((R + 15) // 16):
        for k in range(K):
            for i in range(16):
                row = s * 16 + i
                if row >= R:
                    continue
                if planes == 2:
                    v = (aw[((s * 2) * K) + k][i] << 8) | aw[((s * 2 + 1) * K) + k][i]
                    v = v - 65536 if v & 0x8000 else v
                else:
                    v = aw[s * K + k][i]
                    v = v - 256 if v & 0x80 else v
                assert v == x[row * K + k], f"A mismatch s={s} k={k} i={i}"
    stride = 16 + K
    for t in range(C // 16):
        for j in range(16):
            e = ww[t * stride + j]
            assert int.from_bytes(e[0:4], "little", signed=True) == bias[t * 16 + j]
            assert int.from_bytes(e[4:8], "little", signed=True) == m0[t * 16 + j]
            assert e[8] == sh[t * 16 + j] & 0xFF
        for k in range(K):
            for j in range(16):
                v = ww[t * stride + 16 + k][j]
                v = v - 256 if v & 0x80 else v
                assert v == w[(t * 16 + j) * K + k], f"W mismatch t={t} k={k} j={j}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    d = Path(args.dir)
    meta, a_words, w_words, raw = pack(d)
    a_lines, w_lines = beats_to_lines(a_words), beats_to_lines(w_words)
    (d / "mem_a.hex").write_text("\n".join(a_lines) + "\n")
    (d / "mem_w.hex").write_text("\n".join(w_lines) + "\n")
    # binary images for the board (same layout as dump_full_layer.py)
    (d / "mem_a.bin").write_bytes(b"".join(a_words))
    (d / "mem_w.bin").write_bytes(b"".join(w_words))
    R_, C_, K_, ib_, ob_ = meta
    y_vals = read_hex(d / "y.hex", 16 if ob_ > 8 else 8)
    fmt = ("<%dh" if ob_ > 8 else "%db") % len(y_vals)
    (d / "y.bin").write_bytes(struct.pack(fmt, *y_vals))
    (d / "params6.txt").write_text(f"{R_} {C_} {K_} {ib_} {ob_} 2\n")
    if args.check:
        check(d, meta, a_lines, w_lines, raw)
    R, C, K, ib, ob = meta
    print(f"{d.name}: R={R} C={C} K={K} in={ib} out={ob} -> A {len(a_lines)} beats, W {len(w_lines)} beats"
          + (" (checked)" if args.check else ""))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pack the first rows of one head's attention (all keys) for the chain testbench tb/tb_attn_core.sv:
Q.K^T (RAW32) -> softmax unit -> P.V, through the registers of fpgate_accel_core.

  python accel/tools/pack_attn_full.py accel/vectors/attn_b0_h0 --rows 32

Reads q8.bin k8.bin v8.bin ctx.bin params.txt (written by `fpgate-accel --dump-attn`) and writes next to them:
  full_a.hex     first --rows query rows, activations tile-major (K = 64, 8-bit)
  full_wk.hex    all keys, padded to 1040, as weight tiles of 16 (constants unused in raw mode, then 64 words)
  full_wv.hex    V as weight tiles of 16 output channels (N = 64): constants {bias 0, M0 = av_m0, shift = av_sh}
                 then K = 1026 words (word k = V[k][16t .. 16t+15], zero for the padding row k = 1025)
  full_ctx.hex   expected INT8 context of the first rows, [rows][64], two hex digits per line
  full_params.txt  rows sm_m0 sm_sh
"""
import argparse
from pathlib import Path


def beats(words):
    assert len(words) % 2 == 0
    return [(words[i] + words[i + 1])[::-1].hex() for i in range(0, len(words), 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--rows", type=int, default=32)
    a = ap.parse_args()
    d = Path(a.dir)
    T, HD, sm_m0, sm_sh, av_m0, av_sh = [int(x) for x in (d / "params.txt").read_text().split()]
    q, k, v = (d / "q8.bin").read_bytes(), (d / "k8.bin").read_bytes(), (d / "v8.bin").read_bytes()
    ctx = (d / "ctx.bin").read_bytes()
    KP, NK = 1026, 1040

    a_words = []
    for s in range((a.rows + 15) // 16):
        for kk in range(HD):
            a_words.append(bytes((q[(s * 16 + i) * HD + kk] & 0xFF) if s * 16 + i < a.rows else 0 for i in range(16)))
    wk = []
    for t in range(NK // 16):
        wk += [bytes(16)] * 16
        for kk in range(HD):
            wk.append(bytes((k[(t * 16 + j) * HD + kk] & 0xFF) if t * 16 + j < T else 0 for j in range(16)))
    wv = []
    const = bytearray(16)
    const[4:8] = av_m0.to_bytes(4, "little")
    const[8] = av_sh
    for t in range(HD // 16):
        wv += [bytes(const)] * 16
        for kk in range(KP):
            wv.append(bytes((v[kk * HD + t * 16 + j] & 0xFF) if kk < T else 0 for j in range(16)))
    (d / "full_a.hex").write_text("\n".join(beats(a_words)) + "\n")
    (d / "full_wk.hex").write_text("\n".join(beats(wk)) + "\n")
    (d / "full_wv.hex").write_text("\n".join(beats(wv)) + "\n")
    (d / "full_ctx.hex").write_text("\n".join(f"{ctx[r * HD + c] & 0xFF:02x}" for r in range(a.rows) for c in range(HD)) + "\n")
    (d / "full_params.txt").write_text(f"{a.rows} {sm_m0} {sm_sh}\n")
    print(f"{d.name}: {a.rows} rows, K tiles {len(wk) // 2} beats, V tiles {len(wv) // 2} beats, M0 = {sm_m0} shift = {sm_sh}, "
          f"av_requant M0 = {av_m0} shift = {av_sh}")


if __name__ == "__main__":
    main()

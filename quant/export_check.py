"""
Verifica el paquete exportado SIN usar IntModel, la calibracion ni el checkpoint: reconstruye la
inferencia solo con manifest.json + los .bin (lo mismo que haria el programa de Linux/HPS) y compara
los logits con vectors/ bit a bit (patron de bits float32) y los intermedios por SHA-256.

Uso (en el contenedor):
    python quant/export_check.py --pkg quant/release/e1_r7
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import float_ref as fr
import golden as gd
import nonlinear as nl

NP = {"int8": np.int8, "int16": np.int16, "int32": np.int32, "uint16": np.uint16,
      "float32": np.float32, "float64": np.float64}


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Pkg:
    def __init__(self, root, dev):
        self.root, self.dev = Path(root), dev
        self.m = json.load(open(self.root / "manifest.json"))
        self.bins = {n: np.memmap(self.root / n, dtype=np.uint8, mode="r") for n in self.m["files"]}
        for n, d in self.m["files"].items():
            h = hashlib.sha256(open(self.root / n, "rb").read()).hexdigest()
            assert h == d["sha256"], f"hash distinto en {n}"

    def get(self, fname, d, dtype=torch.int64):
        dt = np.dtype(d["dtype"])
        n = int(np.prod(d["shape"]))
        a = np.frombuffer(self.bins[fname], dtype=dt, count=n, offset=d["offset"]).reshape(d["shape"])
        t = torch.from_numpy(a.astype(np.int32) if dt == np.uint16 else a.copy())
        return t.to(dtype).to(self.dev) if dtype is not None else t.to(self.dev)


def requant(acc, M0, shift, bits):
    return gd.requantize(acc, M0, shift, bits)


def run(pkg, img):
    m, dev = pkg.m, pkg.dev
    L = m["layers"]
    stages = {}

    def w(name):
        return pkg.get("weights_i8.bin", L[name]["weights"], torch.int8)

    def acc_of(name, x):
        d = L[name]
        xq = gd.quantize_act(x, d["s_x"], d["in_bits"])
        acc = gd.int_matmul_acc(xq, w(name))
        if d["bias"] is not None:
            acc = acc + pkg.get("epilogue_i32.bin", d["bias"])[None, :]
        return acc

    def gemm_int(name, x):
        d = L[name]
        return requant(acc_of(name, x), pkg.get("epilogue_i32.bin", d["M0"]),
                       pkg.get("epilogue_i32.bin", d["shift"]), d["out_bits"])

    def gemm_float(name, x):
        d = L[name]
        return (acc_of(name, x).double() * pkg.get("epilogue_f64.bin", d["deq_f64"], torch.float64)[None, :]).float()

    def hps(d):
        return pkg.get("hps_f32.bin", d, torch.float32)

    def fixed(c, v):
        M0, sh = c["M0"], c["shift"]
        return (v * M0 + (1 << (sh - 1))) >> sh

    def rq(acc, c, bits):
        return fixed(c, acc).clamp(-gd.qmax(bits), gd.qmax(bits))

    etab = pkg.get("tables.bin", m["tables"]["exp"])
    rtab = pkg.get("tables.bin", m["tables"]["recip"])
    C = pkg.get("tables.bin", m["tables"]["rope_cos_q14"])
    S_ = pkg.get("tables.bin", m["tables"]["rope_sin_q14"])
    fb, ps = m["rope"]["frac_bits"], m["rope"]["pre_shift"]
    P = m["softmax_bits"]
    qm_in = gd.qmax(m["silu_in_bits"])

    def rope(xi):
        sh = fb + ps
        pre, pat = xi[:, :1], xi[:, 1:]
        pat = (pat * C + fr.rot_interleaved(pat) * S_ + (1 << (sh - 1))) >> sh
        pre = (pre + (1 << (ps - 1))) >> ps
        return torch.cat([pre, pat], 1).clamp(-127, 127)

    def softmax(S, c):
        n = etab.numel() - 1
        d = S.amax(-1, keepdim=True) - S
        idx = fixed(c, d).clamp(max=n)
        e = etab[idx]
        ssum = e.sum(-1, keepdim=True)
        k = torch.floor(torch.log2(ssum.double())).to(torch.int64)
        mant = (ssum >> (k - 8)) - 256
        sh = 15 + k - P
        p = (e * rtab[mant] + (torch.ones_like(sh) << (sh - 1))) >> sh
        return p.clamp(0, 2 ** P - 1)

    cols = torch.nn.functional.unfold(img.unsqueeze(0), kernel_size=14, stride=14)[0].t()
    x = gemm_float("patch_embed", cols)
    x = torch.cat([hps(m["hps"]["cls"]).reshape(1, -1), x], 0) + hps(m["hps"]["pos"]).reshape(1025, -1)

    for i, B in enumerate(m["blocks"]):
        b = f"b{i}"
        h = B["heads"]
        hp = B["hps"]
        n1 = fr.layer_norm(x, hps(hp["n1_g"]), hps(hp["n1_b"]))
        qi = rope(fr.split_heads(gemm_int(f"{b}.q_w", n1), h))
        ki = rope(fr.split_heads(gemm_int(f"{b}.k_w", n1), h))
        vi = fr.split_heads(gemm_int(f"{b}.v_w", n1), h)
        p_int = torch.cat([softmax(nl.int_bmm(qi[j:j + 1], ki[j:j + 1].transpose(-1, -2)), B["softmax_idx"])
                           for j in range(h)])
        ctx_i = rq(nl.int_bmm(p_int, vi), B["av_requant"], 8)
        ctx = fr.merge_heads((ctx_i.double() * B["s_ctx"]).float())
        a = (gemm_int(f"{b}.proj_w", ctx).double() * B["s_proj_out"]).float()
        x = x + a
        n2 = fr.layer_norm(x, hps(hp["n2_g"]), hps(hp["n2_b"]))
        gi = gemm_int(f"{b}.fc1g_w", n2)
        ui = gemm_int(f"{b}.fc1x_w", n2)
        sg = pkg.get("tables.bin", B["silu_table"])[gi + qm_in]
        h_i = rq(sg * ui, B["swiglu_requant"], 16)
        hh = (h_i.double() * B["s_swiglu_h"]).float()
        hh = fr.layer_norm(hh, hps(hp["mlpn_g"]), hps(hp["mlpn_b"]))
        mm = (gemm_int(f"{b}.fc2_w", hh).double() * B["s_fc2_out"]).float()
        x = x + mm
        stages[f"{b}.ctx_i"], stages[f"{b}.swiglu_i"] = fr.merge_heads(ctx_i), h_i
    pooled = fr.layer_norm(x[1:].mean(0, keepdim=True), hps(m["hps"]["fcn_g"]), hps(m["hps"]["fcn_b"]))
    return gemm_float("head", pooled)[0], stages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", required=True)
    ap.add_argument("--n", type=int, default=None, help="cuantos vectores comprobar (todos por defecto)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pkg = Pkg(args.pkg, dev)
    vecs = pkg.m["vectors"][: args.n]
    bad = 0
    for v in vecs:
        img = np.fromfile(pkg.root / v["input"], dtype="<f4").reshape(v["input_shape"])
        assert sha(img) == v["input_sha256"]
        with torch.inference_mode():
            lg, st = run(pkg, torch.from_numpy(img).to(dev))
        got = lg.cpu().numpy().astype("<f4").view("<u4").tolist()
        ok = got == v["logits_hex"]
        print("  logits identicos:", ok)
        # intermedios que existen en ambos lados (comparacion por SHA-256)
        for k, t in st.items():
            ref = v["stage_sha256"].get(k)
            if ref is not None and k.endswith(("ctx_i", "swiglu_i")) and sha(t.cpu().numpy()) != ref:
                ok = False
                print(f"  intermedio distinto: {k}")
        bad += not ok
        print(f"vector {v['index']:2d}: {'OK ' if ok else 'FALLA'} logits={lg.cpu().tolist()}", flush=True)
    print(f"\n{len(vecs) - bad}/{len(vecs)} vectores identicos bit a bit")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()

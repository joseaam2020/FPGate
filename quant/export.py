"""
Exporta el modelo entero congelado (frozen.py) a un paquete que Linux (HPS de la DE10-Standard)
pueda leer con mmap/fread y cargar a la FPGA, mas vectores de prueba para verificar bit a bit.

Salida (quant/release/<nombre>/):
  manifest.json        capas, formas, offsets, escalas, constantes de punto fijo, hashes
  weights_i8.bin       pesos INT8 (out, in), fila por fila, una escala por fila
  epilogue_i32.bin     por capa: sesgo INT32, M0 INT32, shift INT32 (cada uno de largo `out`)
  epilogue_f64.bin     capas de salida flotante (patch_embed, head): s_x * s_w[c] en float64
  tables.bin           EXP (uint16), RECIP (uint16), SiLU por bloque (int16), RoPE cos/sin Q1.14 (int16)
  hps_f32.bin          parametros flotantes del HPS: cls, pos, LayerNorm de cada bloque, fc_norm
  vectors/             imagenes preprocesadas (float32 CHW), logits esperados, intermedios
Todo little-endian, cada tensor alineado a 64 bytes.

Uso (en el contenedor):
    python quant/export.py --ckpt quant/ckpt/r7_heads33_neur40.pt --calib quant/calib_r7/calib_stats.json \
        --out quant/release/e1_r7 --n-vectors 20
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg

import float_ref as fr
import frozen
import golden as gd
import nonlinear as nl
from calibrate import TIMM_NAME, to_device
from int_model import IntModel

ALIGN = 64
FULL_TRACE_BLOCKS = (0, 11)      # bloques cuyos intermedios se guardan completos (el resto, solo SHA-256)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class Packer:
    """Escribe tensores alineados a 64 bytes en un solo archivo y devuelve sus offsets."""

    def __init__(self, path):
        self.f = open(path, "wb")
        self.pos = 0

    def add(self, arr, dtype):
        a = np.ascontiguousarray(np.asarray(arr).astype(dtype))
        pad = (-self.pos) % ALIGN
        self.f.write(b"\0" * pad)
        self.pos += pad
        off = self.pos
        self.f.write(a.tobytes())
        self.pos += a.nbytes
        return dict(offset=off, shape=list(a.shape), dtype=np.dtype(dtype).name)

    def close(self):
        self.f.close()


class Tracer(IntModel):
    """IntModel que registra las constantes de cada capa (una vez) y los intermedios de la imagen."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.layers = {}      # nombre -> dict con constantes de la capa
        self.tr = {}          # nombre -> tensor entero/flotante de la imagen actual
        self._nrope = 0

    def gemm_int(self, x, wname, bias, in_tap, s_y, out_bits, in_bits=8):
        wq, s_w = self.qw[wname]
        s_x = self.scale(in_tap, in_bits)
        acc = gd.int_matmul_acc(gd.quantize_act(x, s_x, in_bits), wq)
        bq = gd.quantize_bias(bias, s_x, s_w) if bias is not None else None
        if bq is not None:
            acc = acc + bq[None, :]
        M0, shift = gd.fixed_point_multiplier(s_x * s_w.double() / s_y)
        y = gd.requantize(acc, M0, shift, out_bits)
        self.layers[wname] = dict(kind="int", in_bits=in_bits, s_x=s_x, s_y=float(s_y),
                                  out_bits=out_bits, bias=bq, M0=M0, shift=shift)
        self.tr[wname] = y
        return y

    def gemm(self, x, wname, bias, in_tap, in_bits=8, out_tap=None, out_bits=16):
        wq, s_w = self.qw[wname]
        s_x = self.scale(in_tap, in_bits)
        acc = gd.int_matmul_acc(gd.quantize_act(x, s_x, in_bits), wq)
        bq = gd.quantize_bias(bias, s_x, s_w) if bias is not None else None
        if bq is not None:
            acc = acc + bq[None, :]
        if out_tap is None:
            deq = s_x * s_w.double()
            self.layers[wname] = dict(kind="float", in_bits=in_bits, s_x=s_x, bias=bq, deq=deq)
            out = (acc.double() * deq[None, :]).float()
            self.tr[wname] = out
            return out
        s_y = self.scale(out_tap, out_bits)
        M0, shift = gd.fixed_point_multiplier(s_x * s_w.double() / s_y)
        yq = gd.requantize(acc, M0, shift, out_bits)
        self.layers[wname] = dict(kind="int", in_bits=in_bits, s_x=s_x, s_y=float(s_y),
                                  out_bits=out_bits, bias=bq, M0=M0, shift=shift)
        self.tr[wname] = yq
        return (yq.double() * s_y).float()

    def rope_int(self, xi):
        out = super().rope_int(xi)
        self.tr[f"rope{self._nrope}"] = out
        self._nrope += 1
        return out

    def attention_int(self, b, qh, kh, vh):
        ctx = super().attention_int(b, qh, kh, vh)
        self.tr[f"{b}.ctx_i"] = torch.round(ctx.double() / self.scale(f"{b}.ctx", 8)).to(torch.int64)
        return ctx

    def swiglu_from_int(self, b, gi, ui):
        h = super().swiglu_from_int(b, gi, ui)
        self.tr[f"{b}.swiglu_i"] = torch.round(h.double() / self.scale(f"{b}.swiglu_h", 16)).to(torch.int64)
        return h

    def forward_traced(self, img):
        self.tr, self._nrope = {}, 0
        out = self.forward(img)
        self.tr["logits"] = out
        return out, self.tr


def mult(m):
    """escalar flotante -> (M0, shift) del epilogo de punto fijo."""
    M0, sh = gd.fixed_point_multiplier(torch.tensor([m], dtype=torch.float64))
    return int(M0), int(sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--csv", default="split/val_holdout.csv")
    ap.add_argument("--n-vectors", type=int, default=20)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    params = to_device(ckpt["params"], dev)
    stats = json.load(open(args.calib))
    cfg = frozen.frozen_config()
    model = Tracer(params, stats, cfg, dev)
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.n_vectors]

    out = args.out
    (out / "vectors").mkdir(parents=True, exist_ok=True)

    # ---- vectores de prueba (tambien llena model.layers con la primera imagen)
    vec = []
    for i, r in enumerate(rows):
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(dev)
        with torch.inference_mode():
            logits, tr = model.forward_traced(img)
        img_np = img.cpu().numpy().astype("<f4")
        img_np.tofile(out / "vectors" / f"img_{i:02d}.f32")
        lg = logits.cpu().numpy().astype("<f4")
        entry = dict(index=i, filepath=r["filepath"], label=r["binary_label"],
                     input=f"vectors/img_{i:02d}.f32", input_shape=list(img_np.shape),
                     input_sha256=sha(img_np), logits=lg.tolist(), logits_hex=lg.view("<u4").tolist(),
                     stage_sha256={k: sha(v.cpu().numpy()) for k, v in tr.items()})
        if i == 0:
            keep = {k: v.cpu().numpy() for k, v in tr.items()
                    if k == "logits" or any(k.startswith(f"b{b}.") for b in FULL_TRACE_BLOCKS)}
            np.savez(out / "vectors" / "stages_img_00.npz", **keep)
            entry["stages_full"] = "vectors/stages_img_00.npz"
            entry["stages_full_keys"] = sorted(keep)
        vec.append(entry)
        print(f"  vector {i + 1}/{len(rows)}", flush=True)

    # ---- pesos y epilogos
    W = Packer(out / "weights_i8.bin")
    E32 = Packer(out / "epilogue_i32.bin")
    E64 = Packer(out / "epilogue_f64.bin")
    layers = {}
    order = ["patch_embed"] + [f"b{i}.{m}" for i in range(len(params["blocks"]))
                               for m in ("q_w", "k_w", "v_w", "proj_w", "fc1g_w", "fc1x_w", "fc2_w")] + ["head"]
    for name in order:
        L = model.layers[name]
        wq, s_w = model.qw[name]
        o, k = wq.shape
        d = dict(kind=L["kind"], out=o, in_=k, in_bits=L["in_bits"], s_x=L["s_x"],
                 weights=W.add(wq.cpu().numpy(), "<i1"))
        d["bias"] = E32.add(L["bias"].cpu().numpy(), "<i4") if L["bias"] is not None else None
        if L["kind"] == "int":
            d.update(s_y=L["s_y"], out_bits=L["out_bits"],
                     M0=E32.add(L["M0"].cpu().numpy(), "<i4"), shift=E32.add(L["shift"].cpu().numpy(), "<i4"))
        else:
            d["deq_f64"] = E64.add(L["deq"].cpu().numpy(), "<f8")
        layers[name] = d
    for p in (W, E32, E64):
        p.close()

    # ---- tablas y constantes escalares por bloque
    T = Packer(out / "tables.bin")
    P, c = cfg.softmax_bits, cfg
    tables = dict(exp=T.add(model.etab.cpu().numpy(), "<u2"), recip=T.add(model.rtab.cpu().numpy(), "<u2"))
    f = 2 ** model.ROPE_FRAC
    tables["rope_cos_q14"] = T.add(torch.round(params["rope_cos"].double() * f).long().cpu().numpy(), "<i2")
    tables["rope_sin_q14"] = T.add(torch.round(params["rope_sin"].double() * f).long().cpu().numpy(), "<i2")
    blocks = []
    hd = fr.HEAD_DIM
    for i, p in enumerate(params["blocks"]):
        b = f"b{i}"
        s_q, s_k, s_v, s_ctx = (model.scale(f"{b}.{t}", 8) for t in ("q", "k", "v", "ctx"))
        cs = s_q * s_k * hd ** -0.5
        tab, s_sg = nl.silu_table(model.scale(f"{b}.fc1g_out", c.silu_in_bits), c.silu_in_bits, c.sg_bits)
        s_u, s_h = model.scale(f"{b}.fc1x_out", 8), model.scale(f"{b}.swiglu_h", 16)
        blocks.append(dict(
            heads=p["q_w"].shape[0] // hd, neurons=p["fc1g_w"].shape[0],
            s_q=s_q, s_k=s_k, s_v=s_v, s_ctx=s_ctx,
            softmax_idx=dict(zip(("M0", "shift"), mult(cs / model.delta))),
            av_requant=dict(zip(("M0", "shift"), mult(s_v / 2 ** P / s_ctx))),
            silu_table=T.add(tab.cpu().numpy(), "<i2"), s_sg=s_sg,
            swiglu_requant=dict(zip(("M0", "shift"), mult(s_sg * s_u / s_h))),
            s_fc1g_out=model.scale(f"{b}.fc1g_out", c.silu_in_bits), s_fc1x_out=s_u, s_swiglu_h=s_h,
            s_proj_out=model.scale(f"{b}.proj_out", 16), s_fc2_out=model.scale(f"{b}.fc2_out", 16)))
    T.close()

    # ---- parametros flotantes del HPS
    H = Packer(out / "hps_f32.bin")
    hps = dict(cls=H.add(params["cls"].cpu().numpy().reshape(-1), "<f4"),
               pos=H.add(params["pos"].cpu().numpy().reshape(1025, -1), "<f4"),
               fcn_g=H.add(params["fcn_g"].cpu().numpy(), "<f4"), fcn_b=H.add(params["fcn_b"].cpu().numpy(), "<f4"))
    for i, p in enumerate(params["blocks"]):
        blocks[i]["hps"] = {k: H.add(p[k].cpu().numpy(), "<f4") for k in ("n1_g", "n1_b", "n2_g", "n2_b", "mlpn_g", "mlpn_b")}
    H.close()

    manifest = dict(
        name=out.name, model="Freepik EVA02-Base ViT, 448 px, podado (ver ckpt)",
        ckpt=args.ckpt, ckpt_sha256=frozen.sha256(args.ckpt), ckpt_dims=ckpt["dims"] if "dims" in ckpt else None,
        calib=args.calib, calib_sha256=frozen.sha256(args.calib), frozen_config=frozen.FROZEN,
        endianness="little", align=ALIGN, image=dict(size=448, patch=14, tokens=1025, channels=3),
        ln_eps=fr.LN_EPS, rope=dict(frac_bits=model.ROPE_FRAC, pre_shift=model.ROPE_PRE_SHIFT),
        exp_delta=model.delta, softmax_bits=P, silu_in_bits=c.silu_in_bits, sg_bits=c.sg_bits,
        score="softmax(logits)[1:].sum() >= 0.5 -> NSFW",
        files={n: dict(sha256=frozen.sha256(out / n)) for n in
               ("weights_i8.bin", "epilogue_i32.bin", "epilogue_f64.bin", "tables.bin", "hps_f32.bin")},
        layers=layers, tables=tables, hps=hps, blocks=blocks, vectors=vec)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    sizes = {n: (out / n).stat().st_size for n in manifest["files"]}
    print("tamanos (bytes):", sizes, "total", sum(sizes.values()))


if __name__ == "__main__":
    main()

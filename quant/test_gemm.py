"""
Prueba de UNA GEMM entera contra su version flotante, con entradas reales.

Para cada (bloque, GEMM) se toma la entrada flotante real del modelo de
referencia, se cuantiza con la escala calibrada, se ejecuta la GEMM entera y se
compara contra el resultado flotante exacto. Mide el error introducido por
cuantizar esa operacion en aislamiento (la entrada llega exacta, sin el error
acumulado de capas previas).

GEMMs probadas (entrada INT8, salida INT16, como fija el diseno):
    proj : ctx      -> proj_out   (proyeccion de salida de la atencion)
    fc2  : mlpn_out -> fc2_out

Uso (en el contenedor):
    python quant/test_gemm.py --csv split/val.csv --n-images 8 --percentile 99.9
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg
from transformers import AutoModelForImageClassification

import float_ref
import golden
import weights as qw
from calibrate import TIMM_NAME, to_device

# (nombre, tap de entrada, tap de salida, matriz, sesgo)
GEMMS = {
    "proj": ("ctx", "proj_out", "proj_w", "proj_b"),
    "fc2": ("mlpn_out", "fc2_out", "fc2_w", "fc2_b"),
}


def rel_l2(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="split/val.csv")
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 5, 11])
    ap.add_argument("--percentile", default="99.9")
    ap.add_argument("--calib", default="quant/calib/calib_stats.json")
    ap.add_argument("--out", default="quant/results/gemm_test.json", type=Path)
    args = ap.parse_args()

    stats = json.load(open(args.calib))
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    params = float_ref.extract_params(hf.timm_model)
    del hf
    qweights = qw.quantize_params(params)
    dev_params = to_device(params, "cuda" if torch.cuda.is_available() else "cpu")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.n_images]

    results = {}
    for r in rows:
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(dev)
        taps = {}

        def tap(name, t):
            if any(name == f"b{b}.{n}" for b in args.blocks
                   for g in GEMMS.values() for n in g[:2]):
                taps[name] = t.detach().cpu().double()

        with torch.inference_mode():
            float_ref.forward(img, dev_params, tap)

        for b in args.blocks:
            for gname, (tin, tout, wkey, bkey) in GEMMS.items():
                x, y_ref = taps[f"b{b}.{tin}"], taps[f"b{b}.{tout}"]
                wq, s_w = qweights[f"b{b}.{wkey}"]
                bias = params["blocks"][b][bkey].double()
                s_x = golden.act_scale(stats[f"b{b}.{tin}"][args.percentile]["mean"], 8)
                s_y = golden.act_scale(stats[f"b{b}.{tout}"][args.percentile]["mean"], 16)

                xq = golden.quantize_act(x, s_x, 8)
                yq = golden.int_gemm(xq, s_x, wq.cpu(), s_w.cpu(), bias, s_y, 16)
                y_fake = golden.fake_quant_gemm(xq, s_x, wq.cpu(), s_w.cpu(), bias, s_y, 16)

                y_int = yq.double() * s_y
                sat_in = (x.abs() > s_x * golden.qmax(8)).double().mean().item()
                sat_out = (yq.abs() == golden.qmax(16)).double().mean().item()
                res = results.setdefault(f"b{b}.{gname}", dict(
                    rel_l2=[], max_lsb_vs_fakequant=[], sat_in=[], sat_out=[]))
                res["rel_l2"].append(rel_l2(y_int, y_ref))
                res["max_lsb_vs_fakequant"].append((yq - y_fake).abs().max().item())
                res["sat_in"].append(sat_in)
                res["sat_out"].append(sat_out)

    summary = {k: {m: sum(v) / len(v) if m != "max_lsb_vs_fakequant" else max(v)
                   for m, v in d.items()} for k, d in results.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        percentile=args.percentile, n_images=len(rows), csv=args.csv,
        blocks=args.blocks, summary=summary), indent=1))

    print(f"percentil={args.percentile}  imagenes={len(rows)}  ({args.csv})")
    print(f"{'GEMM':10s} {'err rel L2':>11s} {'LSB vs fake':>12s} {'sat entrada':>12s} {'sat salida':>11s}")
    for k, v in summary.items():
        print(f"{k:10s} {v['rel_l2']*100:10.3f}% {v['max_lsb_vs_fakequant']:12.0f} "
              f"{v['sat_in']*100:11.3f}% {v['sat_out']*100:10.4f}%")


if __name__ == "__main__":
    main()

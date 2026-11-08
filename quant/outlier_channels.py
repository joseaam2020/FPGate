"""
Diagnostico: DONDE estan los valores atipicos de la entrada de fc2 (mlpn_out).

Para cada bloque mide, sobre N imagenes de calibracion (train):
  - cuantos canales (de 2048) concentran los valores > umbral (p99.9 calibrado);
  - cuantos tokens (de 1025) los concentran, y si el token 0 es el cls;
  - cuanto mayor es el maximo del canal peor que la mediana de maximos de canal;
  - si los canales atipicos son los MISMOS entre imagenes (Jaccard del top-10).

Uso (en el contenedor):
    python quant/outlier_channels.py --n-images 20
"""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg
from transformers import AutoModelForImageClassification

import float_ref
from calibrate import TIMM_NAME, to_device

TAP = "mlpn_out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="quant/calib/calib_images.csv")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--calib", default="quant/calib/calib_stats.json")
    ap.add_argument("--percentile", default="99.9")
    ap.add_argument("--out", default="quant/results/outlier_channels.json", type=Path)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = json.load(open(args.calib))
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    params = to_device(float_ref.extract_params(hf.timm_model), dev)
    del hf
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))

    with open(args.images, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    step = max(1, len(rows) // args.n_images)
    rows = rows[::step][: args.n_images]
    print(f"{len(rows)} imagenes de {args.images} (train)")

    nb = len(params["blocks"])
    ch_count = [torch.zeros(2048, device=dev) for _ in range(nb)]
    tok_count = [torch.zeros(1025, device=dev) for _ in range(nb)]
    ch_max = [torch.zeros(2048, device=dev) for _ in range(nb)]
    top10 = [[] for _ in range(nb)]
    cls_tok_outliers = [0.0] * nb
    total_out = [0.0] * nb

    for r in rows:
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(dev)
        cur = {}

        def tap(name, t):
            if name.endswith("." + TAP):
                cur[int(name.split(".")[0][1:])] = t.detach()

        with torch.inference_mode():
            float_ref.forward(img, params, tap)

        for b in range(nb):
            x = cur[b].abs()                      # (1025, 2048)
            thr = stats[f"b{b}.{TAP}"][args.percentile]["mean"]
            mask = x > thr
            ch_count[b] += mask.sum(0)
            tok_count[b] += mask.sum(1)
            cmax = x.amax(0)
            ch_max[b] = torch.maximum(ch_max[b], cmax)
            top10[b].append(set(torch.topk(cmax, 10).indices.tolist()))
            total_out[b] += mask.sum().item()
            cls_tok_outliers[b] += mask[0].sum().item()

    report = {}
    print(f"\n{'blq':>3s} {'#canales':>9s} {'top5 can':>9s} {'top20 can':>10s} "
          f"{'#tokens':>8s} {'top5 tok':>9s} {'cls':>6s} {'max/med canal':>14s} {'Jaccard top10':>14s}")
    for b in range(nb):
        tot = max(total_out[b], 1.0)
        cs, _ = torch.sort(ch_count[b], descending=True)
        ts, _ = torch.sort(tok_count[b], descending=True)
        cm = ch_max[b]
        jac = [len(a & c) / len(a | c) for a, c in combinations(top10[b], 2)]
        rep = dict(
            canales_con_atipicos=int((ch_count[b] > 0).sum()),
            top5_canales=float(cs[:5].sum() / tot),
            top20_canales=float(cs[:20].sum() / tot),
            tokens_con_atipicos=int((tok_count[b] > 0).sum()),
            top5_tokens=float(ts[:5].sum() / tot),
            share_token_cls=cls_tok_outliers[b] / tot,
            max_sobre_mediana_canal=float(cm.max() / cm.median()),
            jaccard_top10=sum(jac) / len(jac),
        )
        report[f"b{b}"] = rep
        print(f"{b:3d} {rep['canales_con_atipicos']:9d} {rep['top5_canales']*100:8.1f}% "
              f"{rep['top20_canales']*100:9.1f}% {rep['tokens_con_atipicos']:8d} "
              f"{rep['top5_tokens']*100:8.1f}% {rep['share_token_cls']*100:5.1f}% "
              f"{rep['max_sobre_mediana_canal']:14.1f} {rep['jaccard_top10']:14.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(
        tap=TAP, percentile=args.percentile, n_images=len(rows), per_block=report), indent=1))


if __name__ == "__main__":
    main()

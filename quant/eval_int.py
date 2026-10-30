"""
Evalua el modelo dorado (int_model) contra el flotante sobre imagenes de VALIDACION.

Para cada imagen guarda los 4 logits del flotante y de cada configuracion. Metricas
por configuracion (puntaje NSFW = low+medium+high, umbral 0.5, como el benchmark):
  exactitud, FN, FP, cambios de veredicto vs. flotante, error medio de logits.

Las imagenes son las primeras N de val.csv (el orden ya esta barajado con semilla fija
en make_split.py); la lista exacta se guarda en el CSV de salida.

Uso (en el contenedor):
    python quant/eval_int.py --n-images 100 --suite ablation --tag ablation_v1
"""

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg
from transformers import AutoModelForImageClassification

import float_ref
from calibrate import TIMM_NAME, to_device
from int_model import KINDS, IntModel, QConfig

INT16_MAX_PCT = {"swiglu_h": "100.0", "proj_out": "100.0", "fc2_out": "100.0"}

SUITES = {
    "smoke": [QConfig(name="all_p99.9")],
    "big": [
        QConfig(percentile="99.99", pct_override={**INT16_MAX_PCT, "mlpn_out": "100.0"},
                fc2_in_bits=16, name="best_fc2in16_max"),
        QConfig(percentile="99.99", pct_override=dict(INT16_MAX_PCT), name="best_all_int8"),
    ],
    "fix3": [
        QConfig(percentile="99.99", pct_override=dict(INT16_MAX_PCT), name="best_all"),
        QConfig(percentile="99.99", pct_override=dict(INT16_MAX_PCT), fc2_in_bits=16,
                name="best_fc2in16"),
        QConfig(percentile="99.99", pct_override={**INT16_MAX_PCT, "mlpn_out": "100.0"},
                fc2_in_bits=16, name="best_fc2in16_max"),
        QConfig(percentile="99.99", pct_override={**INT16_MAX_PCT, "mlpn_out": "100.0"},
                name="best_fc2in8_max"),
    ],
    "fix2": (
        [QConfig(percentile="99.99", pct_override=dict(INT16_MAX_PCT), name="best_all")]
        + [QConfig(quantize=(k,), percentile="99.99", pct_override=dict(INT16_MAX_PCT),
                   name=f"best_solo_{k}") for k in KINDS]
    ),
    "fix": [
        QConfig(pct_override={"swiglu_h": "100.0"}, name="swiglu_h_max"),
        QConfig(pct_override=dict(INT16_MAX_PCT), name="int16_max"),
        QConfig(percentile="99.99", pct_override=dict(INT16_MAX_PCT), name="int16_max_in99.99"),
    ],
    "ablation": (
        [QConfig(name="all_p99.9")]
        + [QConfig(quantize=(k,), name=f"solo_{k}") for k in KINDS]
        + [QConfig(percentile="99.99", name="all_p99.99"),
           QConfig(fc2_in_bits=16, name="all_p99.9_fc2in16")]
    ),
}


def nsfw_prob(logits):
    return torch.softmax(logits.float(), dim=-1)[1:].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="split/val.csv")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--suite", default="ablation", choices=SUITES)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--calib", default="quant/calib/calib_stats.json")
    ap.add_argument("--outdir", default="quant/results", type=Path)
    args = ap.parse_args()
    assert "test" not in Path(args.csv).name, "no se usa test para elegir configuraciones"

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = json.load(open(args.calib))
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    params = to_device(float_ref.extract_params(hf.timm_model), dev)
    del hf
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))
    models = {c.name: IntModel(params, stats, c, dev) for c in SUITES[args.suite]}

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.n_images]

    args.outdir.mkdir(parents=True, exist_ok=True)
    names = ["float"] + list(models)
    logits = {n: [] for n in names}
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(dev)
        with torch.inference_mode():
            logits["float"].append(float_ref.forward(img, params).cpu())
            for n, m in models.items():
                logits[n].append(m.forward(img).cpu())
        if i % 10 == 0:
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    labels = [1 if r["binary_label"] == "NSFW" else 0 for r in rows]
    pred = {n: [int(nsfw_prob(l) >= 0.5) for l in logits[n]] for n in names}

    summary = {}
    for n in names:
        tp = sum(p == 1 and y == 1 for p, y in zip(pred[n], labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(pred[n], labels))
        fp = sum(p == 1 and y == 0 for p, y in zip(pred[n], labels))
        tn = sum(p == 0 and y == 0 for p, y in zip(pred[n], labels))
        flips = sum(a != b for a, b in zip(pred[n], pred["float"]))
        err = (torch.stack(logits[n]) - torch.stack(logits["float"])).abs().mean().item()
        summary[n] = dict(acc=(tp + tn) / len(rows), fn=fn, fp=fp, flips_vs_float=flips,
                          mean_abs_logit_err=err)

    (args.outdir / f"{args.tag}.json").write_text(json.dumps(dict(
        csv=args.csv, n_images=len(rows), suite=args.suite, calib=args.calib,
        score="low+medium+high>=0.5", seconds=round(time.time() - t0, 1),
        configs={c.name: dict(quantize=list(c.quantize), percentile=c.percentile,
                              pct_override=c.pct_override, fc2_in_bits=c.fc2_in_bits)
                 for c in SUITES[args.suite]},
        summary=summary), indent=1))
    with open(args.outdir / f"{args.tag}_logits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "binary_label"] + [f"{n}_l{j}" for n in names for j in range(4)])
        for i, r in enumerate(rows):
            w.writerow([r["filepath"], r["binary_label"]]
                       + [f"{v:.6f}" for n in names for v in logits[n][i].tolist()])

    print(f"\n{len(rows)} imagenes de {args.csv}")
    print(f"{'config':22s} {'acc':>7s} {'FN':>4s} {'FP':>4s} {'flips':>6s} {'err logit':>10s}")
    for n, s in summary.items():
        print(f"{n:22s} {s['acc']*100:6.1f}% {s['fn']:4d} {s['fp']:4d} "
              f"{s['flips_vs_float']:6d} {s['mean_abs_logit_err']:10.4f}")


if __name__ == "__main__":
    main()

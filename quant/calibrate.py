"""
Calibracion de activaciones para DES-1.2.

Pasa N imagenes de TRAIN (nunca de val ni test) por el modelo de referencia en
flotante y registra, para cada tensor observado (ver los `tap` de float_ref),
varios percentiles del valor absoluto, por imagen. Luego agrega sobre las
imagenes (media y maximo). Guardar varios percentiles permite probar otros
(99.0, 99.99, max) sin repetir la corrida.

Muestreo: estratificado por clase original (5 clases), semilla fija.

Uso (dentro del contenedor):
    python quant/calibrate.py --train split/train.csv --n-per-class 51 \
        --out quant/calib
"""

import argparse
import csv
import json
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg
from transformers import AutoModelForImageClassification

import float_ref
import weights as qw

PERCENTILES = (99.0, 99.9, 99.99, 100.0)
TIMM_NAME = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"


def to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_device(v, device) for v in obj]
    return obj


def sample_images(csv_path, n_per_class, seed):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["class_name"]].append(r)
    rng = random.Random(seed)
    chosen = []
    for cls in sorted(by_class):
        chosen += rng.sample(by_class[cls], n_per_class)
    return chosen


def abs_percentiles(t):
    """Percentiles del |valor| de un tensor, sin torch.quantile (limite de tamano)."""
    a = t.detach().abs().flatten().float()
    a, _ = torch.sort(a)
    n = a.numel()
    out = {}
    for p in PERCENTILES:
        idx = min(n - 1, int(round(p / 100.0 * (n - 1))))
        out[str(p)] = a[idx].item()
    return out


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="split/train.csv")
    ap.add_argument("--n-per-class", type=int, default=51)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="quant/calib", type=Path)
    args = ap.parse_args()
    assert "train" in Path(args.train).name, "la calibracion solo usa TRAIN"

    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    params = float_ref.extract_params(hf.timm_model)
    del hf

    # --- pesos INT8: error de cuantizacion por matriz
    qweights = qw.quantize_params(params)
    werr = qw.weight_error_report(params, qweights)
    with open(args.out / "weight_quant_error.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "shape", "rel_error"])
        for name, shape, rel in werr:
            w.writerow([name, "x".join(map(str, shape)), f"{rel:.6f}"])
    print(f"error relativo de pesos: max={max(r[2] for r in werr):.4f} "
          f"media={sum(r[2] for r in werr)/len(werr):.4f}")

    params = to_device(params, device)
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))

    sample = sample_images(args.train, args.n_per_class, args.seed)
    print(f"calibrando con {len(sample)} imagenes de {args.train} en {device}")

    per_image = defaultdict(list)  # nombre -> lista de dicts de percentiles
    t0 = time.time()
    for n, r in enumerate(sample, 1):
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(device)

        def tap(name, t):
            per_image[name].append(abs_percentiles(t))

        with torch.inference_mode():
            float_ref.forward(img, params, tap)
        if n % 10 == 0:
            print(f"  {n}/{len(sample)}  ({time.time() - t0:.0f}s)", flush=True)

    stats = {}
    for name, lst in per_image.items():
        stats[name] = {
            p: {"mean": sum(d[p] for d in lst) / len(lst), "max": max(d[p] for d in lst)}
            for p in map(str, PERCENTILES)
        }

    (args.out / "calib_stats.json").write_text(json.dumps(stats, indent=1))
    (args.out / "calib_run.json").write_text(json.dumps({
        "seed": args.seed, "train_csv": args.train, "n_per_class": args.n_per_class,
        "n_images": len(sample), "percentiles": PERCENTILES, "device": device,
        "git_commit": git_commit(), "seconds": round(time.time() - t0, 1),
    }, indent=1))
    with open(args.out / "calib_images.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class_name", "binary_label"])
        for r in sample:
            w.writerow([r["filepath"], r["class_name"], r["binary_label"]])
    print(f"listo: {len(stats)} tensores observados -> {args.out}")


if __name__ == "__main__":
    main()

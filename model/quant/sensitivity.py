"""
DES-1.3 paso 1: sensibilidad del modelo flotante a la poda estructurada.

1. Importancia de cabezas y neuronas (calibracion: primeras 32 imagenes de calib_images.csv, train).
2. Para cada bloque y tipo (cabezas / neuronas) se quita el 25 % y el 50 % de lo MENOS importante
   y se mide el cambio en validacion (primeras N de val.csv). Ademas se prueba podar todos los
   bloques a la vez con la misma fraccion.

Metricas por configuracion: cambios de veredicto vs. el modelo completo, exactitud, FN, FP,
error medio de logits. Neuronas siempre en multiplos de 16.

Uso (en el contenedor):
    python quant/sensitivity.py --n-images 100 --tag sens_v1
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

import float_ref as fr
import prune
from calibrate import TIMM_NAME, to_device

NUM_BLOCKS, NUM_NEURONS = 12, 2048
FRACS = (0.25, 0.5)


def read_rows(path, n):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))[:n]


def block_inputs(img, params):
    x = prune.embed(img, params)
    xs = []
    for p in params["blocks"]:
        xs.append(x)
        x = fr.block(x, p, params["rope_sin"], params["rope_cos"])
    return xs


def make_plan(blocks, kinds, frac, imp):
    """Plan que quita `frac` de lo menos importante en `blocks` para los `kinds`."""
    plan = {}
    for b in blocks:
        hi, ni = imp[b]
        heads = list(range(fr.NUM_HEADS))
        neurons = torch.arange(NUM_NEURONS)
        if "h" in kinds:
            keep = fr.NUM_HEADS - round(frac * fr.NUM_HEADS)
            heads = sorted(hi.argsort(descending=True)[:keep].tolist())
        if "n" in kinds:
            keep = NUM_NEURONS - round(frac * NUM_NEURONS)
            assert keep % 16 == 0
            neurons = ni.argsort(descending=True)[:keep].sort().values
        plan[b] = (heads, neurons)
    return plan


def nsfw_prob(l):
    return torch.softmax(l.float(), -1)[1:].sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="split/val.csv")
    ap.add_argument("--calib", default="quant/calib/calib_images.csv")
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=32)
    ap.add_argument("--tag", default="sens_v1")
    ap.add_argument("--outdir", default="quant/results", type=Path)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    params = to_device(fr.extract_params(hf.timm_model), dev)
    del hf
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))
    load = lambda r: tf(Image.open(r["filepath"]).convert("RGB")).to(dev)  # noqa: E731

    # ---- 1. importancia
    sin, cos = params["rope_sin"], params["rope_cos"]
    hsum = torch.zeros(NUM_BLOCKS, fr.NUM_HEADS, device=dev)
    nsum = torch.zeros(NUM_BLOCKS, NUM_NEURONS, device=dev)
    crows = read_rows(args.calib, args.n_calib)
    with torch.inference_mode():
        for r in crows:
            for b, x in enumerate(block_inputs(load(r), params)):
                hi, ni = prune.importance(x, params["blocks"][b], sin, cos)
                hsum[b] += hi
                nsum[b] += ni
    imp = {b: ((hsum[b] / len(crows)).cpu(), (nsum[b] / len(crows)).cpu())
           for b in range(NUM_BLOCKS)}
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"{args.tag}_importance.json").write_text(json.dumps(
        {str(b): dict(heads=imp[b][0].tolist(), neurons=imp[b][1].tolist()) for b in imp}))
    print("importancia calculada", flush=True)

    # ---- 2. configuraciones
    configs = {}
    for b in range(NUM_BLOCKS):
        for kinds, tag in (("h", "heads"), ("n", "neurons")):
            for fr_ in FRACS:
                configs[f"b{b}_{tag}_{int(fr_ * 100)}"] = make_plan([b], kinds, fr_, imp)
    for kinds, tag in (("h", "heads"), ("n", "neurons"), ("hn", "both")):
        for fr_ in FRACS:
            configs[f"all_{tag}_{int(fr_ * 100)}"] = make_plan(range(NUM_BLOCKS), kinds, fr_, imp)

    vrows = read_rows(args.val, args.n_images)
    labels = [r["binary_label"] == "NSFW" for r in vrows]
    logits = {n: [] for n in ["full"] + list(configs)}
    t0 = time.time()
    with torch.inference_mode():
        for i, r in enumerate(vrows, 1):
            xs = block_inputs(load(r), params)
            logits["full"].append(prune.run_from(xs[0], params, 0, {}).cpu())
            for n, plan in configs.items():
                s = min(plan)
                logits[n].append(prune.run_from(xs[s], params, s, plan).cpu())
            if i % 10 == 0:
                print(f"  {i}/{len(vrows)}  ({time.time() - t0:.0f}s)", flush=True)

    full = torch.stack(logits["full"])
    pred_full = [nsfw_prob(l) >= 0.5 for l in full]
    summary = {}
    for n, ls in logits.items():
        L = torch.stack(ls)
        pred = [nsfw_prob(l) >= 0.5 for l in L]
        summary[n] = dict(
            acc=sum(p == y for p, y in zip(pred, labels)) / len(labels),
            fn=sum((not p) and y for p, y in zip(pred, labels)),
            fp=sum(p and (not y) for p, y in zip(pred, labels)),
            flips=sum(a != b for a, b in zip(pred, pred_full)),
            logit_err=(L - full).abs().mean().item())
    (args.outdir / f"{args.tag}.json").write_text(json.dumps(dict(
        val=args.val, n_images=len(vrows), n_calib=len(crows), fracs=FRACS,
        seconds=round(time.time() - t0, 1), summary=summary), indent=1))

    print(f"\nreferencia (modelo completo): acc {summary['full']['acc']*100:.1f}%  "
          f"FN {summary['full']['fn']}  FP {summary['full']['fp']}")
    print("\nerror medio de logits (cambios de veredicto) al quitar 25% / 50%")
    print(f"{'blq':>4s} | {'cabezas 25':>14s} {'cabezas 50':>14s} | {'neuronas 25':>14s} {'neuronas 50':>14s}")
    for b in range(NUM_BLOCKS):
        cells = [summary[f"b{b}_{k}_{p}"] for k in ("heads", "neurons") for p in (25, 50)]
        f = lambda s: f"{s['logit_err']:.3f} ({s['flips']})"  # noqa: E731
        print(f"{b:4d} | {f(cells[0]):>14s} {f(cells[1]):>14s} | {f(cells[2]):>14s} {f(cells[3]):>14s}")
    print("\ntodos los bloques a la vez")
    for n in summary:
        if n.startswith("all_"):
            s = summary[n]
            print(f"  {n:18s} acc {s['acc']*100:5.1f}%  FN {s['fn']:2d}  FP {s['fp']:2d}  "
                  f"flips {s['flips']:3d}  err {s['logit_err']:.3f}")


if __name__ == "__main__":
    main()

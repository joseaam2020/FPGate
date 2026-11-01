"""
Medicion FINAL en test del modelo entero congelado (frozen.py). Se corre una sola vez por
version congelada. Referencia: torch_results.csv (benchmark fp16, 504 imagenes).

Uso (en el contenedor):
    python quant/eval_test.py --tag test_final
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
import frozen
from calibrate import TIMM_NAME, to_device
from int_model import IntModel

REF_MODEL = "vit_freepik_eva02"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="split/test.csv")
    ap.add_argument("--ref", default="torch_results.csv")
    ap.add_argument("--tag", default="test_final")
    ap.add_argument("--outdir", default="quant/results", type=Path)
    ap.add_argument("--ckpt", default=None, help="punto de control podado (quant/ckpt/*.pt)")
    ap.add_argument("--calib", default=frozen.CALIB_STATS)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = json.load(open(args.calib))
    digest = frozen.sha256(args.calib)
    cfg = frozen.frozen_config()

    if args.ckpt:
        params = to_device(torch.load(args.ckpt, map_location="cpu")["params"], dev)
    else:
        hf = AutoModelForImageClassification.from_pretrained(
            "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
        params = to_device(float_ref.extract_params(hf.timm_model), dev)
        del hf
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))
    model = IntModel(params, stats, cfg, dev)

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    with open(args.ref, newline="", encoding="utf-8") as f:
        ref = {r["filepath"]: r for r in csv.DictReader(f) if r["model_name"] == REF_MODEL}
    assert all(r["filepath"] in ref for r in rows), "faltan imagenes en la referencia"

    out = []
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(dev)
        with torch.inference_mode():
            lg = model.forward(img).cpu()
        out.append(lg)
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)

    prob = [torch.softmax(l.float(), -1)[1:].sum().item() for l in out]
    y = [r["binary_label"] == "NSFW" for r in rows]
    pred = [p >= 0.5 for p in prob]
    pref = [ref[r["filepath"]]["predicted_label"] == "NSFW" for r in rows]
    n = len(rows)
    tp = sum(p and t for p, t in zip(pred, y))
    fn = sum((not p) and t for p, t in zip(pred, y))
    fp = sum(p and (not t) for p, t in zip(pred, y))
    tn = n - tp - fn - fp
    flips = sum(a != b for a, b in zip(pred, pref))
    ref_fn = sum((not p) and t for p, t in zip(pref, y))
    ref_fp = sum(p and (not t) for p, t in zip(pref, y))
    # FN por clase
    fn_by_class = {}
    for r, p, t in zip(rows, pred, y):
        if t and not p:
            fn_by_class[r["class_name"]] = fn_by_class.get(r["class_name"], 0) + 1
    # criterio de docs/e1_referencia.md, seccion 7: columna "INT8" o "INT8 podado"
    lim = dict(acc=472, fn=12, fp=20, flips=20) if args.ckpt else dict(acc=476, fn=11, fp=18, flips=10)
    crit = dict(acc_ok=(tp + tn) >= lim["acc"], fn_ok=fn <= lim["fn"], fp_ok=fp <= lim["fp"],
                flips_ok=flips <= lim["flips"])

    res = dict(csv=args.csv, ref=args.ref, ref_model=REF_MODEL, n_images=n,
               ckpt=args.ckpt, calib=args.calib, criterion_limits=lim,
               calib_sha256=digest, config=frozen.FROZEN, seconds=round(time.time() - t0, 1),
               int8=dict(acc=(tp + tn) / n, correct=tp + tn, fn=fn, fp=fp, flips_vs_ref=flips,
                         fn_by_class=fn_by_class),
               baseline=dict(acc=sum(p == t for p, t in zip(pref, y)) / n, fn=ref_fn, fp=ref_fp),
               criterion=crit, accepted=all(crit.values()))
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
    with open(args.outdir / f"{args.tag}_logits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "class_name", "binary_label", "l0", "l1", "l2", "l3",
                    "nsfw_prob", "ref_prob"])
        for r, l, p in zip(rows, out, prob):
            w.writerow([r["filepath"], r["class_name"], r["binary_label"]]
                       + [f"{v:.6f}" for v in l.tolist()]
                       + [f"{p:.6f}", ref[r["filepath"]]["nsfw_probability"]])

    print(json.dumps({k: res[k] for k in ("int8", "baseline", "criterion", "accepted")}, indent=1))


if __name__ == "__main__":
    main()

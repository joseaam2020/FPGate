"""
DES-1.3: una ronda de poda estructurada + reentrenamiento por destilacion.

  1. Parte de un modelo (el original o un punto de control `--start`).
  2. Poda hasta la fraccion objetivo (relativa al modelo ORIGINAL) segun la importancia
     calculada con el modelo actual sobre imagenes de calibracion (train).
  3. Mide en validacion (antes de reentrenar).
  4. Reentrena `--train-images` imagenes de train: el maestro es el modelo original sin podar
     (nunca cambia) y el alumno imita sus 4 logits (KL + 0.1 * MSE de logits). Sin etiquetas.
  5. Mide otra vez y guarda el punto de control (quant/ckpt/<tag>.pt) y los resultados.

La validacion de decision es val.csv[:--eval-images]; las ultimas 1000 imagenes de val
(split/val_holdout.csv) y test no se tocan.

Uso (en el contenedor):
    python quant/prune_train.py --heads-frac 0.25 --train-images 1000 --tag r1
"""

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg
from transformers import AutoModelForImageClassification

import float_ref as fr
import prune
from calibrate import TIMM_NAME, to_device

H0, N0, TOKENS, D = 12, 2048, 1025, 768


def read_rows(path, n=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows if n is None else rows[:n]


def dims(params):
    return [(p["q_w"].shape[0] // prune.HD, p["fc1g_w"].shape[0]) for p in params["blocks"]]


def macs(params):
    """MACs de los bloques (GEMM + atencion), 1025 tokens."""
    return sum(h * (4 * TOKENS * D * 64 + 2 * TOKENS * TOKENS * 64) + 3 * TOKENS * D * n
               for h, n in dims(params))


def prob(l):
    return torch.softmax(l.float(), -1)[1:].sum().item()


@torch.no_grad()
def evaluate(params, rows, load, teacher_logits=None):
    logits = [prune.forward_dense(load(r), params).cpu() for r in rows]
    labels = [r["binary_label"] == "NSFW" for r in rows]
    pred = [prob(l) >= 0.5 for l in logits]
    out = dict(acc=sum(p == y for p, y in zip(pred, labels)) / len(rows),
               fn=sum((not p) and y for p, y in zip(pred, labels)),
               fp=sum(p and (not y) for p, y in zip(pred, labels)))
    if teacher_logits is not None:
        pt = [prob(l) >= 0.5 for l in teacher_logits]
        out["flips_vs_teacher"] = sum(a != b for a, b in zip(pred, pt))
        out["logit_err"] = (torch.stack(logits) - torch.stack(teacher_logits)).abs().mean().item()
    return out, logits


@torch.no_grad()
def make_plan(params, calib_rows, load, heads_frac, neurons_frac, protect):
    """Plan {bloque: (cabezas_que_quedan, neuronas_que_quedan)} para llegar a la fraccion objetivo."""
    sin, cos = params["rope_sin"], params["rope_cos"]
    keep_h = H0 - round(heads_frac * H0)
    keep_n = N0 - round(neurons_frac * N0)
    assert keep_n % 16 == 0
    nb = len(params["blocks"])
    hsum = [0] * nb
    nsum = [0] * nb
    for r in calib_rows:
        x = prune.embed(load(r), params)
        for b, p in enumerate(params["blocks"]):
            hi, ni = prune.importance(x, p, sin, cos)
            hsum[b] = hsum[b] + hi
            nsum[b] = nsum[b] + ni
            x = prune.block_dense(x, p, sin, cos)
    plan = {}
    for b, (h, n) in enumerate(dims(params)):
        heads = hsum[b].argsort(descending=True)[:min(h, keep_h)].sort().values
        k = n if b in protect else min(n, keep_n)
        neurons = nsum[b].argsort(descending=True)[:k].sort().values
        plan[b] = (heads, neurons)
    return plan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="punto de control de partida (si no, el original)")
    ap.add_argument("--heads-frac", type=float, default=0.0)
    ap.add_argument("--neurons-frac", type=float, default=0.0)
    ap.add_argument("--protect-neurons", type=int, nargs="*", default=[1])
    ap.add_argument("--train-images", type=int, default=1000)
    ap.add_argument("--eval-images", type=int, default=500)
    ap.add_argument("--calib-images", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="quant/results", type=Path)
    ap.add_argument("--ckptdir", default="quant/ckpt", type=Path)
    args = ap.parse_args()
    assert args.eval_images <= 3145, "las ultimas 1000 imagenes de val estan reservadas"

    dev = "cuda"
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    original = to_device(fr.extract_params(hf.timm_model), dev)
    del hf
    teacher = prune.detach_params(original)
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))
    load = lambda r: tf(Image.open(r["filepath"]).convert("RGB")).to(dev)  # noqa: E731

    if args.start:
        ck = torch.load(args.start, map_location=dev)
        student = to_device(ck["params"], dev)
    else:
        student = prune.detach_params(original)
    full_macs = macs(original)

    eval_rows = read_rows("split/val.csv", args.eval_images)
    calib_rows = read_rows("quant/calib/calib_images.csv", args.calib_images)
    t0 = time.time()
    t_metrics, t_logits = evaluate(teacher, eval_rows, load)
    print(f"maestro (modelo original) en {len(eval_rows)} imagenes de val: {t_metrics}", flush=True)

    # ---- 2. poda
    plan = make_plan(student, calib_rows, load, args.heads_frac, args.neurons_frac,
                     set(args.protect_neurons))
    student = prune.apply_plan(student, plan)
    ops = macs(student) / full_macs
    print(f"dimensiones (cabezas, neuronas) por bloque: {dims(student)}")
    print(f"operaciones restantes: {ops * 100:.1f} % del original", flush=True)

    # ---- 3. antes de reentrenar
    before, _ = evaluate(student, eval_rows, load, t_logits)
    print("tras podar, sin reentrenar:", before, flush=True)

    # ---- 4. reentrenamiento por destilacion
    train_rows = random.Random(args.seed).sample(read_rows("split/train.csv"), args.train_images)
    student = prune.make_trainable(student)
    params = prune.trainable_tensors(student)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, args.train_images // args.accum)
    loss_hist, step = [], 0
    t1 = time.time()
    for i, r in enumerate(train_rows, 1):
        img = load(r)
        with torch.no_grad():
            tl = prune.forward_dense(img, teacher)
        sl = prune.forward_dense(img, student, ckpt=True)
        kl = F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1), reduction="sum")
        loss = kl + 0.1 * F.mse_loss(sl, tl)
        (loss / args.accum).backward()
        loss_hist.append(loss.item())
        if i % args.accum == 0:
            for g in opt.param_groups:   # decaimiento lineal hasta 10 % del lr
                g["lr"] = args.lr * (1 - 0.9 * step / total_steps)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        if i % 100 == 0:
            print(f"  entrenamiento {i}/{len(train_rows)}  perdida media(ult.100) "
                  f"{sum(loss_hist[-100:]) / 100:.4f}  ({time.time() - t1:.0f}s)", flush=True)
    opt.zero_grad(set_to_none=True)

    # ---- 5. despues
    after, a_logits = evaluate(student, eval_rows, load, t_logits)
    print("tras reentrenar:", after, flush=True)

    args.ckptdir.mkdir(parents=True, exist_ok=True)
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.save(dict(params=to_device(prune.detach_params(student), "cpu"), dims=dims(student),
                    args=vars(args) | {"outdir": str(args.outdir), "ckptdir": str(args.ckptdir)}),
               args.ckptdir / f"{args.tag}.pt")
    (args.outdir / f"prune_{args.tag}.json").write_text(json.dumps(dict(
        args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        eval_set=f"split/val.csv[:{args.eval_images}]", teacher=t_metrics,
        ops_fraction=ops, dims=dims(student), before_training=before, after_training=after,
        loss_first100=sum(loss_hist[:100]) / min(100, len(loss_hist)),
        loss_last100=sum(loss_hist[-100:]) / min(100, len(loss_hist)),
        seconds=round(time.time() - t0, 1)), indent=1))
    print(f"\nlisto: {args.ckptdir / (args.tag + '.pt')}")


if __name__ == "__main__":
    main()

"""
Perfil del original y de las rondas podadas: MACs, parametros, tamano y latencia de software.

Latencia = tiempo de float_ref.forward (flotante FP32, sin cuantizar) por imagen ya preprocesada,
mediana de N imagenes de val_holdout tras calentamiento; GPU con sincronizacion. NO es la latencia
del acelerador ni la del golden model entero (simulacion en PyTorch, ~1.7 s/img): mide cuanto baja el
costo de computo al podar, con el mismo codigo para ambos modelos.

Uso (en el contenedor):
    python quant/profile_model.py --rounds r1 r2 r3 r4 r5 r6 r7 r8 --lat unpruned r7
"""

import argparse
import csv
import glob
import json
import statistics
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
from prune_train import dims, macs

TOKENS, D = 1025, 768
TABLES = ("rope_sin", "rope_cos")     # tablas fijas, no son parametros entrenables


def total_macs(params):
    patch = (TOKENS - 1) * D * 3 * 14 * 14
    return macs(params) + patch + 4 * D


def n_params(params):
    n = 0

    def walk(o, key=""):
        nonlocal n
        if torch.is_tensor(o):
            n += 0 if key in TABLES else o.numel()
        elif isinstance(o, dict):
            for k, v in o.items():
                walk(v, k)
        elif isinstance(o, list):
            for v in o:
                walk(v, key)
    walk(params)
    return n


def int8_gemm_bytes(params):
    n = params["pe_w"].numel() + params["head_w"].numel()
    for p in params["blocks"]:
        n += sum(p[k].numel() for k in ("q_w", "k_w", "v_w", "proj_w", "fc1g_w", "fc1x_w", "fc2_w"))
    return n


def latency(params, imgs, dev, warm):
    ts = []
    with torch.inference_mode():
        for i, img in enumerate(imgs):
            img = img.to(dev)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            float_ref.forward(img, params)
            if dev == "cuda":
                torch.cuda.synchronize()
            if i >= warm:
                ts.append((time.perf_counter() - t) * 1000)
    return dict(median_ms=statistics.median(ts), mean_ms=statistics.mean(ts), min_ms=min(ts), n=len(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="*", default=[])
    ap.add_argument("--lat", nargs="*", default=["unpruned", "r7"], help="modelos con latencia: unpruned y/o rondas")
    ap.add_argument("--gpu-images", type=int, default=23)
    ap.add_argument("--cpu-images", type=int, default=6)
    ap.add_argument("--out", default="quant/results/profile.json")
    args = ap.parse_args()

    hf = AutoModelForImageClassification.from_pretrained("Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    models = {"unpruned": float_ref.extract_params(hf.timm_model)}
    del hf
    for r in args.rounds:
        ck = glob.glob(f"quant/ckpt/{r}_*.pt")
        assert len(ck) == 1, f"checkpoint de {r}: {ck}"
        models[r] = torch.load(ck[0], map_location="cpu")["params"]

    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))
    rows = list(csv.DictReader(open("split/val_holdout.csv", encoding="utf-8")))[: max(args.gpu_images, args.cpu_images)]
    imgs = [tf(Image.open(r["filepath"]).convert("RGB")) for r in rows]

    base = total_macs(models["unpruned"])
    prof = {}
    for name, p in models.items():
        d = dict(dims=[list(x) for x in dims(p)], gmacs=total_macs(p) / 1e9, ops_vs_original=total_macs(p) / base,
                 params_m=n_params(p) / 1e6, fp32_mb=n_params(p) * 4 / 2**20, int8_gemm_mb=int8_gemm_bytes(p) / 2**20)
        prof[name] = d
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name in args.lat:
        p = to_device(models[name], dev)
        float_ref.forward(imgs[0].to(dev), p)    # calentamiento extra (cuDNN, asignador)
        prof[name]["gpu_fp32" if dev == "cuda" else "cpu_fp32"] = latency(p, imgs[: args.gpu_images], dev, warm=3)
        prof[name]["lat_gpu_ms"] = prof[name].get("gpu_fp32", {}).get("median_ms")
        print(name, prof[name].get("gpu_fp32"), flush=True)
        del p
        torch.cuda.empty_cache()
    for name in args.lat:
        p = to_device(models[name], "cpu")
        prof[name]["cpu_fp32"] = latency(p, imgs[: args.cpu_images], "cpu", warm=1)
        print(name, "cpu", prof[name]["cpu_fp32"], flush=True)

    out = dict(device=dev, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else None,
               torch=torch.__version__, cpu_threads=torch.get_num_threads(),
               note="latencia = float_ref.forward FP32 por imagen preprocesada; no es el acelerador", models=prof)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {m: v for m, v in d.items() if m in ("gmacs", "ops_vs_original", "params_m", "fp32_mb", "int8_gemm_mb")}
                      for k, d in prof.items()}, indent=1))


if __name__ == "__main__":
    main()

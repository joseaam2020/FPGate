"""
Prueba aislada del softmax entero con puntajes de atencion REALES.

Para cada bloque se toman q,k (post-RoPE) del modelo flotante, se cuantizan a INT8
con la escala calibrada, se calcula S = q.k^T entero y se compara:
  ref_exacto : softmax en flotante sobre los puntajes SIN cuantizar
  ref_cuant  : softmax en flotante sobre los puntajes cuantizados (aisla la tabla)
  entero     : int_softmax con tablas
Metricas por fila, promedio sobre filas: L1 = sum|p - p_ref| (0 = identico, 2 = disjunto)
y error maximo por elemento.

Uso (en el contenedor):
    python quant/test_softmax.py --n-images 4
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
import golden as gd
import nonlinear as nl
from calibrate import TIMM_NAME, to_device

CONFIGS = [(n, t_max, P) for n in (256, 1024, 4096) for t_max in (16.0,) for P in (16, 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="split/val.csv")
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 5, 11])
    ap.add_argument("--percentile", default="99.99")
    ap.add_argument("--calib", default="quant/calib/calib_stats.json")
    ap.add_argument("--out", default="quant/results/softmax_test.json", type=Path)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = json.load(open(args.calib))
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    params = to_device(float_ref.extract_params(hf.timm_model), dev)
    del hf
    tf = create_transform(**resolve_data_config(get_pretrained_cfg(TIMM_NAME).__dict__))
    rtab = nl.recip_table(dev)
    etabs = {(n, t): nl.exp_table(n, t, dev) for n, t, _ in CONFIGS}

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.n_images]

    acc = {}
    for r in rows:
        img = tf(Image.open(r["filepath"]).convert("RGB")).to(dev)
        taps = {}

        def tap(name, t):
            if any(name in (f"b{b}.q", f"b{b}.k") for b in args.blocks):
                taps[name] = t.detach()

        with torch.inference_mode():
            float_ref.forward(img, params, tap)

        for b in args.blocks:
            q, k = taps[f"b{b}.q"], taps[f"b{b}.k"]
            s_q = gd.act_scale(stats[f"b{b}.q"][args.percentile]["mean"], 8)
            s_k = gd.act_scale(stats[f"b{b}.k"][args.percentile]["mean"], 8)
            qi, ki = gd.quantize_act(q, s_q, 8), gd.quantize_act(k, s_k, 8)
            S = nl.int_bmm(qi, ki.transpose(-1, -2))
            c = s_q * s_k * (q.shape[-1] ** -0.5)
            p_exact = torch.softmax(float_ref.attention_scores(q, k).double(), -1)
            p_quant = torch.softmax(S.double() * c, -1)
            for n, t_max, P in CONFIGS:
                etab, delta = etabs[(n, t_max)]
                p_int = torch.cat([nl.int_softmax(S[h:h + 1], c, P, etab, delta, rtab)
                                   for h in range(S.shape[0])])
                p = p_int.double() / 2 ** P
                for ref_name, ref in (("exacto", p_exact), ("cuant", p_quant)):
                    l1 = (p - ref).abs().sum(-1).mean().item()
                    mx = (p - ref).abs().max().item()
                    a = acc.setdefault((b, n, P, ref_name), dict(l1=[], max=[]))
                    a["l1"].append(l1)
                    a["max"].append(mx)
                row_sum = p.sum(-1).mean().item()
                acc.setdefault((b, n, P, "suma"), dict(l1=[], max=[]))["l1"].append(row_sum)

    out = {}
    print(f"percentil q/k={args.percentile}, {len(rows)} imagenes")
    print(f"{'blq':>3s} {'tabla':>6s} {'P':>3s} {'L1 vs exacto':>13s} {'L1 vs cuant':>12s} "
          f"{'max vs cuant':>13s} {'suma filas':>11s}")
    for b in args.blocks:
        for n, t_max, P in CONFIGS:
            g = lambda ref, m: sum(acc[(b, n, P, ref)][m]) / len(acc[(b, n, P, ref)][m])  # noqa: E731
            vals = dict(l1_exacto=g("exacto", "l1"), l1_cuant=g("cuant", "l1"),
                        max_cuant=g("cuant", "max"), suma=g("suma", "l1"))
            out[f"b{b}_n{n}_P{P}"] = vals
            print(f"{b:3d} {n:6d} {P:3d} {vals['l1_exacto']:13.5f} {vals['l1_cuant']:12.5f} "
                  f"{vals['max_cuant']:13.5f} {vals['suma']:11.4f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(percentile=args.percentile, n_images=len(rows),
                                        results=out), indent=1))


if __name__ == "__main__":
    main()

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import get_pretrained_cfg
from transformers import AutoModelForImageClassification

REPO = "Freepik/nsfw_image_detector"
TIMM_NAME = "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k"
LABELS = {0: "neutral", 1: "low", 2: "medium", 3: "high"}


def titulo(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="reporte_vit", type=Path)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"],
                    help="Freepik recomienda bf16; usa float32 para comparar con tu HW INT8")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dtype = getattr(torch, args.dtype)

    # ------------------------------------------------------------------ 1
    titulo("1. Carga del modelo y estructura general")
    model = AutoModelForImageClassification.from_pretrained(REPO, torch_dtype=dtype).eval()
    print("Clase del wrapper HF :", type(model).__name__)
    backbone = getattr(model, "timm_model", None)
    print("Atributo timm interno :", "timm_model" if backbone is not None else "NO ENCONTRADO")
    if backbone is None:
        print("Imprime `print(model)` y ajusta el nombre del atributo.")
        return
    print("Clase timm            :", type(backbone).__name__)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"Parametros totales    : {n_par/1e6:.2f} M  (~{n_par*2/1e6:.0f} MB en bf16/fp16)")
    (args.out / "arquitectura.txt").write_text(str(model))
    print("Arbol completo guardado en arquitectura.txt")

    # hiperparametros reales leidos del modelo (no de memoria)
    blocks = backbone.blocks
    b0 = blocks[0]
    hp = OrderedDict(
        depth=len(blocks),
        embed_dim=backbone.embed_dim,
        num_heads=b0.attn.num_heads,
        patch_size=backbone.patch_embed.patch_size,
        img_size=backbone.patch_embed.img_size,
        grid_size=backbone.patch_embed.grid_size,
        num_prefix_tokens=backbone.num_prefix_tokens,
        tiene_rope=getattr(backbone, "rope", None) is not None,
        mlp_clase=type(b0.mlp).__name__,
        global_pool=backbone.global_pool,
        tiene_fc_norm=not isinstance(backbone.fc_norm, nn.Identity),
        head=str(backbone.head),
    )
    for k, v in hp.items():
        print(f"  {k:20s}: {v}")
    n_tokens = backbone.patch_embed.grid_size[0] * backbone.patch_embed.grid_size[1] \
        + backbone.num_prefix_tokens
    print(f"  tokens por imagen   : {n_tokens}")
    print("\nBloque 0 (compara con la guia: norm1, attn, norm2, mlp):")
    print(b0)

    # ------------------------------------------------------------------ 2
    titulo("2. Preprocesamiento (lo que ocurre antes de la red)")
    cfg = get_pretrained_cfg(TIMM_NAME)
    data_cfg = resolve_data_config(cfg.__dict__)
    print("Config de datos resuelta:", json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in data_cfg.items()}, indent=2))
    processor = create_transform(**data_cfg)
    print("\nPipeline de transformaciones:")
    print(processor)
    img = Image.open(args.image).convert("RGB")
    print(f"\nImagen original : {img.size} (ancho, alto), modo {img.mode}")
    x = processor(img)
    print(f"Tensor de entrada: shape={tuple(x.shape)} dtype={x.dtype} "
          f"min={x.min():.3f} max={x.max():.3f} media={x.mean():.3f}")
    x = x.unsqueeze(0).to(dtype)

    # ------------------------------------------------------------------ 3
    titulo("3. Traza del forward: forma de cada tensor y operaciones por tipo")
    trazas, handles = [], []

    def hook(nombre):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            i = inp[0] if len(inp) and torch.is_tensor(inp[0]) else None
            trazas.append((nombre, type(mod).__name__,
                           tuple(i.shape) if i is not None else None,
                           tuple(o.shape) if torch.is_tensor(o) else None))
        return f

    for nombre, mod in backbone.named_modules():
        if len(list(mod.children())) == 0 or nombre.endswith(("attn", "mlp")) or nombre in ("patch_embed",):
            handles.append(mod.register_forward_hook(hook(nombre)))
    with torch.inference_mode():
        logits = model(x).logits
    for h in handles:
        h.remove()

    with open(args.out / "traza_forward.tsv", "w") as f:
        f.write("modulo\ttipo\tshape_entrada\tshape_salida\n")
        for t in trazas:
            f.write("\t".join(map(str, t)) + "\n")
    print(f"{len(trazas)} modulos ejecutados; detalle en traza_forward.tsv")
    print("Primeras entradas (patch_embed y bloque 0):")
    for t in trazas[:14]:
        print("  ", t)

    # ------------------------------------------------------------------ 4
    titulo("4. MACs por tipo de capa (lineales) para dimensionar el acelerador")
    macs = {}
    for nombre, mod in backbone.named_modules():
        if isinstance(mod, nn.Linear):
            macs[nombre] = mod.in_features * mod.out_features * n_tokens
    conv = backbone.patch_embed.proj
    k = conv.kernel_size[0] * conv.kernel_size[1] * conv.in_channels
    macs["patch_embed.proj(conv)"] = k * conv.out_channels * (n_tokens - backbone.num_prefix_tokens)

    por_tipo = {}
    for nombre, v in macs.items():
        clave = ("patch_embed" if "patch_embed" in nombre else
                 "cabeza" if nombre.startswith("head") else nombre.split(".", 2)[-1])
        por_tipo[clave] = por_tipo.get(clave, 0) + v
    total_lin = sum(macs.values())
    # atencion QK^T y AV (activacion x activacion) por bloque
    d, h = hp["embed_dim"], hp["num_heads"]
    macs_attn = 2 * n_tokens * n_tokens * d * hp["depth"]
    for clave, v in sorted(por_tipo.items(), key=lambda kv: -kv[1]):
        print(f"  {clave:30s} {v/1e9:8.3f} GMAC")
    print(f"  {'TOTAL capas lineales':30s} {total_lin/1e9:8.3f} GMAC")
    print(f"  {'QK^T + AV (12 bloques)':30s} {macs_attn/1e9:8.3f} GMAC")
    print(f"  {'TOTAL aprox.':30s} {(total_lin+macs_attn)/1e9:8.3f} GMAC "
          f"(atencion = {100*macs_attn/(total_lin+macs_attn):.1f}%)")
    print(f"  Matriz de atencion por cabeza: {n_tokens}x{n_tokens} = "
          f"{n_tokens*n_tokens/1e6:.2f} M elementos ({h} cabezas)")

    # ------------------------------------------------------------------ 5
    titulo("5. Postprocesamiento: logits -> probabilidades -> decision binaria")
    print("logits crudos       :", logits.float().squeeze().tolist())
    probs = F.softmax(logits.float(), dim=-1).squeeze()
    print("softmax por clase   :", {LABELS[i]: round(float(p), 4) for i, p in enumerate(probs)})
    # acumulado "al menos este nivel" (misma logica que el README de Freepik)
    cum, acc = {}, 0.0
    for j in range(len(probs) - 1, -1, -1):
        acc += float(probs[j])
        cum[LABELS[j]] = float(probs[j]) if j == 0 else acc
    print("acumulado (>= nivel):", {k: round(v, 4) for k, v in cum.items()})
    for nivel in ("low", "medium", "high"):
        print(f"  is_nsfw(nivel>={nivel}, umbral 0.5) -> {cum[nivel] >= 0.5}")
    print("\n>> Verifica en TU codigo de evaluacion cual de estos niveles usaste")
    print("   para convertir 4 clases a SFW/NSFW; cambia los fallos reportados.")

    # ------------------------------------------------------------------ 6
    titulo("6. Equivalencia: forward manual por etapas == model(x)")
    with torch.inference_mode():
        t = backbone.patch_embed(x)
        t, rot = backbone._pos_embed(t)
        for blk in backbone.blocks:
            t = blk(t, rope=rot)
        t = backbone.norm(t)
        t = backbone.forward_head(t, pre_logits=True)
        manual = backbone.head(t)
    err = (manual.float() - logits.float()).abs().max().item()
    print(f"Diferencia maxima manual vs model(x): {err:.3e}  (esperado ~0)")
    print("Si falla, tu version de timm usa otra firma; revisa timm/models/eva.py")

    # ------------------------------------------------------------------ 7
    titulo("7. Perfil de tiempo por operador (CPU)")
    from torch.profiler import ProfilerActivity, profile
    with torch.inference_mode(), profile(activities=[ProfilerActivity.CPU]) as prof:
        model(x)
    tabla = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    print(tabla)
    (args.out / "perfil_cpu.txt").write_text(tabla)
    print("Interpreta: aten::linear/addmm/mm = lo que va al acelerador;")
    print("            layer_norm, softmax, silu/gelu, mul = lo que queda en el HPS.")

    print(f"\nListo. Archivos en {args.out.resolve()}")


if __name__ == "__main__":
    main()
"""
Cuantizacion de pesos a INT8, simetrica y POR CANAL DE SALIDA (una escala por
fila de la matriz de pesos), segun el esquema de DES-1.2.

    w_q[c, :] = clamp(round(w[c, :] / scale[c]), -127, 127)
    scale[c]  = max|w[c, :]| / 127

El rango [-127, 127] (no -128) mantiene la simetria: el cero cae exacto en 0 y
el negado de cualquier peso no desborda.
"""

import torch

QMAX = 127


def quantize_per_channel(w: torch.Tensor):
    """w (out, in) -> (w_int8 (out, in), scale (out,) float32)."""
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    scale = amax / QMAX
    q = torch.round(w / scale[:, None]).clamp(-QMAX, QMAX).to(torch.int8)
    return q, scale.float()


def dequantize(q: torch.Tensor, scale: torch.Tensor):
    return q.float() * scale[:, None]


# Matrices que pasan por el acelerador (el resto: LayerNorm, sesgos, cls, pos,
# RoPE, se quedan en flotante / se aplican en el epilogo).
BLOCK_MATRICES = ("q_w", "k_w", "v_w", "proj_w", "fc1g_w", "fc1x_w", "fc2_w")


def quantize_params(params):
    """Devuelve dict {nombre: (w_int8, scale)} para todas las matrices del modelo."""
    out = {}
    pe = params["pe_w"]
    out["patch_embed"] = quantize_per_channel(pe.reshape(pe.shape[0], -1))
    for i, p in enumerate(params["blocks"]):
        for name in BLOCK_MATRICES:
            out[f"b{i}.{name}"] = quantize_per_channel(p[name])
    out["head"] = quantize_per_channel(params["head_w"])
    return out


def weight_error_report(params, qweights):
    """Error relativo ||W - dequant(Wq)|| / ||W|| por matriz."""
    rows = []
    for name, (q, s) in qweights.items():
        if name == "patch_embed":
            w = params["pe_w"].reshape(params["pe_w"].shape[0], -1)
        elif name == "head":
            w = params["head_w"]
        else:
            i, key = name.split(".", 1)
            w = params["blocks"][int(i[1:])][key]
        rel = ((w - dequantize(q, s)).norm() / w.norm()).item()
        rows.append((name, tuple(w.shape), rel))
    return rows

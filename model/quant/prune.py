"""
Poda estructurada del modelo FLOTANTE (DES-1.3).

Se quitan de verdad filas/columnas de las matrices (no se ponen en cero), porque la
LayerNorm interna del SwiGLU cambia sus estadisticos al tener menos canales:
  cabeza h        filas [64h, 64h+64) de Wq, Wk, Wv (y sesgos) y columnas de Wproj
  neurona j       fila j de Wfc1g, Wfc1x (y sesgos), columna j de Wfc2, y j de gamma/beta de la LN
Un plan es {bloque: (indices_de_cabezas_que_quedan, indices_de_neuronas_que_quedan)}; un
bloque ausente queda completo. Los bloques "densos" (tras podar) infieren su numero de
cabezas y de neuronas de la forma de sus pesos.
"""

import torch
from torch.utils.checkpoint import checkpoint

import float_ref as fr

HD = 64


def slice_block(p, heads, neurons):
    """Copia de los pesos de un bloque con solo las cabezas y neuronas indicadas."""
    dev = p["q_w"].device
    heads = torch.as_tensor(heads, device=dev, dtype=torch.long)
    nr = torch.as_tensor(neurons, device=dev, dtype=torch.long)
    rows = (heads[:, None] * HD + torch.arange(HD, device=dev)[None, :]).reshape(-1)
    return dict(
        n1_g=p["n1_g"].clone(), n1_b=p["n1_b"].clone(),
        q_w=p["q_w"][rows].clone(), q_b=p["q_b"][rows].clone(),
        k_w=p["k_w"][rows].clone(),
        v_w=p["v_w"][rows].clone(), v_b=p["v_b"][rows].clone(),
        proj_w=p["proj_w"][:, rows].clone(), proj_b=p["proj_b"].clone(),
        n2_g=p["n2_g"].clone(), n2_b=p["n2_b"].clone(),
        fc1g_w=p["fc1g_w"][nr].clone(), fc1g_b=p["fc1g_b"][nr].clone(),
        fc1x_w=p["fc1x_w"][nr].clone(), fc1x_b=p["fc1x_b"][nr].clone(),
        mlpn_g=p["mlpn_g"][nr].clone(), mlpn_b=p["mlpn_b"][nr].clone(),
        fc2_w=p["fc2_w"][:, nr].clone(), fc2_b=p["fc2_b"].clone(),
    )


def block_dense(x, p, sin, cos):
    """Un bloque cuyas dimensiones salen de la forma de sus pesos (0 cabezas / 0 neuronas permitido)."""
    nh = p["q_w"].shape[0] // HD
    if nh == 0:
        x = x + p["proj_b"]
    else:
        n1 = fr.layer_norm(x, p["n1_g"], p["n1_b"])
        q = fr.split_heads(fr.linear(n1, p["q_w"], p["q_b"]), nh)
        k = fr.split_heads(fr.linear(n1, p["k_w"], None), nh)
        v = fr.split_heads(fr.linear(n1, p["v_w"], p["v_b"]), nh)
        q, k = fr.apply_rope(q, sin, cos), fr.apply_rope(k, sin, cos)
        ctx = fr.merge_heads(fr.softmax(fr.attention_scores(q, k)) @ v)
        x = x + fr.linear(ctx, p["proj_w"], p["proj_b"])
    if p["fc1g_w"].shape[0] == 0:
        return x + p["fc2_b"]
    n2 = fr.layer_norm(x, p["n2_g"], p["n2_b"])
    g = fr.linear(n2, p["fc1g_w"], p["fc1g_b"])
    u = fr.linear(n2, p["fc1x_w"], p["fc1x_b"])
    h = fr.layer_norm(fr.silu(g) * u, p["mlpn_g"], p["mlpn_b"])
    return x + fr.linear(h, p["fc2_w"], p["fc2_b"])


def block_pruned(x, p, sin, cos, heads, neurons):
    return block_dense(x, slice_block(p, heads, neurons), sin, cos)


def embed(img, params):
    x = fr.patch_embed(img, params["pe_w"], params["pe_b"])
    return fr.add_cls_and_pos(x, params["cls"], params["pos"])


def head_out(x, params):
    return fr.head(x, params["fcn_g"], params["fcn_b"], params["head_w"], params["head_b"])


def forward_dense(img, params, ckpt=False):
    """Forward de un modelo (posiblemente podado) cuyos bloques son densos."""
    x = embed(img, params)
    sin, cos = params["rope_sin"], params["rope_cos"]
    for p in params["blocks"]:
        if ckpt and torch.is_grad_enabled():
            x = checkpoint(block_dense, x, p, sin, cos, use_reentrant=False)
        else:
            x = block_dense(x, p, sin, cos)
    return head_out(x, params)


def run_from(x, params, start, plan):
    """Ejecuta los bloques start..11 sobre la entrada x del bloque `start` (modelo completo + plan)."""
    sin, cos = params["rope_sin"], params["rope_cos"]
    for i in range(start, len(params["blocks"])):
        p = params["blocks"][i]
        if i in plan:
            x = block_pruned(x, p, sin, cos, *plan[i])
        else:
            x = fr.block(x, p, sin, cos)
    return head_out(x, params)


def apply_plan(params, plan):
    """Nuevo dict de parametros con los bloques del plan podados (copias; no toca `params`)."""
    out = dict(params)
    out["blocks"] = [slice_block(p, *plan[i]) if i in plan else {k: v.clone() for k, v in p.items()}
                     for i, p in enumerate(params["blocks"])]
    return out


def make_trainable(params):
    """Copia con requires_grad en todo salvo las tablas de RoPE."""
    def f(o, key=None):
        if isinstance(o, dict):
            return {k: f(v, k) for k, v in o.items()}
        if isinstance(o, list):
            return [f(v) for v in o]
        if torch.is_tensor(o) and key not in ("rope_sin", "rope_cos"):
            return torch.nn.Parameter(o.detach().clone())
        return o
    return f(params)


def trainable_tensors(params):
    out = []

    def f(o):
        if isinstance(o, dict):
            [f(v) for v in o.values()]
        elif isinstance(o, list):
            [f(v) for v in o]
        elif isinstance(o, torch.nn.Parameter):
            out.append(o)
    f(params)
    return out


def detach_params(params):
    def f(o):
        if isinstance(o, dict):
            return {k: f(v) for k, v in o.items()}
        if isinstance(o, list):
            return [f(v) for v in o]
        return o.detach().clone() if torch.is_tensor(o) else o
    return f(params)


@torch.no_grad()
def importance(x, p, sin, cos):
    """Importancia de cada cabeza y de cada neurona de UN bloque (denso) dada su entrada x.
    Cabeza: norma media de su aporte a la salida de proj.
    Neurona: |activacion tras la LN| media * norma de su columna de fc2."""
    nh = p["q_w"].shape[0] // HD
    n1 = fr.layer_norm(x, p["n1_g"], p["n1_b"])
    q = fr.split_heads(fr.linear(n1, p["q_w"], p["q_b"]), nh)
    k = fr.split_heads(fr.linear(n1, p["k_w"], None), nh)
    v = fr.split_heads(fr.linear(n1, p["v_w"], p["v_b"]), nh)
    q, k = fr.apply_rope(q, sin, cos), fr.apply_rope(k, sin, cos)
    ctx = fr.softmax(fr.attention_scores(q, k)) @ v                        # (H,N,hd)
    wp = p["proj_w"].reshape(p["proj_w"].shape[0], nh, HD)                 # (D,H,hd)
    head_imp = torch.einsum("hnd,ohd->hno", ctx, wp).norm(dim=-1).mean(-1)
    x = x + fr.linear(fr.merge_heads(ctx), p["proj_w"], p["proj_b"])
    n2 = fr.layer_norm(x, p["n2_g"], p["n2_b"])
    g = fr.linear(n2, p["fc1g_w"], p["fc1g_b"])
    u = fr.linear(n2, p["fc1x_w"], p["fc1x_b"])
    h = fr.layer_norm(fr.silu(g) * u, p["mlpn_g"], p["mlpn_b"])
    return head_imp, h.abs().mean(0) * p["fc2_w"].norm(dim=0)

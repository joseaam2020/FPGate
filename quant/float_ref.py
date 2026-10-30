"""
Modelo de referencia en flotante del ViT Freepik (EVA02-Base, parche 14, 448 px),
escrito como funciones sueltas sobre tensores, SIN nn.Module.

Es el esqueleto del modelo dorado de DES-1.2: cada operacion es una funcion con
entradas y salidas explicitas, para poder sustituirla despues por su version
entera y comparar pieza por pieza.

Detalles de esta variante (leidos de timm 1.0.30):
  - Sin layer-scale (gamma_1/gamma_2 = None).
  - Q, K, V son tres Linear separados (k_proj sin sesgo).
  - RoPE intercalado ([-x1, x0, -x3, x2, ...]) aplicado solo a tokens de parche.
  - Atencion con escala head_dim**-0.5 = 0.125 (12 cabezas de 64).
  - SwiGLU con LayerNorm de 2048 canales entre silu(g)*x y fc2.
  - Pooling: promedio de los 1024 tokens de parche (sin cls), luego fc_norm y head.

Uso (dentro del contenedor):
    python quant/float_ref.py --image <ruta> [--n-images 3]
"""

import argparse
import math

import torch
import torch.nn.functional as F

NUM_HEADS = 12
NUM_PREFIX = 1
LN_EPS = 1e-6


# --------------------------------------------------------------------------- ops
def patch_embed(img, w, b, patch=14):
    """img (3,H,W) -> tokens (N, D). Convolucion 14x14 stride 14 como GEMM."""
    cols = F.unfold(img.unsqueeze(0), kernel_size=patch, stride=patch)[0]  # (3*14*14, N)
    return cols.t() @ w.reshape(w.shape[0], -1).t() + b


def add_cls_and_pos(patches, cls_token, pos_embed):
    x = torch.cat([cls_token.reshape(1, -1), patches], dim=0)
    return x + pos_embed.reshape(x.shape)


def layer_norm(x, g, b, eps=LN_EPS):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + eps) * g + b


def linear(x, w, b=None):
    y = x @ w.t()
    return y if b is None else y + b


def rot_interleaved(x):
    """[x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(x, sin, cos, num_prefix=NUM_PREFIX):
    """x (H, N, hd); sin/cos (N-num_prefix, hd). Rota solo los tokens de parche."""
    pre, pat = x[:, :num_prefix], x[:, num_prefix:]
    pat = pat * cos + rot_interleaved(pat) * sin
    return torch.cat([pre, pat], dim=1)


def softmax(x):
    return torch.softmax(x, dim=-1)


def silu(x):
    return x * torch.sigmoid(x)


def split_heads(x, heads=NUM_HEADS):
    n, d = x.shape
    return x.reshape(n, heads, d // heads).transpose(0, 1)  # (H, N, hd)


def merge_heads(x):
    h, n, hd = x.shape
    return x.transpose(0, 1).reshape(n, h * hd)


def attention_scores(q, k):
    """QK^T escalado. q,k (H,N,hd) -> (H,N,N)"""
    return (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)


def _noop(name, t):
    pass


def attention(x, p, sin, cos, tap=_noop):
    """tap(nombre, tensor): punto de observacion para calibracion (no altera el calculo)."""
    q = split_heads(linear(x, p["q_w"], p["q_b"]))
    k = split_heads(linear(x, p["k_w"], None))
    v = split_heads(linear(x, p["v_w"], p["v_b"]))
    q, k = apply_rope(q, sin, cos), apply_rope(k, sin, cos)
    tap("q", q), tap("k", k), tap("v", v)
    probs = softmax(attention_scores(q, k))
    tap("probs", probs)
    ctx = merge_heads(probs @ v)
    tap("ctx", ctx)
    out = linear(ctx, p["proj_w"], p["proj_b"])
    tap("proj_out", out)
    return out


def swiglu(x, p, tap=_noop):
    g = linear(x, p["fc1g_w"], p["fc1g_b"])
    u = linear(x, p["fc1x_w"], p["fc1x_b"])
    tap("fc1g_out", g), tap("fc1x_out", u)
    h = silu(g) * u
    tap("swiglu_h", h)
    h = layer_norm(h, p["mlpn_g"], p["mlpn_b"])
    tap("mlpn_out", h)
    out = linear(h, p["fc2_w"], p["fc2_b"])
    tap("fc2_out", out)
    return out


def block(x, p, sin, cos, tap=_noop):
    t = lambda n, v: tap(n, v)  # noqa: E731
    n1 = layer_norm(x, p["n1_g"], p["n1_b"])
    t("ln1_out", n1)
    x = x + attention(n1, p, sin, cos, tap)
    n2 = layer_norm(x, p["n2_g"], p["n2_b"])
    t("ln2_out", n2)
    x = x + swiglu(n2, p, tap)
    t("resid_out", x)
    return x


def head(x, fc_norm_g, fc_norm_b, head_w, head_b, tap=_noop):
    pooled = layer_norm(x[NUM_PREFIX:].mean(0), fc_norm_g, fc_norm_b)
    tap("head_in", pooled)
    return linear(pooled, head_w, head_b)


def forward(img, params, tap=_noop):
    """tap(nombre, tensor); los bloques anteponen 'b{i}.' al nombre."""
    tap("patch_in", img)
    x = patch_embed(img, params["pe_w"], params["pe_b"])
    x = add_cls_and_pos(x, params["cls"], params["pos"])
    sin, cos = params["rope_sin"], params["rope_cos"]
    for i, p in enumerate(params["blocks"]):
        x = block(x, p, sin, cos, lambda n, v, i=i: tap(f"b{i}.{n}", v))
    return head(x, params["fcn_g"], params["fcn_b"], params["head_w"], params["head_b"], tap)


# ----------------------------------------------------------------- extraccion
def extract_params(timm_model):
    """Copia los pesos del modelo timm a un dict de tensores planos."""
    sd = {k: v.detach().float() for k, v in timm_model.state_dict().items()}
    rope = timm_model.rope.get_embed().detach().float()
    sin, cos = rope.chunk(2, -1)
    blocks = []
    for i in range(len(timm_model.blocks)):
        pre = f"blocks.{i}."
        g = lambda n: sd[pre + n]  # noqa: E731
        blocks.append(dict(
            n1_g=g("norm1.weight"), n1_b=g("norm1.bias"),
            q_w=g("attn.q_proj.weight"), q_b=g("attn.q_proj.bias"),
            k_w=g("attn.k_proj.weight"),
            v_w=g("attn.v_proj.weight"), v_b=g("attn.v_proj.bias"),
            proj_w=g("attn.proj.weight"), proj_b=g("attn.proj.bias"),
            n2_g=g("norm2.weight"), n2_b=g("norm2.bias"),
            fc1g_w=g("mlp.fc1_g.weight"), fc1g_b=g("mlp.fc1_g.bias"),
            fc1x_w=g("mlp.fc1_x.weight"), fc1x_b=g("mlp.fc1_x.bias"),
            mlpn_g=g("mlp.norm.weight"), mlpn_b=g("mlp.norm.bias"),
            fc2_w=g("mlp.fc2.weight"), fc2_b=g("mlp.fc2.bias"),
        ))
    return dict(
        pe_w=sd["patch_embed.proj.weight"], pe_b=sd["patch_embed.proj.bias"],
        cls=sd["cls_token"], pos=sd["pos_embed"],
        rope_sin=sin, rope_cos=cos,
        blocks=blocks,
        fcn_g=sd["fc_norm.weight"], fcn_b=sd["fc_norm.bias"],
        head_w=sd["head.weight"], head_b=sd["head.bias"],
    )


# ---------------------------------------------------------------- verificacion
def main():
    import csv
    from PIL import Image
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.models import get_pretrained_cfg
    from transformers import AutoModelForImageClassification

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="split/val.csv")
    ap.add_argument("--n-images", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hf = AutoModelForImageClassification.from_pretrained(
        "Freepik/nsfw_image_detector", torch_dtype=torch.float32).eval()
    tm = hf.timm_model
    params = extract_params(tm)
    cfg = get_pretrained_cfg("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k")
    tf = create_transform(**resolve_data_config(cfg.__dict__))

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[: args.n_images]

    worst = 0.0
    for r in rows:
        img = tf(Image.open(r["filepath"]).convert("RGB"))
        with torch.inference_mode():
            ref = hf(img.unsqueeze(0)).logits[0]
            mine = forward(img, params)
        err = (ref - mine).abs().max().item()
        worst = max(worst, err)
        print(f"{r['class_name']:9s} ref={ref.tolist()} mine={mine.tolist()} max|err|={err:.2e}")
    print(f"error maximo en logits: {worst:.2e}")
    assert worst < 1e-3, "el modelo de referencia no reproduce al original"


if __name__ == "__main__":
    main()

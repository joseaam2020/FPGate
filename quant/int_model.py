"""
Modelo dorado completo (etapa 1): las GEMM son enteras (golden.int_gemm); lo
demas (LayerNorm, RoPE, QK^T, softmax, AV, SiLU, residual) sigue en flotante.
Las etapas siguientes sustituyen esas piezas por sus versiones enteras.

Que se cuantiza (esquema de DES-1.2):
  entrada de cada GEMM            INT8 (escala = percentil calibrado / 127)
  proj_out y fc2_out              INT16 (saturado al rango calibrado), luego al flujo residual flotante
  swiglu_h = silu(g)*u            INT16 (idem), antes de la LayerNorm interna
  patch_embed, q/k/v, fc1g/fc1x, head   salida = acumulador reescalado, sin requantizar
                                   (no hay escala de salida calibrada para ellos todavia)

`QConfig` permite apagar GEMM individuales (ablacion), cambiar el percentil
global o de un tensor concreto, y pasar la entrada de fc2 a INT16.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import float_ref as fr
import golden as gd
import weights as qw

KINDS = ("patch", "qkv", "proj", "fc1", "fc2", "head")


@dataclass
class QConfig:
    quantize: tuple = KINDS                      # que GEMM son enteras
    percentile: str = "99.9"                     # percentil global de activaciones
    pct_override: dict = field(default_factory=dict)   # p. ej. {"mlpn_out": "99.99"}
    fc2_in_bits: int = 8                         # 8 o 16
    name: str = "all"

    def pct(self, tensor):
        return self.pct_override.get(tensor, self.percentile)


class IntModel:
    def __init__(self, params, stats, cfg: QConfig, device):
        self.p, self.stats, self.cfg, self.dev = params, stats, cfg, device
        self.qw = {k: (q.to(device), s.to(device))
                   for k, (q, s) in qw.quantize_params(_cpu(params)).items()}

    # escala de activacion para un tensor observado
    def scale(self, tap, bits):
        return gd.act_scale(self.stats[tap][self.cfg.pct(tap.split(".")[-1])]["mean"], bits)

    def gemm(self, x, wname, bias, in_tap, in_bits=8, out_tap=None, out_bits=16):
        wq, s_w = self.qw[wname]
        s_x = self.scale(in_tap, in_bits)
        xq = gd.quantize_act(x, s_x, in_bits)
        acc = gd.int_matmul_acc(xq, wq)
        if bias is not None:
            acc = acc + gd.quantize_bias(bias, s_x, s_w)[None, :]
        if out_tap is None:
            return (acc.double() * (s_x * s_w.double())[None, :]).float()
        s_y = self.scale(out_tap, out_bits)
        M0, shift = gd.fixed_point_multiplier(s_x * s_w.double() / s_y)
        yq = gd.requantize(acc, M0, shift, out_bits)
        return (yq.double() * s_y).float()

    def forward(self, img):
        P, c = self.p, self.cfg
        q = set(c.quantize)

        cols = F.unfold(img.unsqueeze(0), kernel_size=14, stride=14)[0].t()
        if "patch" in q:
            x = self.gemm(cols, "patch_embed", P["pe_b"], "patch_in")
        else:
            x = cols @ P["pe_w"].reshape(P["pe_w"].shape[0], -1).t() + P["pe_b"]
        x = fr.add_cls_and_pos(x, P["cls"], P["pos"])

        for i, p in enumerate(P["blocks"]):
            b = f"b{i}"
            n1 = fr.layer_norm(x, p["n1_g"], p["n1_b"])
            if "qkv" in q:
                qq = self.gemm(n1, f"{b}.q_w", p["q_b"], f"{b}.ln1_out")
                kk = self.gemm(n1, f"{b}.k_w", None, f"{b}.ln1_out")
                vv = self.gemm(n1, f"{b}.v_w", p["v_b"], f"{b}.ln1_out")
            else:
                qq = fr.linear(n1, p["q_w"], p["q_b"])
                kk = fr.linear(n1, p["k_w"], None)
                vv = fr.linear(n1, p["v_w"], p["v_b"])
            qh, kh, vh = fr.split_heads(qq), fr.split_heads(kk), fr.split_heads(vv)
            qh = fr.apply_rope(qh, P["rope_sin"], P["rope_cos"])
            kh = fr.apply_rope(kh, P["rope_sin"], P["rope_cos"])
            ctx = fr.merge_heads(fr.softmax(fr.attention_scores(qh, kh)) @ vh)
            if "proj" in q:
                a = self.gemm(ctx, f"{b}.proj_w", p["proj_b"], f"{b}.ctx",
                              out_tap=f"{b}.proj_out")
            else:
                a = fr.linear(ctx, p["proj_w"], p["proj_b"])
            x = x + a

            n2 = fr.layer_norm(x, p["n2_g"], p["n2_b"])
            if "fc1" in q:
                g = self.gemm(n2, f"{b}.fc1g_w", p["fc1g_b"], f"{b}.ln2_out")
                u = self.gemm(n2, f"{b}.fc1x_w", p["fc1x_b"], f"{b}.ln2_out")
                h = fr.silu(g) * u
                s_h = self.scale(f"{b}.swiglu_h", 16)       # resultado del SwiGLU en INT16
                h = (gd.quantize_act(h, s_h, 16).double() * s_h).float()
            else:
                g = fr.linear(n2, p["fc1g_w"], p["fc1g_b"])
                u = fr.linear(n2, p["fc1x_w"], p["fc1x_b"])
                h = fr.silu(g) * u
            h = fr.layer_norm(h, p["mlpn_g"], p["mlpn_b"])
            if "fc2" in q:
                m = self.gemm(h, f"{b}.fc2_w", p["fc2_b"], f"{b}.mlpn_out",
                              in_bits=c.fc2_in_bits, out_tap=f"{b}.fc2_out")
            else:
                m = fr.linear(h, p["fc2_w"], p["fc2_b"])
            x = x + m

        pooled = fr.layer_norm(x[fr.NUM_PREFIX:].mean(0, keepdim=True), P["fcn_g"], P["fcn_b"])
        if "head" in q:
            return self.gemm(pooled, "head", P["head_b"], "head_in")[0]
        return fr.linear(pooled, P["head_w"], P["head_b"])[0]


def _cpu(params):
    def f(o):
        if torch.is_tensor(o):
            return o.cpu()
        if isinstance(o, dict):
            return {k: f(v) for k, v in o.items()}
        if isinstance(o, list):
            return [f(v) for v in o]
        return o
    return f(params)

"""
Modelo dorado completo. Etapa 1: las GEMM son enteras (golden.int_gemm). Etapa 2
(banderas de QConfig): atencion (QK^T, softmax, AV) y SiLU tambien enteros
(nonlinear.py). LayerNorm, RoPE y el flujo residual siguen en flotante (HPS).

Que se cuantiza (esquema de DES-1.2):
  entrada de cada GEMM            INT8 (escala = percentil calibrado / 127)
  proj_out y fc2_out              INT16 (saturado al rango calibrado), luego al flujo residual flotante
  swiglu_h = silu(g)*u            INT16 (idem), antes de la LayerNorm interna
  patch_embed, q/k/v, fc1g/fc1x, head   salida = acumulador reescalado, sin requantizar

Etapa 2:
  int_scores   q, k (post-RoPE) a INT8; S = q.k^T entero (INT32), puntaje real = S * s_q*s_k/sqrt(d)
  int_softmax  softmax entero en 3 pasadas con tablas; probabilidad UQ0.P (P = softmax_bits)
  int_av       v a INT8; ctx = p @ v entero, reescalado a INT8 con la escala calibrada de ctx
  int_silu     g a `silu_in_bits`, SiLU por tabla (salida `sg_bits`), producto con u (INT8)
               y resultado del SwiGLU en INT16

`QConfig` permite apagar piezas (ablacion), cambiar el percentil global o de un tensor y
pasar la entrada de fc2 a INT16.
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

import float_ref as fr
import golden as gd
import nonlinear as nl
import weights as qw

KINDS = ("patch", "qkv", "proj", "fc1", "fc2", "head")


@dataclass
class QConfig:
    quantize: tuple = KINDS                      # que GEMM son enteras
    percentile: str = "99.9"                     # percentil global de activaciones
    pct_override: dict = field(default_factory=dict)   # p. ej. {"mlpn_out": "99.99"}
    fc2_in_bits: int = 8                         # 8 o 16
    # etapa 2
    int_scores: bool = False
    int_softmax: bool = False
    int_av: bool = False
    int_silu: bool = False
    int_io: bool = False                         # etapa 3: q,k,v,fc1g,fc1x salen requantizados
                                                 # (punto fijo) y RoPE entero; exige etapa 2 completa
    softmax_bits: int = 16                       # bits de la probabilidad (16 u 8)
    exp_entries: int = 1024                      # tamano de la tabla de exponencial
    exp_tmax: float = 16.0
    silu_in_bits: int = 8                        # bits de la entrada de la tabla SiLU
    sg_bits: int = 16                            # bits de la salida de la tabla SiLU
    name: str = "all"

    def pct(self, tensor):
        return self.pct_override.get(tensor, self.percentile)


class IntModel:
    def __init__(self, params, stats, cfg: QConfig, device):
        self.p, self.stats, self.cfg, self.dev = params, stats, cfg, device
        self.qw = {k: (q.to(device), s.to(device))
                   for k, (q, s) in qw.quantize_params(_cpu(params)).items()}
        self.etab, self.delta = nl.exp_table(cfg.exp_entries, cfg.exp_tmax, device)
        self.rtab = nl.recip_table(device)
        self._silu = {}

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

    def gemm_int(self, x, wname, bias, in_tap, s_y, out_bits, in_bits=8):
        """GEMM entera con epilogo de punto fijo; devuelve el entero saturado (escala s_y)."""
        wq, s_w = self.qw[wname]
        s_x = self.scale(in_tap, in_bits)
        acc = gd.int_matmul_acc(gd.quantize_act(x, s_x, in_bits), wq)
        if bias is not None:
            acc = acc + gd.quantize_bias(bias, s_x, s_w)[None, :]
        M0, shift = gd.fixed_point_multiplier(s_x * s_w.double() / s_y)
        return gd.requantize(acc, M0, shift, out_bits)

    # RoPE entero: q/k llegan en INT16 con escala s/2^ROPE_PRE_SHIFT (pre-RoPE), seno y
    # coseno en Q1.14; la salida es INT8 con la escala calibrada s (post-RoPE).
    ROPE_FRAC = 14
    ROPE_PRE_SHIFT = 7

    def rope_int(self, xi):
        """xi (H,N,hd) int64 INT16 -> (H,N,hd) INT8 post-RoPE. El token cls no rota."""
        if not hasattr(self, "_rope"):
            f = 2 ** self.ROPE_FRAC
            self._rope = (torch.round(self.p["rope_cos"].double() * f).long(),
                          torch.round(self.p["rope_sin"].double() * f).long())
        C, S = self._rope
        sh = self.ROPE_FRAC + self.ROPE_PRE_SHIFT
        pre, pat = xi[:, :fr.NUM_PREFIX], xi[:, fr.NUM_PREFIX:]
        pat = (pat * C + fr.rot_interleaved(pat) * S + (1 << (sh - 1))) >> sh
        pre = (pre + (1 << (self.ROPE_PRE_SHIFT - 1))) >> self.ROPE_PRE_SHIFT
        return torch.cat([pre, pat], dim=1).clamp(-127, 127)

    def attention_int(self, b, qh, kh, vh):
        """qh,kh,vh (H,N,hd) INT8 enteros -> ctx (N,D) flotante (= ctx_int * s_ctx)."""
        c, P = self.cfg, self.cfg.softmax_bits
        hd = qh.shape[-1]
        s_q, s_k, s_v = (self.scale(f"{b}.{t}", 8) for t in "qkv")
        S = nl.int_bmm(qh, kh.transpose(-1, -2))
        cs = s_q * s_k * hd ** -0.5
        p_int = torch.cat([nl.int_softmax(S[h:h + 1], cs, P, self.etab, self.delta, self.rtab)
                           for h in range(S.shape[0])])
        acc = nl.int_bmm(p_int, vh)
        s_ctx = self.scale(f"{b}.ctx", 8)
        ctx_i = nl.requantize_scalar(acc, s_v / 2 ** P / s_ctx, 8)
        return fr.merge_heads((ctx_i.double() * s_ctx).float())

    def swiglu_from_int(self, b, gi, ui):
        """gi: fc1g INT(silu_in_bits), ui: fc1x INT8 -> swiglu_h flotante (INT16 * s_h)."""
        c = self.cfg
        qm = gd.qmax(c.silu_in_bits)
        if b not in self._silu:
            self._silu[b] = nl.silu_table(self.scale(f"{b}.fc1g_out", c.silu_in_bits),
                                          c.silu_in_bits, c.sg_bits)
        tab, s_sg = self._silu[b]
        sg = tab.to(gi.device)[gi.long() + qm]
        s_u, s_h = self.scale(f"{b}.fc1x_out", 8), self.scale(f"{b}.swiglu_h", 16)
        h_i = nl.requantize_scalar(sg * ui.long(), s_sg * s_u / s_h, 16)
        return (h_i.double() * s_h).float()

    # ------------------------------------------------------------- atencion
    def attention_core(self, b, qh, kh, vh):
        """qh,kh,vh (H,N,hd) flotantes (q,k ya con RoPE) -> ctx (N, D) flotante."""
        c = self.cfg
        P = c.softmax_bits
        H = qh.shape[0]
        hd = qh.shape[-1]

        if c.int_scores:
            s_q, s_k = self.scale(f"{b}.q", 8), self.scale(f"{b}.k", 8)
            S = nl.int_bmm(gd.quantize_act(qh, s_q, 8), gd.quantize_act(kh, s_k, 8).transpose(-1, -2))
            cs = s_q * s_k * hd ** -0.5
        else:
            scores = fr.attention_scores(qh, kh)

        p_int = None
        if c.int_softmax:
            assert c.int_scores
            p_int = torch.cat([nl.int_softmax(S[h:h + 1], cs, P, self.etab, self.delta, self.rtab)
                               for h in range(H)])
            probs = (p_int.double() / 2 ** P).float()
        elif c.int_scores:
            probs = torch.softmax((S.double() * cs).float(), dim=-1)
        else:
            probs = fr.softmax(scores)

        if c.int_av:
            if p_int is None:
                p_int = torch.round(probs.double() * 2 ** P).clamp(0, 2 ** P - 1).to(torch.int64)
            s_v = self.scale(f"{b}.v", 8)
            vi = gd.quantize_act(vh, s_v, 8)
            acc = nl.int_bmm(p_int, vi)
            s_ctx = self.scale(f"{b}.ctx", 8)
            ctx_i = nl.requantize_scalar(acc, s_v / 2 ** P / s_ctx, 8)
            ctx = (ctx_i.double() * s_ctx).float()
        else:
            ctx = probs @ vh
        return fr.merge_heads(ctx)

    # ----------------------------------------------------------------- SwiGLU
    def swiglu_int(self, b, g, u):
        c = self.cfg
        qm = gd.qmax(c.silu_in_bits)
        s_g = self.scale(f"{b}.fc1g_out", c.silu_in_bits)
        if b not in self._silu:
            self._silu[b] = nl.silu_table(s_g, c.silu_in_bits, c.sg_bits)
        tab, s_sg = self._silu[b]
        sg = tab.to(g.device)[gd.quantize_act(g, s_g, c.silu_in_bits).long() + qm]
        s_u = self.scale(f"{b}.fc1x_out", 8)
        ui = gd.quantize_act(u, s_u, 8).long()
        s_h = self.scale(f"{b}.swiglu_h", 16)
        h_i = nl.requantize_scalar(sg * ui, s_sg * s_u / s_h, 16)
        return (h_i.double() * s_h).float()

    # ---------------------------------------------------------------- forward
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
            if c.int_io:
                assert {"qkv", "fc1", "proj", "fc2"} <= q and c.int_scores and c.int_softmax \
                    and c.int_av and c.int_silu
                s_q, s_k, s_v = (self.scale(f"{b}.{t}", 8) for t in "qkv")
                f = 2 ** self.ROPE_PRE_SHIFT
                qi = self.gemm_int(n1, f"{b}.q_w", p["q_b"], f"{b}.ln1_out", s_q / f, 16)
                ki = self.gemm_int(n1, f"{b}.k_w", None, f"{b}.ln1_out", s_k / f, 16)
                vi = self.gemm_int(n1, f"{b}.v_w", p["v_b"], f"{b}.ln1_out", s_v, 8)
                qh = self.rope_int(fr.split_heads(qi))
                kh = self.rope_int(fr.split_heads(ki))
                ctx = self.attention_int(b, qh, kh, fr.split_heads(vi))
            else:
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
                ctx = self.attention_core(b, qh, kh, vh)
            if "proj" in q:
                a = self.gemm(ctx, f"{b}.proj_w", p["proj_b"], f"{b}.ctx",
                              out_tap=f"{b}.proj_out")
            else:
                a = fr.linear(ctx, p["proj_w"], p["proj_b"])
            x = x + a

            n2 = fr.layer_norm(x, p["n2_g"], p["n2_b"])
            if c.int_io:
                gi = self.gemm_int(n2, f"{b}.fc1g_w", p["fc1g_b"], f"{b}.ln2_out",
                                   self.scale(f"{b}.fc1g_out", c.silu_in_bits), c.silu_in_bits)
                ui = self.gemm_int(n2, f"{b}.fc1x_w", p["fc1x_b"], f"{b}.ln2_out",
                                   self.scale(f"{b}.fc1x_out", 8), 8)
                h = self.swiglu_from_int(b, gi, ui)
            elif "fc1" in q:
                g = self.gemm(n2, f"{b}.fc1g_w", p["fc1g_b"], f"{b}.ln2_out")
                u = self.gemm(n2, f"{b}.fc1x_w", p["fc1x_b"], f"{b}.ln2_out")
                if c.int_silu:
                    h = self.swiglu_int(b, g, u)
                else:
                    h = fr.silu(g) * u
                    s_h = self.scale(f"{b}.swiglu_h", 16)     # resultado del SwiGLU en INT16
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

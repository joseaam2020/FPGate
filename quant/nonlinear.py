"""
Operaciones no lineales y de atencion con aritmetica entera, como las hara el
hardware. Cada funcion se ejecuta y se compara por separado (ver test_softmax.py).

Softmax entero en TRES PASADAS por fila:
  1. maximo de la fila (restarlo da exponentes <= 0, sin desbordes).
  2. exponencial por TABLA + suma: idx = round((max - s) * c / delta); e = EXP[idx]
     (UQ0.16, exp(-idx*delta)); suma en int32.
  3. normalizacion con RECIPROCO por tabla: se localiza el bit mas alto de la suma
     (k), se toman los 8 bits siguientes como indice de una tabla de 256 entradas
     R[i] ~ 2^15 / mantisa, y p = (e * R[i]) >> (15 + k - P). La salida es UQ0.P.

SiLU por tabla: la entrada se cuantiza a `in_bits` y la tabla tiene una entrada por
valor posible (2^in_bits - 1), con la salida ya en `out_bits`.
"""

import torch

import golden as gd

EXP_OUT_BITS = 16
RECIP_ENTRIES = 256


# ------------------------------------------------------------------- tablas
def exp_table(n_entries=1024, t_max=16.0, device="cpu"):
    """EXP[i] = round(exp(-i*delta) * (2^16-1)), i = 0..n_entries; EXP[n_entries] = 0
    (t >= t_max). Devuelve (tabla int64, delta)."""
    delta = t_max / n_entries
    i = torch.arange(n_entries + 1, dtype=torch.float64)
    tab = torch.round(torch.exp(-i * delta) * (2 ** EXP_OUT_BITS - 1)).to(torch.int64)
    tab[-1] = 0
    return tab.to(device), delta


def recip_table(device="cpu"):
    """R[i] = round(2^15 / (1 + (i + 0.5)/256)), i = 0..255, valores en (2^14, 2^15)."""
    i = torch.arange(RECIP_ENTRIES, dtype=torch.float64)
    return torch.round(2 ** 15 / (1.0 + (i + 0.5) / RECIP_ENTRIES)).to(torch.int64).to(device)


def silu_table(s_in, in_bits, out_bits):
    """Tabla de SiLU. Indice = g_int + qmax(in_bits). Devuelve (tabla int64, s_out)."""
    qm = gd.qmax(in_bits)
    xs = torch.arange(-qm, qm + 1, dtype=torch.float64) * s_in
    y = xs * torch.sigmoid(xs)
    s_out = y.abs().max().item() / gd.qmax(out_bits)
    return torch.round(y / s_out).to(torch.int64), s_out


# ------------------------------------------------------------ requantizacion
def requantize_scalar(acc, m, bits):
    """acc entero de cualquier forma; m escalar flotante (>0): y = round(acc*m), saturado."""
    M0, shift = gd.fixed_point_multiplier(torch.tensor([m], dtype=torch.float64))
    M0, shift = int(M0), int(shift)
    y = (acc * M0 + (1 << (shift - 1))) >> shift
    qm = gd.qmax(bits)
    return y.clamp(-qm, qm)


# ----------------------------------------------------------------- bmm entero
def int_bmm(a, b):
    """a (...,N,K) @ b (...,K,M), enteros -> int64 exacto (rango int32).

    float32 por trozos de K cuando cada suma parcial cabe en 2^24; si no, float64.
    Ambos caminos son identicos a un acumulador INT32.
    """
    amax, bmax = int(a.abs().max()), int(b.abs().max())
    K = a.shape[-1]
    chunk = (2 ** 24 - 1) // max(amax * bmax, 1)
    if chunk >= 64:
        chunk = min(chunk, K)
        acc = torch.zeros(*a.shape[:-1], b.shape[-1], dtype=torch.int64, device=a.device)
        for k0 in range(0, K, chunk):
            part = a[..., k0:k0 + chunk].float() @ b[..., k0:k0 + chunk, :].float()
            acc += part.to(torch.int64)
    else:
        acc = (a.double() @ b.double()).to(torch.int64)
    assert acc.abs().max().item() <= gd.INT32_MAX, "desbordamiento del acumulador INT32"
    return acc


# ------------------------------------------------------------------- softmax
def int_softmax(S, c, P, etab, delta, rtab):
    """Softmax entero por filas.

    S  (..., N) int64: puntajes enteros; valor real = S * c
    P  bits de la probabilidad de salida (UQ0.P: p_real = p_int / 2^P)
    Devuelve p_int (..., N) int64 en [0, 2^P - 1].
    """
    n_entries = etab.numel() - 1
    # pasada 1: maximo
    d = S.amax(-1, keepdim=True) - S                       # >= 0
    # pasada 2: exponencial por tabla + suma
    M0, shift = gd.fixed_point_multiplier(
        torch.tensor([c / delta], dtype=torch.float64))
    M0, shift = int(M0), int(shift)
    idx = ((d * M0 + (1 << (shift - 1))) >> shift).clamp(max=n_entries)
    e = etab[idx]
    ssum = e.sum(-1, keepdim=True)                          # >= 65535 (el maximo da 1.0)
    # pasada 3: normalizacion con reciproco
    k = torch.floor(torch.log2(ssum.double())).to(torch.int64)
    one = torch.ones_like(k)
    assert ((one << k) <= ssum).all() and (ssum < (one << (k + 1))).all()
    mant = (ssum >> (k - 8)) - RECIP_ENTRIES                # 8 bits tras el bit alto
    r = rtab[mant]
    sh = 15 + k - P
    p = (e * r + (torch.ones_like(sh) << (sh - 1))) >> sh
    return p.clamp(0, 2 ** P - 1)

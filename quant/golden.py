"""
Modelo dorado: operaciones con aritmetica ENTERA, tal como las hara el
acelerador. Cada funcion se puede ejecutar y comparar por separado.

Convenciones
  - Cuantizacion simetrica: real ~= q * scale, q en [-(2^(b-1)-1), 2^(b-1)-1].
  - Escala de activacion: scale = percentil_abs / qmax  (percentil de calibracion).
  - Pesos INT8 por canal de salida (ver weights.py).
  - Acumulador INT32: producto INT8*INT8 sumado sobre K. Se calcula con float64,
    que es EXACTO aqui (todo entero intermedio < 2^53) y por tanto identico bit
    a bit a un acumulador INT32; se verifica que no desborde 2^31.
  - Reescalado (epilogo): y = round(acc * m[c]) con m[c] = s_x * s_w[c] / s_y,
    implementado en punto fijo: multiplicador entero M0 (31 bits) y desplazamiento
    `shift`, con redondeo al mas cercano (mitad hacia +inf) y saturacion.
"""

import math

import torch

INT32_MAX = 2**31 - 1


def qmax(bits: int) -> int:
    return 2 ** (bits - 1) - 1


def act_scale(abs_percentile: float, bits: int) -> float:
    return abs_percentile / qmax(bits)


def quantize_act(x: torch.Tensor, scale: float, bits: int = 8) -> torch.Tensor:
    """float -> entero con signo (int32 como contenedor), saturado a +-qmax."""
    qm = qmax(bits)
    return torch.round(x / scale).clamp(-qm, qm).to(torch.int32)


def dequantize_act(q: torch.Tensor, scale: float) -> torch.Tensor:
    return q.to(torch.float64) * scale


torch.backends.cuda.matmul.allow_tf32 = False  # TF32 romperia la exactitud entera

_FP32_CHUNK = 1024  # 127*127*1024 = 16 516 096 < 2^24: sumas exactas en float32


def int_matmul_acc(xq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """xq (N,K) int, wq (M,K) int8 -> acumulador (N,M) como int64 (rango int32).

    Si ambas entradas caben en INT8 se multiplica en float32 por trozos de K=1024
    (cada suma parcial es un entero < 2^24, exacto en float32) y se acumula en
    int64. Con entradas de 16 bits se usa float64 (exacto hasta 2^53).
    Ambos caminos son identicos bit a bit a un acumulador INT32.
    """
    if xq.abs().max().item() <= 127 and wq.abs().max().item() <= 127:
        xf, wf = xq.to(torch.float32), wq.to(torch.float32)
        acc = torch.zeros(xq.shape[0], wq.shape[0], dtype=torch.int64, device=xq.device)
        for k0 in range(0, xq.shape[1], _FP32_CHUNK):
            part = xf[:, k0:k0 + _FP32_CHUNK] @ wf[:, k0:k0 + _FP32_CHUNK].t()
            acc += part.to(torch.int64)
    else:
        acc = (xq.to(torch.float64) @ wq.to(torch.float64).t()).to(torch.int64)
    assert acc.abs().max().item() <= INT32_MAX, "desbordamiento del acumulador INT32"
    return acc


def quantize_bias(bias: torch.Tensor, s_x: float, s_w: torch.Tensor) -> torch.Tensor:
    """Sesgo en el dominio del acumulador: round(b / (s_x * s_w[c])) como int32."""
    b = torch.round(bias.to(torch.float64) / (s_x * s_w.to(torch.float64)))
    b = b.to(torch.int64)
    assert b.abs().max().item() <= INT32_MAX, "sesgo fuera de rango int32"
    return b


def fixed_point_multiplier(m: torch.Tensor):
    """m (float, >0) -> (M0 int en [2^30, 2^31), shift) con m ~= M0 * 2^-shift."""
    m = m.to(torch.float64)
    mant, exp = torch.frexp(m)  # m = mant * 2^exp, mant en [0.5, 1)
    M0 = torch.round(mant * 2**31).to(torch.int64)
    shift = 31 - exp.to(torch.int64)
    # mant redondeado a 1.0 -> renormalizar
    over = M0 == 2**31
    M0 = torch.where(over, M0 // 2, M0)
    shift = torch.where(over, shift - 1, shift)
    return M0, shift


def requantize(acc: torch.Tensor, M0: torch.Tensor, shift: torch.Tensor, bits: int):
    """acc (N,M) int64, M0/shift (M,) -> entero saturado a +-qmax(bits).

    y = (acc * M0 + 2^(shift-1)) >> shift   (redondeo mitad hacia +inf, shift aritmetico)
    acc (<=2^31) * M0 (<2^31) cabe en int64.
    """
    prod = acc * M0[None, :]
    rnd = torch.ones_like(shift) << (shift - 1)
    y = (prod + rnd[None, :]) >> shift[None, :]
    qm = qmax(bits)
    return y.clamp(-qm, qm)


def int_gemm(xq, s_x, wq, s_w, bias, s_y, out_bits):
    """GEMM entera completa con epilogo.

    xq   (N,K) entero de activacion (escala s_x)
    wq   (M,K) int8 (escala por canal s_w (M,))
    bias (M,) flotante o None
    s_y  escala de salida; out_bits 8 o 16
    Devuelve entero (N,M) saturado.
    """
    acc = int_matmul_acc(xq, wq)
    if bias is not None:
        acc = acc + quantize_bias(bias, s_x, s_w)[None, :]
        assert acc.abs().max().item() <= INT32_MAX, "desbordamiento tras el sesgo"
    m = s_x * s_w.to(torch.float64) / s_y
    M0, shift = fixed_point_multiplier(m)
    return requantize(acc, M0, shift, out_bits)


def fake_quant_gemm(xq, s_x, wq, s_w, bias, s_y, out_bits):
    """Referencia 'en flotante' de lo mismo (sin punto fijo), para verificar que el
    camino entero coincide: difiere en 1-2 LSB (redondeo del multiplicador fijo y
    del sesgo en el dominio del acumulador); en INT16 eso es ~0.003 % del rango."""
    x = dequantize_act(xq, s_x)
    w = wq.to(torch.float64) * s_w.to(torch.float64)[:, None]
    y = x @ w.t()
    if bias is not None:
        y = y + bias.to(torch.float64)
    qm = qmax(out_bits)
    return torch.round(y / s_y).clamp(-qm, qm).to(torch.int64)

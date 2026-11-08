"""
Configuracion CONGELADA del modelo entero de E1 (INT8, sin podar). Unica fuente de verdad:
eval_test.py y el exportador deben construir el modelo con `frozen_config()`.

Cualquier cambio aqui invalida las medidas en test y obliga a repetirlas.
"""

import hashlib

from int_model import QConfig

CALIB_STATS = "quant/calib/calib_stats.json"   # 255 imagenes de train, semilla 0

FROZEN = dict(
    percentile="99.99",                       # entradas INT8 de las GEMM y q,k,v,ctx
    pct_override={"swiglu_h": "100.0", "proj_out": "100.0", "fc2_out": "100.0",
                  "mlpn_out": "100.0"},       # tensores INT16 (y entrada de fc2): rango maximo
    fc2_in_bits=16,
    int_scores=True, int_softmax=True, int_av=True, int_silu=True, int_io=True,
    softmax_bits=16, exp_entries=1024, exp_tmax=16.0,
    silu_in_bits=8, sg_bits=16,
)


def frozen_config(name="frozen"):
    return QConfig(name=name, **FROZEN)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

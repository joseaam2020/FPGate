#!/usr/bin/env bash
# Evaluacion estandar de un punto de control podado (protocolo de context.md, seccion 6):
#   1. calibra con 255 imagenes de train si falta quant/calib_<corto>/
#   2. evalua el modelo entero congelado en val[:500] contra el maestro
# Uso (desde la raiz del repo): quant/run_round_eval.sh <corto> <tag-del-ckpt>
#   p. ej. quant/run_round_eval.sh r3 r3_heads25_neur25
set -euo pipefail
short=$1
tag=$2
run="docker compose run --rm -e PYTHONUNBUFFERED=1 torch"

if [ ! -f "quant/calib_${short}/calib_stats.json" ]; then
    $run quant/calibrate.py --ckpt "quant/ckpt/${tag}.pt" --out "quant/calib_${short}" \
        > "quant/logs/calib_${short}.log" 2>&1
fi
$run quant/eval_int.py --suite frozen --ckpt "quant/ckpt/${tag}.pt" \
    --calib "quant/calib_${short}/calib_stats.json" --n-images 500 --tag "std500_${short}" \
    > "quant/logs/std500_${short}.log" 2>&1
tail -6 "quant/logs/std500_${short}.log"

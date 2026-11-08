# Comparación de resultados E1 (generado por quant/compare.py)

Solo lee archivos de resultados. Las tres muestras son distintas y no se mezclan.

## A. Rondas de poda — val[:500], modelo entero congelado
Original flotante (maestro) en la misma muestra: acc 95.0%, FN 17, FP 8. Peso INT8 del original (solo GEMM): 81.4 MB. Cabezas y neuronas = valores por bloque (neuronas distintas por el bloque 1 protegido).

| Ronda | Cabezas | Neuronas | Ops (% orig.) | Peso INT8 (MB) | Acc | FN | FP | Cambios vs. maestro | Err. logits | Imgs. entren. |
|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 9 | 2048 | 88.6% | 74.7 | 94.4% | 19 | 9 | 7 | 0.608 | 1000 |
| r2 | 9 | 2048 | 88.6% | 74.7 | 93.6% | 21 | 11 | 11 | 0.602 | 3000 |
| r3 | 9 | 1536/2048 | 76.1% | 62.3 | 94.2% | 15 | 14 | 22 | 0.717 | 3000 |
| r4 | 9 | 1536/2048 | 76.1% | 62.3 | 94.4% | 17 | 11 | 15 | 0.637 | 5000 |
| r5 | 9 | 1232/2048 | 68.7% | 55.0 | 94.8% | 18 | 8 | 13 | 0.688 | 3000 |
| r6 | 9 | 928/2048 | 61.3% | 47.6 | 93.0% | 23 | 12 | 20 | 0.760 | 3000 |
| r7 | 8 | 1232/2048 | 64.9% | 52.7 | 95.2% | 15 | 9 | 9 | 0.705 | 3000 |
| r8 | 6 | 1232/2048 | 57.3% | 48.2 | 93.8% | 16 | 15 | 20 | 0.823 | 3000 |

## B. Holdout — 1000 imágenes nunca usadas, modelo entero congelado

| Modelo | Ops | Acc | FN | FP | Cambios vs. original | FN por clase | FP por clase |
|---|---|---|---|---|---|---|---|
| Original flotante | 100.0% | 95.4% | 17 | 29 | 0 | hentai 17 | neutral 15, drawings 14 |
| INT8 sin podar | 100.0% | 95.6% | 19 | 25 | 8 | hentai 19 | drawings 13, neutral 12 |
| r5 INT8 | 68.7% | 95.2% | 26 | 22 | 32 | hentai 23, porn 2, sexy 1 | drawings 17, neutral 5 |
| r7 INT8 | 64.9% | 95.3% | 19 | 28 | 33 | hentai 18, sexy 1 | drawings 22, neutral 6 |

## C. Test — 504 imágenes (LlavaGuard: 159)

| Modelo | N | Acc | FN | FP | Cambios vs. benchmark | FN por clase | FP por clase | Ops | Peso (MB) | Latencia mediana (ms) | Cumple criterio E1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| clip_laion_probed | 504 | 96.83% | 12 | 4 | — | — | — | — | 1710.5 | 2956 | — |
| cnn_gantman_mobilenetv2 | 504 | 92.26% | 31 | 8 | — | — | — | — | — | 164 | — |
| vit_freepik_eva02 | 504 | 95.24% | 9 | 15 | — | — | — | — | 172.7 | 674 | — |
| vlm_llavaguard | 159 | 85.53% | 21 | 2 | — | — | — | — | 1787.5 | 36716 | — |
| ViT INT8 sin podar | 504 | 95.24% | 10 | 14 | 2 | hentai 8, porn 1, sexy 1 | neutral 9, drawings 5 | 100.0% | — | — | sí |
| ViT r7 INT8 podado | 504 | 95.04% | 15 | 10 | 25 | hentai 14, porn 1 | neutral 7, drawings 3 | 64.9% | 57.0 | — | no (excepción §8) |

Peso: benchmark = peso del modelo en disco (HF); r7 = paquete exportado (INT8 + tablas + parámetros del HPS). Latencia: del benchmark (GPU, fp16); la de r7 no es comparable, ver tabla D. Un guion indica dato no medido.

## D. Perfil de cómputo y tamaño (original y rondas)
Latencia: float_ref FP32 por imagen, mediana (NVIDIA GeForce GTX 1650, CPU 8 hilos). No es la latencia del acelerador; mide cuánto baja el cómputo al podar. Ops aquí incluye patch_embed y cabeza (r7: 65.1 %; las tablas A y C cuentan solo los bloques: 64.9 %). Peso INT8 = solo matrices GEMM.

| Modelo | GMACs | Ops | Parámetros (M) | FP32 (MB) | INT8 GEMM (MB) | Lat. GPU (ms) | Δ GPU | Lat. CPU (ms) | Δ CPU |
|---|---|---|---|---|---|---|---|---|---|
| original | 106.9 | 100.0% | 86.4 | 329.4 | 81.4 | 207 | 0% | 2084 | 0% |
| r1 | 94.8 | 88.7% | 79.3 | 302.4 | 74.7 | — | — | — | — |
| r2 | 94.8 | 88.7% | 79.3 | 302.4 | 74.7 | — | — | — | — |
| r3 | 81.5 | 76.2% | 66.3 | 252.8 | 62.3 | — | — | — | — |
| r4 | 81.5 | 76.2% | 66.3 | 252.8 | 62.3 | — | — | — | — |
| r5 | 73.6 | 68.9% | 58.6 | 223.4 | 55.0 | — | — | — | — |
| r6 | 65.7 | 61.5% | 50.8 | 193.9 | 47.6 | — | — | — | — |
| r7 | 69.6 | 65.1% | 56.2 | 214.4 | 52.7 | 126 | 39% | 1434 | 31% |
| r8 | 61.5 | 57.5% | 51.5 | 196.3 | 48.2 | — | — | — | — |

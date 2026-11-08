# E1 — Referencia, decisiones congeladas y criterio de aceptación

## 1. Modelo de referencia
- `Freepik/nsfw_image_detector` (EVA02-Base, parche 14, 448 px nativos), FP32 sin ajustar.
- Resolución congelada en **448 px** por decisión de diseño (sin barrido 224/336/448).
  Implica 32x32 = 1024 tokens + 1 de clase = 1025 tokens por imagen.
- Sin ajuste fino por defecto. El entrenamiento solo se hace si DES-1.3 (reentrenamiento tras podar)
  o una decisión posterior sobre los FN en ilustraciones lo exige.

## 2. Puntaje NSFW y umbral (heredados del benchmark)
- Puntaje: `softmax(logits)[1:].sum()` = low + medium + high (`models/vit_freepik.py`).
- Umbral: 0.5, NSFW si `puntaje >= 0.5` (`models/NSFWModelWrapper.py`).
- Binarización de etiquetas: porn, hentai, sexy = NSFW; neutral, drawings = SFW (`split/to_binary.py`).

## 3. Preprocesamiento exacto (timm 1.0.30, torch 2.2.0)
Obtenido con `resolve_data_config` para `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k`.
El proxy y el modelo dorado deben repetirlo bit a bit.

| Parámetro | Valor |
|---|---|
| Tamaño de entrada | 3 x 448 x 448 |
| Redimensionado | `Resize((448, 448))`, bicubic, antialias=True |
| `crop_pct` / `crop_mode` | 1.0 / squash (**no conserva la relación de aspecto**) |
| Recorte | `CenterCrop(448)`, sin efecto tras el redimensionado |
| Escala | uint8 -> float en [0, 1] |
| Media (RGB) | 0.48145466, 0.4578275, 0.40821073 |
| Desviación (RGB) | 0.26862954, 0.26130258, 0.27577711 |
| Modo de imagen | RGB (`.convert("RGB")`) |

## 4. Datos
- **test**: 504 imágenes (302 NSFW, 202 SFW), las del benchmark; intactas. Solo para el reporte final de cada etapa.
- **train**: 23 158 imágenes (13 892 NSFW, 9 266 SFW).
- **val**: 4 145 imágenes (2 477 NSFW, 1 668 SFW).
- Sin hashes MD5 compartidos entre conjuntos ni duplicados internos (`split/dedupe_splits.py`,
  verificable con `split/hashes.csv`).
- Calibración y reentrenamiento: solo train. Selección de hiperparámetros: solo val. Nunca test.
- Se excluyeron 2 hashes con etiqueta contradictoria (4 archivos en train): ilustraciones sugerentes
  que el scraper asignó a hentai y a drawings a la vez.

## 5. Línea base (test, 504 imágenes, `torch_results.csv`, ejecución fp16 en GPU)
| Métrica | Valor |
|---|---|
| Exactitud | 95.24 % (480/504) |
| FN | 9 de 302 NSFW (recall 97.0 %); 8 hentai, 1 sexy |
| FP | 15 de 202 SFW (especificidad 92.6 %) |

Nota: el benchmark corrió en fp16; la referencia FP32 de validación aún no se ha medido.

## 6. Modelo entero congelado (INT8, sin podar)
Fuente única: `quant/frozen.py` (`frozen_config()`); calibración: `quant/calib/calib_stats.json`
(255 imágenes de train, semilla 0). Modelo dorado: `quant/int_model.py`. Cualquier cambio invalida
la medición en test.

| Elemento | Decisión |
|---|---|
| Pesos | INT8 simétrico por canal de salida (±127) |
| Entradas de GEMM | INT8, escala = percentil 99.99 calibrado / 127 |
| Acumulador | INT32; sesgo cuantizado en el dominio del acumulador |
| Epílogo | multiplicador de punto fijo por canal (M0 de 31 bits + desplazamiento), redondeo mitad hacia +inf, saturación |
| `proj_out`, `fc2_out`, `swiglu_h` | INT16, **rango máximo** (recortarlos antes de la LayerNorm interna destruye la exactitud) |
| Entrada de `fc2` (`mlpn_out`) | **INT16**, rango máximo (en INT8 se pierden ~8 FN por 1000 imágenes) |
| Q, K | salida INT16 con escala s/128 (pre-RoPE); RoPE entero con seno/coseno Q1.14; salida INT8 (desplazamiento de 21 bits); el token cls no rota |
| V, fc1g, fc1x | requantizados a INT8 en el epílogo (fc1g con la entrada de 8 bits de la tabla SiLU) |
| QK^T | INT8 x INT8 -> INT32; puntaje real = S · s_q · s_k / sqrt(64) |
| Softmax | entero en 3 pasadas: máximo; exponencial por tabla de 1024 entradas en [0,16) (UQ0.16) + suma; recíproco por detección del bit alto + tabla de 256. Probabilidad UQ0.16 |
| AV | probabilidad 16 bits x V INT8; salida INT8 con la escala calibrada de `ctx` |
| SiLU | tabla de 255 entradas por bloque (entrada de 8 bits, salida de 16); producto con fc1x INT8; resultado INT16 |
| En el HPS (flotante) | patch embedding, suma de embeddings de posición, LayerNorm (x3 tipos), flujo residual, media de parches, cabeza |

Descartado: softmax con probabilidad de 8 bits (pierde ~20 % de masa de probabilidad en los
bloques iniciales y empeora 2.4x el error de logits); SiLU con entrada de 12 bits (igual que 8);
reescalado por canal tipo SmoothQuant (los atípicos están dispersos entre canales y tokens).

Validación (primeras 100 imágenes de val, mismas en todas las configuraciones):

| Configuración | Exactitud | FN | FP | Cambios vs. flotante | Error medio de logits |
|---|---|---|---|---|---|
| FP32 | 88.0 % | 9 | 3 | – | – |
| Entero, RoPE y epílogos en flotante | 88.0 % | 9 | 3 | 0 | 0.170 |
| Entero completo (congelado) | 89.0 % | 8 | 3 | 1 | 0.176 |

Medición final en test (504 imágenes, una sola vez, `quant/eval_test.py`, resultados en
`quant/results/test_final.json` y `test_final_logits.csv`). Referencia: `torch_results.csv`
(benchmark fp16, modelo `vit_freepik_eva02`).

| Métrica | Benchmark fp16 | INT8 congelado | Criterio INT8 | Cumple |
|---|---|---|---|---|
| Exactitud | 95.24 % (480) | 95.24 % (480) | >= 94.4 % (476) | sí |
| FN | 9 | 10 (8 hentai, 1 porn, 1 sexy) | <= 11 | sí |
| FP | 15 | 14 | <= 18 | sí |
| Cambios de veredicto vs. referencia | – | 2 (0.4 %) | <= 10 | sí |

Nota: los cambios de veredicto se miden contra el benchmark fp16, no contra una corrida FP32 propia.

## 7. Criterio de aceptación de E1
Con 504 imágenes, una imagen vale 0.2 puntos y el error estándar de la exactitud es de ~1 punto,
por eso los márgenes se expresan en imágenes.

| Criterio (test) | INT8 (cuantización) | INT8 podado |
|---|---|---|
| Exactitud | >= 94.4 % (>= 476/504) | >= 93.7 % (>= 472/504) |
| FN (métrica principal) | <= 11 | <= 12 |
| FP | <= 18 | <= 20 |
| Cambios de veredicto vs. FP32 | <= 2 % (<= 10 imágenes) | <= 4 % (<= 20 imágenes) |

Reglas:
1. Los FN pesan más que los FP; por eso hay un tope explícito de FN además de la exactitud.
2. Las decisiones intermedias (calibración, capas en INT16, punto de corte de la poda) se toman en
   validación, con los mismos márgenes relativos a la referencia medida en validación.
3. Test solo se usa para el reporte final de cada etapa; no para elegir hiperparámetros ni calibrar.
4. Si ningún recorte cumple la columna de poda, se documenta como resultado negativo y se entrega
   el INT8 sin podar.
5. Estos márgenes son una propuesta del equipo, no un requisito externo. Si existe un requisito
   externo de FN, prevalece sobre estas cifras.

## 8. Excepción documentada: r7 (INT8 podado)
Los criterios de la sección 7 **se mantienen sin cambios**. r7 no los cumple en test y se adopta como excepción decidida por el equipo.

| Métrica (test, 504 imgs) | Criterio INT8 podado | r7 | Cumple |
|---|---|---|---|
| Exactitud | >= 93.7 % (472) | 95.04 % (479) | sí |
| FN | <= 12 | 15 (14 hentai, 1 porn) | **no** |
| FP | <= 20 | 10 | sí |
| Cambios de veredicto vs. benchmark | <= 20 | 25 (18 NSFW→SFW, 7 SFW→NSFW) | **no** |

Referencias en test: benchmark fp16 95.24 % / 9 FN / 15 FP; INT8 sin podar 95.24 % / 10 FN / 14 FP / 2 cambios.

**Modelo:** r7 = 8 de 12 cabezas por bloque, neuronas SwiGLU 1232 de 2048 (bloque 1 protegido), 64.9 % de las operaciones del original,
destilación desde el original; checkpoint `quant/ckpt/r7_heads33_neur40.pt`, calibración `quant/calib_r7/`.

**Evidencia en validación** (`val_holdout.csv`, 1000 imgs, nunca usadas antes; modelo entero congelado):

| Modelo | Ops | Acc | FN | FP |
|---|---|---|---|---|
| Original flotante | 100 % | 95.4 % | 17 | 29 |
| INT8 sin podar | 100 % | 95.6 % | 19 | 25 |
| r5 (solo neuronas, 68.7 %) | 68.7 % | 95.2 % | 26 | 22 |
| r7 | 64.9 % | 95.3 % | 19 | 28 |

**Motivo:** la reducción de ~35 % de las operaciones puede acelerar el acelerador, y el FN de r7 (15 en test) sigue muy por debajo del de la CNN GantMan
rechazada en el benchmark (31) y comparable a LlavaGuard (21). En validación r7 iguala en FN al INT8 sin podar.

**Riesgos reconocidos:**
- Test se midió para r7 y para el INT8 sin podar antes de cerrar la decisión (regla 3); r7 se eligió por validación y la comparación r5/r7 se hizo en `val_holdout`, no en test.
- La validación no anticipó el resultado en test (9 cambios de veredicto en val[:500] frente a 25 en test); los FN suben en hentai (8 → 14).
- La regla 4 pide entregar el INT8 sin podar si ningún recorte cumple; esta excepción la reemplaza para r7. El INT8 sin podar sigue disponible como respaldo.
- Si existe un requisito externo de FN, prevalece sobre esta excepción (regla 5).

# Contexto del proyecto FPGate (para retomar sin repetir trabajo)

Actualizado: 2026-10-30. Rama: `JoseArroyo-ModelPrunning`. Idioma de trabajo: español.
Fuentes: archivos del repo (`docs/e1_referencia.md`, `quant/results/*.json`, logs, `git log`) y esta sesión.
**Lo que no sé:** el inicio de la conversación original se perdió en un crash de disco. Las razones de
decisiones anteriores (p. ej. por qué se cuantizó antes de podar) no están documentadas; no inventarlas.

## 1. Proyecto
FPGate: filtrado de contenido NSFW en tiempo real sobre FPGA para redes domésticas. Etapa actual **E1**:
llevar el detector a un modelo **entero (INT8)**, posiblemente **podado**, que sirva de "modelo dorado"
(contrato bit a bit para el hardware).

Modelo: `Freepik/nsfw_image_detector` = EVA02-Base (timm `eva02_base_patch14_448`), 12 bloques, 12 cabezas de 64,
D=768, SwiGLU con 2048 neuronas + LayerNorm interna, RoPE, 448 px → 1025 tokens. Salida de 4 logits;
puntaje NSFW = `softmax(logits)[1:].sum()`, umbral 0.5.

Entorno: Windows, Docker (`docker compose run --rm torch quant/<script>.py`, el entrypoint ya es `python`),
GPU **GTX 1650 de 4 GB**. No hay `python` en el host. Usar `-e PYTHONUNBUFFERED=1` y redirigir a `quant/logs/`.
Git se queja de "dubious ownership": usar `git -c safe.directory='*'`.

**Objetivo de la poda (dicho por el usuario):** reducir los recursos y las operaciones que el acelerador debe realizar
**sin perder precisión**. Se poda para que el modelo rinda mejor *dentro del acelerador*, no para minimizar parámetros en abstracto.
Por eso la métrica de éxito es "menos operaciones/recursos con el criterio de aceptación cumplido"; podar solo cabezas topa en ~55 % de las
operaciones y el MLP (≈55 % de las MACs) también debe entrar en juego.

## 2. Datos y reglas (ya hechas, no repetir)
- `split/`: `make_split.py` (estratificado por etiqueta binaria, semilla 42), `to_binary.py`
  (porn, hentai, sexy = NSFW; neutral, drawings = SFW), `dedupe_splits.py`, `hashes.csv`.
- **Dedupe ya hecho** (commit `5d573c8`): por **MD5 del contenido**. Quita de train/val lo que está en test,
  de val lo que está en train, duplicados internos y 2 hashes con etiqueta contradictoria (4 archivos en train).
  Verificable con `split/hashes.csv`. (Alcance: exacto; no es perceptual. Se da por cerrado.)
- Tamaños: test 504 (302 NSFW / 202 SFW; **las del benchmark, intactas**), train 23 158 (60.0 % NSFW),
  val 4 145 (59.8 % NSFW). `val_holdout.csv` = últimas 1000 de val, **reservadas y sin usar**.
  Decisiones se toman con `val.csv[:N]`; `prune_train.py` exige `--eval-images <= 3145`.
- Reglas: calibración y reentrenamiento solo con train; selección de hiperparámetros solo con val;
  **test una sola vez por etapa, nunca para elegir**.
- Muestras verificadas en esta sesión: val[:500] = 59.2 % NSFW con las 5 clases equilibradas (89–108 c/u);
  muestras de entrenamiento de poda = 59.3–59.6 % NSFW. Sin sesgo de clase.

## 3. Referencia y criterio de aceptación
- **Referencia = benchmark de 504 imágenes** (`torch_results.csv`, modelo `vit_freepik_eva02`, que se corrió
  en fp16 en GPU). Decisión del usuario: se acepta como referencia, no se mide otra FP32 propia.
  Valores: acc 95.24 % (480/504), FN 9 (8 hentai, 1 sexy), FP 15.
- Benchmark entre modelos (`analysis/resumen.csv`): CLIP-LAION sondeado 96.83 % (1715 MB, ~3 s/img),
  ViT Freepik 95.24 % (173 MB, ~0.67 s), CNN GantMan MobileNetV2 92.26 % (18 MB), LlavaGuard 85.53 % en 159 imgs (36 s/img).
- Criterio (test, 504 imgs; `docs/e1_referencia.md` §7):

| | INT8 | INT8 podado |
|---|---|---|
| Exactitud | ≥ 94.4 % (476) | ≥ 93.7 % (472) |
| FN | ≤ 11 | ≤ 12 |
| FP | ≤ 18 | ≤ 20 |
| Cambios de veredicto vs. referencia | ≤ 10 | ≤ 20 |

Si ningún recorte cumple la columna de poda: documentar como resultado negativo y entregar el INT8 sin podar.
Los FN pesan más que los FP.

## 4. Cuantización (hecha y congelada)
Código: `quant/int_model.py` (modelo dorado), `golden.py`, `nonlinear.py` (softmax/SiLU por tabla), `weights.py`,
`calibrate.py`, `frozen.py` (**única fuente de la config congelada**), `eval_int.py`, `eval_test.py`.
Config congelada: pesos INT8 por canal; entradas de GEMM INT8 con percentil 99.99; acumulador INT32; epílogo de punto fijo;
`proj_out`, `fc2_out`, `swiglu_h` en INT16 con **rango máximo**; **entrada de `fc2` en INT16** (en INT8 se pierden ~8 FN/1000 imgs);
Q,K INT16 pre-RoPE y RoPE entero; softmax entero de 3 pasadas (UQ0.16, tabla de 1024 + recíproco de 256);
AV 16 bits × V INT8; SiLU por tabla (entrada 8 bits). En flotante (HPS): patch embedding, LayerNorms, residual, pooling, cabeza.
Calibración congelada: 255 imágenes de train, semilla 0 (`quant/calib/`).

Historia (val, mismas imágenes en cada tabla; "cambios" = veredictos distintos al flotante):
- `fix`/`fix2`/`fix3` (40 imgs): cada capa sola en INT8 casi no daña; **`fc2` era el cuello de botella**
  (error de logits 0.50 solo con fc2). INT16 con rango máximo en la entrada de `fc2`: 0.92 → 0.107.
- `big` (1000 imgs): FP32 95.5 % / 28 FN; todo INT8 95.0 % / 36 FN / 17 cambios / err 0.475;
  con `fc2` de entrada INT16: 95.5 % / 28 FN / 2 cambios / err 0.101.
- `stage2`/`stage3` (100 imgs): softmax de 8 bits **descartado** (err 0.389 vs 0.16); SiLU de 12 bits = 8 bits;
  todo entero: 89 %, 8 FN, 1 cambio, err 0.176 (FP32: 88 %, 9 FN, 3 FP).
- Descartado: SmoothQuant por canal (atípicos dispersos entre canales y tokens, `outlier_channels.json`).
- `test_gemm`/`test_softmax`: la GEMM entera difiere 1–2 LSB de fake-quant.
- **Test final INT8 sin podar** (`test_final.json`): 95.24 %, FN 10, FP 14, 2 cambios → **cumple todo, aceptado**.

## 5. Poda (DES-1.3) — estado actual
Código: `quant/prune.py` (poda estructurada, importancia = norma del aporte de cada cabeza / |act|·‖col fc2‖ por neurona),
`sensitivity.py`, `prune_train.py` (poda + destilación), `train_probe.py` (viabilidad de memoria).
Distilación: maestro = original sin podar; alumno imita los 4 logits (KL + 0.1·MSE), **sin etiquetas**;
AdamW lr 2e-5 con decaimiento lineal a 10 %, acumulación 8, todos los tensores entrenables salvo tablas RoPE.
~0.87 s/imagen de entrenamiento. Neuronas en múltiplos de 16; `--protect-neurons` por defecto = bloque 1.

**Sensibilidad sin reentrenar** (`sens_v1`, 100 imgs de val, base 88 %/9 FN):
- Un bloque suelto casi no duele (err 0.07–0.4). Los errores se acumulan al podar todos los bloques.
- Todas las cabezas 25 %: 88 %, err 1.02, 0 cambios. Cabezas 50 %: 76 %, 14 cambios, err 2.39.
- Neuronas 25 %: 72 %, 18 cambios. Neuronas 50 %: 56 %. Cabezas+neuronas 25 %: 68 %.
- **Bloque 1 es sensible en neuronas** (25 % → 80 %, 10 cambios). Cabezas aguantan mucho más que neuronas.
- Con 100 imgs una imagen = 1 punto: diferencias de 1–3 imágenes son ruido.

**Costo en operaciones** (derivado de `macs()` con 1025 tokens; no medido en hardware): por bloque, atención ≈ 45 % de las MACs
y MLP ≈ 55 %. Quitar el 25 % de las cabezas deja **88.6 %** de las operaciones; 50 % de cabezas ≈ 77 %; todas las cabezas ≈ 55 %.
Las cabezas solas tienen un tope de reducción; para bajar más hay que tocar neuronas.

**Rondas hechas** (500 imgs de val, flotante, sin cuantizar; maestro 95.0 %, FN 17, FP 8):

| Ronda | Qué hizo | Acc | FN | FP | Cambios | Err logits |
|---|---|---|---|---|---|---|
| `r1_heads25` | Poda 25 % cabezas (todos los bloques) + 1000 imgs (seed 0) | 91.8 % → **94.2 %** | 28 → 18 | 13 → 11 | 16 → 8 | 1.022 → **0.591** |
| `r2_heads25` | Parte de r1, **no podó nada** (ver abajo) + 3000 imgs (seed 1) | 94.2 % | 18 | 11 | 8 | 0.591 → 0.553 |
| `r3_heads25_neur25` | Parte de r2, neuronas 25 % (bloque 1 protegido, 1536/2048 neuronas) + 3000 imgs (seed 2); **76.1 % de las ops** | 90.4 % → **94.8 %** | 41 → 12 | 7 → 14 | 27 → **19** | 1.240 → **0.690** |
| `r4_heads25_neur25` | Parte de r3, **no podó nada**, solo entrenamiento: 5000 imgs (seed 3); 76.1 % de las ops | 94.8 % → **95.0 %** | 12 → 14 | 14 → 11 | 19 → **12** | 0.690 → **0.597** |

r3: el reentrenamiento recupera mucho (FN 41 → 12), pero los veredictos que difieren del maestro (19 = 3.8 %) y el err de logits (0.69)
son peores que en r2 (8 y 0.55): la exactitud igual al maestro (94.8 vs 95.0, ruido) esconde que los errores pasaron de FN a FP y que el
alumno se aleja más del maestro. La pérdida de entrenamiento seguía bajando (0.31 → 0.19), no hay meseta.

**Aclaración importante (malentendido de esta sesión):** `--heads-frac` es **relativo al modelo original**. Con `--start r1` y
`--heads-frac 0.25` el plan no recorta nada (ya hay 9 de 12). La "ronda 2" fue solo más entrenamiento, no poda adicional.
La intención del usuario era que cada ronda **pode más y reentrene, para ver hasta dónde llega la poda**.
Para podar más hay que subir la fracción (p. ej. `--heads-frac 0.5`) o añadir `--neurons-frac`.
Los pesos cambiaron solo 0.27 % de r1 a r2: el entrenamiento está limitado (375 pasos, lr bajo), no hay evidencia de bug.

**Prueba de compatibilidad con el entero** (`frozen_r1.json`, 100 imgs de val, config congelada + calibración nueva
`quant/calib_r1/`, 255 imgs de train): podado flotante 88 %/8 FN/4 FP → podado entero 88 %/8 FN/4 FP, **0 cambios**, err 0.152
(sin podar entero: 0.176). El entero acepta 9 cabezas por bloque. Cambios de código para esto: `float_ref.split_heads`
infiere las cabezas (`d // 64`); `--ckpt` en `calibrate.py` y `eval_int.py`; suite `frozen` en `eval_int.py`.

Misma prueba con r3 (`frozen_r3.json`, calibración `quant/calib_r3/`): podado flotante 88 %/7 FN/5 FP → entero 88 %/7 FN/5 FP, **0 cambios**,
err 0.191 (r1: 0.152; sin podar: 0.176). El entero también acepta neuronas podadas (1536/2048 por bloque, bloque 1 sin podar).

r4 en **500 imgs de val** (`frozen_r4.json`, calibración `quant/calib_r4/`): podado flotante 95.0 %/14 FN/11 FP → **entero 94.4 %/17 FN/11 FP**,
3 cambios (0.6 %, todos NSFW→SFW), err 0.182. La cuantización añade 3 FN (la dirección más costosa). Contra el maestro (95.0 %/17 FN/8 FP en esas
500): FN igual, FP +3. Cambios entero-vs-maestro no calculados directamente (cota: 12 de la poda + 3 de la cuantización = 15 de 500 = 3 %).
Los márgenes del criterio, trasladados a val (regla 2, relativos a la referencia de val: 95.0 %/17 FN/8 FP): acc ≥ 93.5, FN ≤ 20, FP ≤ 13 → r4 entero cumple
los tres. Solo es una lectura en val; el criterio real es en test.

Las cifras de la tabla de rondas se verificaron contra `quant/results/prune_r*.json` (coinciden).

**Cuidado con la métrica:** el titular "94.2 %" esconde un err de logits de 0.55, >3× el costo de cuantizar (0.15).
El error estándar de la exactitud con 500 imgs es ~1 punto.

## 6. Protocolo estándar (acordado con el usuario; aplicar igual en cada ronda) y plan
**Por qué:** hasta r4 el proceso fue inconsistente (rondas que podan vs. solo entrenan; entrenamiento de 1000/3000/5000 imgs; entero medido en
val[:100] o val[:500]; las pruebas de cuantización antiguas usaron 40/100/1000). Las cifras de 88–89 % son de val[:100], una muestra difícil donde
el maestro saca 88 %; no son comparables con val[:500] (maestro 95.0 %).
r2 = r1 con más entrenamiento; r4 = r3 con más entrenamiento (mismos pesos de partida, misma estructura, sin podar más).

Protocolo por ronda:
1. **Una sola variable: la cantidad de poda** (fracciones relativas al original). Mismo presupuesto de entrenamiento (3000 imgs, lr 2e-5, accum 8),
   semilla nueva. Sin rondas de solo entrenar salvo petición explícita.
2. Flotante en val[:500] contra el maestro (lo hace `prune_train.py`).
3. **Entero siempre**: `quant/run_round_eval.sh <corto> <tag>` (calibra con 255 imgs de train y evalúa la config congelada en val[:500]);
   `eval_int.py` reporta contra el maestro (`flips_vs_teacher`, `logit_err_vs_teacher`) y contra su propio flotante.
4. Misma tabla siempre: operaciones restantes, acc, FN, FP, cambios vs. maestro, error de logits; en flotante y en entero.
5. Regla de parada: seguir podando mientras el entero cumpla los límites trasladados a val (acc ≥ 93.5 %, FN ≤ 20, FP ≤ 13, cambios ≤ 4 %).
   La última ronda que cumple es el candidato.
6. Test (`split/test.csv`, 504 imgs): **una sola vez, con el candidato final**. Chequeo intermedio permitido solo en `val_holdout.csv`.

**Tabla homogénea (protocolo aplicado a r1–r4; val[:500]; `std500_r1..r4.json`).** Maestro: 95.0 %, 17 FN, 8 FP.
Límites en val (entero): acc ≥ 93.5, FN ≤ 20, FP ≤ 13, cambios vs. maestro ≤ 20 (4 %).

| Ronda | Ops | Flotante: acc / FN / FP / cambios / err | Entero: acc / FN / FP / cambios vs. maestro / err vs. maestro | ¿Cumple (entero)? |
|---|---|---|---|---|
| r1 | 88.6 % | 94.2 / 18 / 11 / 8 / 0.591 | 94.4 / 19 / 9 / 7 / 0.608 | sí |
| r2 (r1 + entrenar) | 88.6 % | 94.2 / 18 / 11 / 8 / 0.553 | 93.6 / 21 / 11 / 11 / 0.602 | **no** (FN 21; acc 93.6 al límite) |
| r3 | 76.1 % | 94.8 / 12 / 14 / 19 / 0.690 | 94.2 / 15 / 14 / 22 / 0.717 | **no** (FP 14, cambios 22) |
| r4 (r3 + entrenar) | 76.1 % | 95.0 / 14 / 11 / 12 / 0.597 | 94.4 / 17 / 11 / 15 / 0.637 | sí |
| r5 (r4 + neuronas 39.8 %, 1232/2048; poda real) | 68.7 % | 94.8 / 18 / 8 / 13 / 0.664 | 94.8 / 18 / 8 / 13 / 0.688 | sí (FN 18 de 20) |
| r7 (r5 + cabezas 33 %, 8/12; neuronas como r5; poda real) | 64.9 % | 95.4 / 14 / 9 / 10 / 0.690 | 95.2 / 15 / 9 / 9 / 0.705 | **sí, con margen** |
| r8 (r7 + cabezas 50 %, 6/12; neuronas como r5; poda real) | 57.3 % | 93.4 / 17 / 16 / 22 / 0.818 | 93.8 / 16 / 15 / 20 / 0.823 | **no por poco** (FP 15 > 13; cambios 20 al límite; acc 93.8 pasa) |
| r6 (r5 + neuronas 54.7 %, 928/2048; poda real) | 61.3 % | 93.4 / 19 / 14 / 20 / 0.734 | 93.0 / 23 / 12 / 20 / 0.760 | **no** (acc 93.0 < 93.5; FN 23 > 20; cambios 20 al límite) |

Lectura: el costo de cuantizar es 3–5 cambios por 500 en todas las rondas. Que r2 (más entrenamiento) quede peor que r1 en entero (93.6 vs 94.4) es ruido de
~5 imágenes: las diferencias entre rondas de ≤ 1 punto no son señal. Candidato actual que cumple con más reducción: **r5 (68.7 % de las ops)**; antes r4 (76.1 %).
r5: sin reentrenar 92.8 %/28 FN/15 cambios → reentrenado 94.8 %/18 FN/13 cambios (3000 imgs, seed 4); entero idéntico al flotante (0 cambios por cuantizar).
Que r5 salga igual o mejor que r3/r4 con menos ops es ruido de ±1 punto, no una mejora real. Margen más justo en r5: FN 18 de 20.
r6 (seed 5, 3000 imgs): sin reentrenar 89.8 %/44 FN/32 cambios → 93.4 %/19 FN/20 cambios; la cuantización cuesta más que antes (8 cambios, FN 19 → 23).
**Primer límite incumplido: entre 68.7 % (r5, cumple) y 61.3 % (r6, falla) de las ops con cabezas 25 % + neuronas podadas.**
r7 (parte de r5, seed 6, 3000 imgs): quitar 4 cabezas por bloque casi no daña ni sin reentrenar (95.0 %/18 FN/12 cambios) y reentrenado queda 95.4 % en flotante,
95.2 % en entero (1 cambio por cuantizar). **Las cabezas son mejor palanca que las neuronas**: r7 reduce más ops que r5 (64.9 vs 68.7 %) con menos daño que r6 (61.3 %).
Costo por componente (derivado de `macs()`): quitar 1 cabeza en todos los bloques ≈ 3.8 % de las ops totales; 2048→1232 neuronas en 11 bloques ≈ 19.9 %.
r8 (parte de r7, seed 7, 3000 imgs): sin reentrenar 91.4 %/30 FN/24 cambios → 93.4 %/17 FN/16 FP/22 cambios. El error vs. el maestro sube con cada paso de cabezas
(0.69 → 0.82) y los FP crecen (8 → 15). Falla por 2 FP y roza el límite de cambios; la señal es la tendencia, no un solo número.
**Candidato vigente: r7 (64.9 % de las ops).** Pendiente decidir si se prueba 7 cabezas (≈ 61 % de las ops) o se pasa a la medición final.
Nota práctica: `--neurons-frac` debe dejar un número de neuronas múltiplo de 16 (aserción en `make_plan`): usar k/128 (0.25 = 32/128, 0.3984375 = 51/128).

Plan:
1. ~~Ronda de poda + destilación en flotante~~ (r1 hecha).
2. ~~Prueba de compatibilidad con el entero~~ (r1 y r3 pasan: 0 cambios contra su flotante, 100 imgs).
3. ~~Añadir neuronas (r3) y entrenar más (r4)~~: r4 = 76.1 % de las ops, 95.0 % en flotante (500 imgs de val).
4. **En curso:** evaluación entera de r4 en las 500 imgs de val (`frozen_r4.json`); luego **podar más** desde r4 (propuesta: neuronas 40 %,
   cabezas 25 %, ~68 % de las ops) para encontrar el límite, decidiendo en val en flotante.
5. Congelar el checkpoint, recalibrar con train, medir entero en val (puede usarse `val_holdout.csv`) y **test una sola vez**.
   **Ningún modelo podado se ha medido todavía en test**; todas las cifras de poda son de val (100 o 500 imgs).
6. Si no cumple la columna "INT8 podado", entregar INT8 sin podar.

## 6b. Medición en TEST de r7 (`test_r7.json`, 504 imgs, hecha el 2026-11-01 a petición del usuario)
Entero, config congelada + `quant/calib_r7/`. Referencia: benchmark fp16 (95.24 %, 9 FN, 15 FP).

| | INT8 sin podar (1ª vez) | **r7 (64.9 % ops)** | Límite "INT8 podado" |
|---|---|---|---|
| Exactitud | 95.24 % (480) | **95.04 % (479)** | ≥ 93.7 % (472) → cumple |
| FN | 10 (8 hentai) | **15 (14 hentai, 1 porn)** | ≤ 12 → **no cumple** |
| FP | 14 | **10** | ≤ 20 → cumple |
| Cambios vs. benchmark | 2 | **25** (18 NSFW→SFW, 7 SFW→NSFW) | ≤ 20 → **no cumple** |

`accepted: false`. Por clase: hentai FN 8 → 14; neutral 12 de 97 imágenes cambian de veredicto vs. el benchmark (FP 9 → 7); drawings, porn, sexy casi sin cambio.
**La val no predijo esto:** r7 tenía 9 cambios vs. maestro en val[:500] (1.8 %) y en test tiene 25 (5.0 %) vs. el benchmark; el INT8 sin podar tenía 2.
Consecuencias: (1) los límites trasladados a val (FN ≤ 20, cambios ≤ 4 %) fueron demasiado laxos para anticipar test; (2) el daño está concentrado en
**hentai** (ilustraciones), la clase de FN dominante desde el inicio; (3) **test ya se miró para r7 y para el INT8 sin podar: no usarlo para elegir entre candidatos**;
para decisiones intermedias usar `val_holdout.csv` (1000 imgs, nunca usadas) y reportar FN por clase en val.

## 7. Estado del repo y archivos
- Sin commitear: `.gitignore`, `quant/eval_int.py`, `int_model.py`, `calibrate.py`, `float_ref.py`; nuevos: `docs/`, `quant/{eval_test,frozen,nonlinear,prune,prune_train}.py`,
  resultados `sens_v1`, `stage2/3_v1`, `softmax_test`, `frozen_r1`, `prune_r*_heads25`, `test_final*`.
- Checkpoints en `quant/ckpt/` (`r1_heads25`, `r2_heads25`, `r3_heads25_neur25`, `r4_heads25_neur25`, ~300 MB c/u). Logs en `quant/logs/`.
  Calibraciones provisionales: `quant/calib_r1/`, `calib_r3/`, `calib_r4/` (la congelada sigue en `quant/calib/`).
- Tiempos medidos: entrenamiento ≈ 0.81–0.87 s/imagen (r4: 5000 imgs = 68 min); calibración ≈ 3 min; evaluación entera ≈ 1.7 s/imagen.
- Se borraron `smoke*.json` y un directorio basura `C:` en la raíz.
- `docs/e1_referencia.md` tiene problemas de codificación al leerlo desde PowerShell (UTF-8 sin BOM); leerlo con Read o `-Encoding UTF8`.

## 8. Reglas de trabajo para mí
- Antes de proponer algo, revisar este archivo, `docs/` y `quant/results/`: no volver a proponer lo ya hecho
  (dedupe, referencia propia, calibración congelada, etc.).
- Ser explícito con lo que hace cada parámetro (p. ej. `--heads-frac` relativo) y con qué modelo se mide (flotante vs. entero).
- Reportar con tabla y distinguir ruido (<1 punto con 500 imgs) de señal.
- Confirmar el alcance de una "ronda" con el usuario antes de lanzarla (≈15–45 min de GPU).

## 9. Comparación en val_holdout (1000 imgs, entero congelado)
| Modelo | Ops | Acc | FN | FP | Cambios vs. original |
|---|---|---|---|---|---|
| Original flotante | 100 % | 95.4 % | 17 | 29 | – |
| INT8 sin podar | 100 % | 95.6 % | 19 | 25 | 8 |
| r5 INT8 | 68.7 % | 95.2 % | 26 | 22 | 32 |
| **r7 INT8** | **64.9 %** | **95.3 %** | **19** | **28** | 33 |

FN por clase: sin podar 19 hentai; r5 23 hentai + 2 porn + 1 sexy; r7 18 hentai + 1 sexy. Regla previa (r5 reemplaza a r7 solo con ≥3 FN menos): **no se cumple, se queda r7**.
Decisión: r7 como modelo E1 podado, con excepción documentada al criterio original (test: FN 15 > 12, cambios 25 > 20).

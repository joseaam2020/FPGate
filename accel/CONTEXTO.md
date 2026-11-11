# Contexto para empezar DES-2 (acelerador)

Resumen de todo lo que hay en el repositorio y de las decisiones tomadas hasta el 2026-11-08, para retomar sin repetir trabajo. Complementa a `guia_DES-2_acelerador.md` (la guía de la etapa, ya corregida con lo que se explica aquí).

## 1. Qué es el proyecto y en qué punto está

FPGate (CE-1114): filtro de contenido NSFW en tiempo real para redes domésticas. Un proxy en el HPS de una DE10-Standard (Cyclone V SoC, ARMv7) intercepta HTTP/HTTPS, extrae las imágenes y se las manda a un acelerador en FPGA que ejecuta un ViT cuantizado a INT8. Si el puntaje supera el umbral, el proxy sustituye la imagen.

| Etapa | Estado |
|---|---|
| E1 / DES-1: modelo, cuantización y poda | **Terminada.** Modelo congelado r7, paquete exportado y verificado bit a bit. |
| DES-3: proxy (`proxy/`) | **Terminada** y probada (387 verificaciones, fuzzing, pruebas de CA vencida y de no persistencia). Solo faltan las mediciones de latencia, que se harán cuando exista el acelerador. |
| **DES-2: acelerador en SystemVerilog** | **Por empezar.** `accel/` está vacía. |
| DES-4: integración | Después. |

## 2. Repositorio

Rama de trabajo: `JoseArroyo-Accelerator`. `develop` ya contiene Proxy, ModelPrunning y DetectionModelsBenchmark (PR #3). Hay **2 commits locales sin subir** (`afb2ca8`, `394d909`); no se hizo push porque no se ha pedido.

```
model/    E1: benchmarks, splits, y quant/ (cuantización, poda, exportación)
  docs/e1_referencia.md     decisiones congeladas y criterio de aceptación de E1  (LEER PRIMERO)
  context.md                historial del trabajo de E1
  quant/golden.py, nonlinear.py, int_model.py, float_ref.py, frozen.py   modelo entero de referencia
  quant/export.py, export_check.py                                         exportación y verificación
  quant/release/e1_r7/      paquete del modelo (218 MB) -- NO está en git (ignorado)
proxy/    DES-3: proxy, panel de administración, mock del acelerador, pruebas
accel/    DES-2 (vacía; este archivo es lo único)
```

Archivos sin versionar que existen en el disco: `guia_DES-2_acelerador.md` y `guia_DES-3_proxy.md` (las guías), `Main/Dashboard/Certificado.html` (maquetas viejas de la raíz) y `proxy/tools/chrome_via_proxy.ps1`. La guía de DES-2 que hay en el repositorio es una copia **modificada**; el usuario conserva su propia copia en otro lugar.

**Importante:** `model/quant/release/` y los puntos de control (`quant/ckpt/`) están en el `.gitignore`. El paquete solo existe en el disco del usuario. Los datos crudos (`raw_data`) tampoco están en el repositorio.

## 3. El modelo que se va a acelerar (r7)

Freepik `nsfw_image_detector`: EVA02-Base, parche 14, 448 px → 32×32 = 1024 tokens + 1 de clase = **1025 tokens**, 12 bloques, dimensión 768, cuatro logits. Puntaje NSFW = `softmax(logits)[1:].sum()`, NSFW si ≥ 0.5.

**r7 es el modelo podado:** 8 cabezas por bloque (dimensión interna de atención 512) y MLP SwiGLU de **1232** neuronas, salvo el bloque 1 que conserva **2048**. 69.6 GMAC (65 % del original), 56.2 M de parámetros. Entró como **excepción** al criterio de E1 (en test: 15 falsos negativos frente a ≤ 12, 25 cambios de veredicto frente a ≤ 20; ver `e1_referencia.md` §8). El modelo sin podar queda de respaldo.

### Aritmética cuantizada (todo sale de `e1_referencia.md` §6 y de `golden.py`)

- Pesos INT8 simétricos por canal de salida. Acumulador INT32; sesgo INT32 en el dominio del acumulador (`k_w` no tiene sesgo).
- Requantización: `y = sat((acc·M0 + 2^(s-1)) >> s)`, `M0` de 31 bits (siempre en [2³⁰, 2³¹)), `s` por canal (30 a 42), redondeo mitad hacia +∞, producto de **63 bits**. Saturación a ±127 o ±32767.
- **Entradas de 16 bits:** la entrada de `fc2` es INT16 (en INT8 se pierden ~8 falsos negativos por 1000 imágenes) y la probabilidad `P` de la atención es UQ0.16 (sin signo). Todo lo demás es INT8 en la entrada.
- Salidas: Q y K INT16 (antes de RoPE, escala s/128); V, `fc1g`, `fc1x` INT8; `proj`, `swiglu_h` y `fc2` INT16.
- RoPE entero con senos y cosenos Q1.14 (tablas de 1024×64), el token de clase no rota, salida INT8.
- Softmax entero en tres pasadas: máximo; exponencial por tabla de 1024 entradas en [0, 16) (paso 1/64, salida UQ0.16) más la suma; recíproco por detección del bit alto más tabla de 256. Se probó con 8 bits de probabilidad y se descartó.
- SwiGLU: `fc1g` se requantiza a INT8, indexa una **tabla SiLU de 255 entradas por bloque** (salida de 16 bits, sin interpolación), se multiplica por `fc1x` y se requantiza con un `M0` y un desplazamiento escalares por bloque.
- En el HPS (flotante): embedding de parches, suma de embeddings de posición, **tres LayerNorm por bloque** (antes de atención, antes del MLP y la interna del MLP entre SwiGLU y `fc2`), flujo residual, media de parches y cabeza.

### Flujo de un bloque (reparto HPS / FPGA)

Reconstruido de E1 y de `int_model.py`; confírmalo leyendo `IntModel.forward` antes de escribir el modelo de referencia:

1. HPS: LayerNorm 1, cuantiza a INT8.
2. FPGA: Q, K (INT16) y V (INT8).
3. HPS: RoPE entero sobre Q y K, pasan a INT8.
4. FPGA, por cabeza: `S = Q·Kᵀ` (INT8×INT8→INT32), softmax entero, `O = P·V` (16 bits × INT8), salida INT8.
5. FPGA: proyección de salida (INT16). HPS: suma residual y LayerNorm 2, cuantiza a INT8.
6. FPGA: `fc1g` y `fc1x` (INT8), SiLU por tabla y producto, resultado `swiglu_h` INT16.
7. HPS: LayerNorm interna, cuantiza a **INT16**. FPGA: `fc2` (16 bits × INT8), salida INT16. HPS: suma residual.

### El paquete `model/quant/release/e1_r7/`

`manifest.json` (86 capas: `patch_embed`, `head`, y por bloque `bN.q_w, k_w, v_w, proj_w, fc1g_w, fc1x_w, fc2_w`, con formas, offsets, escalas, bits, y SHA-256 de cada archivo), `weights_i8.bin` (55.3 MB, `[salida, entrada]` por filas, tensores alineados a 64 bytes, little-endian), `epilogue_i32.bin` (sesgos, `M0` y desplazamientos por capa), `epilogue_f64.bin`, `tables.bin` (exp, recíproco, SiLU por bloque, RoPE) y `hps_f32.bin` (parámetros flotantes del HPS). En `vectors/` hay **20 imágenes** (CHW float32 normalizado) con logits esperados y SHA-256 de los intermedios; la imagen 0 trae los intermedios completos de los bloques 0 y 11 (`stages_img_00.npz`).

Verificación: `python quant/export_check.py --pkg quant/release/e1_r7` (en el contenedor `torch` de `model/docker-compose.yml`) reconstruye la inferencia solo desde el paquete y reproduce los 20 vectores **bit a bit**. Esa es la referencia ejecutable del acelerador.

## 4. Decisiones tomadas

**Fijadas por el Documento de Diseño:** arquitectura temporal con **un solo motor GEMM**; arreglo 16×16 *output-stationary*; INT8×INT8 en DSP modo 9×9 con acumulador por PE; reloj único (objetivo 150 MHz); pesos por tiles en la DDR3; doble buffer de pesos y salida; softmax entero con tablas; CSR como único contacto con el HPS (base `0xFF200000`, 32 bits); DMA Avalon-MM maestro.

**Cerradas por E1:** probabilidad de atención de 16 bits (UQ0.16); score = `softmax[1:].sum()`; resolución 448 (1025 tokens); requantización con `M0` de 31 bits y desplazamiento por canal.

**Tomadas el 2026-11-08:**
- **C. Entradas de 16 bits en dos pasadas INT8×INT8** (byte alto y bajo) con acumulador desplazable: en la segunda pasada el acumulador se desplaza 8 bits a la izquierda y sigue sumando (`acc = 256·Σ(alto·w) + Σ(bajo·w)`). La activación entra al PE como 9 bits con signo (el byte bajo y el de `P` se extienden con cero). Se descartó la opción de multiplicadores de 16×8 en 18×19: 256 PEs necesitarían 128 bloques DSP de los 112 que hay, y con un arreglo de 14×14 (que cabe) el tiempo es el mismo (≈ 2.35 s frente a 2.29 s) con tiles de 14.
- **D. Epílogo bit-exacto** con el modelo (`M0` completo). El costo se mide en Quartus.
- **E. Verificación bit-exacta etapa por etapa** contra `e1_r7`; tolerancia solo en el score final de punta a punta (hito H6).
- **F. r7 es el objetivo y todo va parametrizado** en SystemVerilog (filas y columnas del arreglo, anchos, cabezas, neuronas, tokens, filas por panel). Nada de constantes que dependan de la forma del modelo.
- Mantener `ACC_W = 32` (no 28): `fc2` produce productos de hasta 22 bits con K hasta 2048.

**Pendientes** (detalle en la sección 4 de la guía): 4 puerto de datos (FPGA-to-SDRAM o ACP, se mide en la placa); 5 reloj final; 7 mapa de CSR (acordar con `fpgate-accel`); 8 acumulador en ALMs frente a la variante de dos multiplicadores 18×19; **9 filas por panel en las operaciones de 16 bits** (`PANEL_M` = 64 ó 128); **10 DSP asignados al epílogo**.

### Las dos cuentas que más pesan (hechas con las dimensiones del manifest)

| Operación | GMAC | Pasadas |
|---|---|---|
| Q, K, V | 14.51 | 1 |
| QKᵀ | 6.46 | 1 |
| P·V | 6.46 | 2 |
| Proyección | 4.84 | 1 |
| fc1g + fc1x | 24.56 | 1 |
| fc2 | 12.28 | 2 |
| Total FPGA | 69.10 → **87.8 equivalentes** | |

Mínimo teórico al 100 % de utilización: **2.29 s a 150 MHz**, 3.43 s a 100 MHz (sin las dos pasadas serían 1.80 s y 2.70 s). El Documento de Diseño estimaba 3–7 s con el modelo completo.

**Memoria** (553 M10K ≈ 691 KiB): el panel de activaciones en INT16 para `fc2` ocupa el doble sean cuales sean las pasadas: 308 KiB (K=1232) o 512 KiB (K=2048, bloque 1) con 128 filas; con `PANEL_M = 64` son 154 y 256 KiB, a costa de leer los pesos de `fc2` 17 veces en vez de 9 (≈ 204 MB frente a ≈ 108 MB por imagen). Este presupuesto es lo primero que hay que cerrar en DES-2.1.

**Estimaciones mías, marcadas como tales en la guía y por confirmar:** el costo de DSP del epílogo (el producto de 63 bits equivale a ≈ 2 bloques DSP por multiplicación completa; con 4 DSP serían ≈ 8 ciclos por fila de 16) y el efecto en el timing del multiplexor del desplazamiento.

## 5. Plan de trabajo (de la guía)

Hitos, cada uno verificable por sí solo: **H0** CSR + `VERSION` + `heartbeat` + sistema en Platform Designer con el HPS; **H1** `pe` + arreglo 16×16 + vaciado; **H2** buffers M10K + secuenciador + DMA con memoria simulada; **H3** epílogo + modo de dos pasadas (capa lineal, SwiGLU y `fc2` bit-exactos); **H4** integración en la placa (GEMM real comparado con NumPy); **H5** unidad de atención con `P·V` en dos pasadas; **H6** modelo completo orquestado por el HPS. Si la semana 9 llega sin atención, el softmax puede hacerlo el HPS temporalmente.

Estructura propuesta de `accel/`: `rtl/` (`fpgate_pkg.sv`, `pe`, `gemm_array`, `tile_buffers`, `tile_sequencer`, `dma_rd/wr`, `epilogue`, `attn_softmax`, `csr`, `fpgate_accel_top`), `tb/`, `ref/` (modelo de referencia a nivel de hardware, reutilizando `model/quant/golden.py`), `tools/` (`pack_weights.py` guiado por `manifest.json`, `gen_luts.py` leyendo `tables.bin`), `quartus/`, `DECISIONES.md`.

Primeros pasos sugeridos: (1) leer `e1_referencia.md` y `IntModel.forward`; (2) correr `export_check.py` para tener la referencia viva; (3) escribir en `accel/ref/` el modelo del motor (tiles, dos pasadas, vaciado, epílogo) y compararlo bit a bit con las capas del paquete; (4) cerrar el presupuesto de memoria y las decisiones 9 y 10 con números; (5) crear `accel/DECISIONES.md` con lo de la sección 4; (6) recién entonces `pe.sv` y su banco de pruebas.

## 6. Contrato con el proxy y el mock

`fpgate-accel` es un proceso C++ en el HPS que orquesta el acelerador (CSR y DMA) y habla con el proxy por un socket Unix. Formato en `proxy/src/accel_proto.h` (fuente de verdad), little-endian, cabecera fija de 24 bytes:

- **Petición** `FPGA`: versión 1, tipo (1 clasificar, 2 estado), canales, formato de píxel (0 = RGB uint8 HWC), `request_id`, ancho, alto, longitud, y los píxeles.
- **Respuesta** `FPGR`: estado (`OK`, `BUSY`, `HW_ERROR`, `UNSUPPORTED`, `DEGRADED`), `request_id`, `score_nsfw` (float 0 a 1), `hw_cycles` (32 bits), `latency_us`.
- El proxy envía **RGB uint8 de 448×448 ya redimensionado con squash** (se cambió el 2026-11-08 para coincidir con E1: antes recortaba el centro). `fpgate-accel` hace todo lo específico del modelo: pasa a CHW flotante, normaliza (media 0.48145466, 0.4578275, 0.40821073; desviación 0.26862954, 0.26130258, 0.27577711), cuantiza y devuelve **solo el score**. El umbral y la política ante fallos son del proxy.
- El proxy atiende una imagen a la vez hacia el acelerador, con cola de 8 y tiempo límite de 10 s por imagen; ante fallos entra en modo degradado y vuelve solo.
- **Mock:** `proxy/tests/mock_accel/mock_accel.py` (configurable) y `mock_accel.c` (corre en la placa). Procesan una imagen a la vez, aceptan solo 448×448 por defecto (`--size`), y su `--mean` califica por brillo medio: **no usan el acelerador**. `test_mock_accel.py` sirve como especificación de comportamiento para el servidor real.
- El protocolo no lleva un código de error de hardware; se podría usar el byte reservado de la respuesta (decisión aplazada a DES-4).

## 7. La placa y el entorno

- DE10-Standard, Cyclone V `5CSXFC6D6F31C6N` (41 910 ALMs, 112 DSP, 553 M10K). HPS: ARMv7 con NEON, Linux 4.5, glibc 2.20, hostname `socfpga`, acceso por SSH (`scp -O`) o UART. La IP cambia por DHCP. `gatord` ocupa el puerto 8080 después de cada arranque. Los binarios para la placa se compilan **estáticos con musl** (`arm-linux-musleabihf`), porque con glibc estática se cae `getaddrinfo`.
- Modelo: Docker (`model/docker-compose.yml`, servicios `tf` y `torch`; los scripts se corren con `docker compose run --rm torch quant/...`). Proxy: Docker (`proxy/docker-compose.yml`, `docker compose run --rm dev make ...`).
- **No sé qué herramientas de FPGA hay instaladas** (Quartus y su versión, simulador: ModelSim/Questa o Verilator) ni si la tarjeta está disponible. Pregúntalo al empezar.
- Particularidades de este equipo (Windows): `python3` del anfitrión puede ser un acceso directo de la tienda que se cuelga, usa el del contenedor; Git avisa de conversiones CRLF/LF, es normal; en el contenedor el ASan de clang falla al arrancar algunas veces por la aleatorización de memoria de WSL2 (no es un fallo del código).

## 8. Cómo prefiere trabajar el usuario

- Código y comentarios en **inglés**; documentación y explicaciones en **español**.
- Commits: *conventional commits*, en inglés, cortos, **sin `Co-Authored-By`**. Propón el mensaje y espera la aprobación antes de commitear.
- **No hagas push ni pull request si no se pide.** Agrupa los commits por tema y deja fuera lo que no es del cambio (las guías y las maquetas sin versionar).
- Prefiere que se le planteen las decisiones de una vez, con una recomendación y las alternativas, y que se mida antes de optimizar (por ejemplo, las mediciones de latencia esperan al acelerador real).
- Antes de tocar algo que dependa de una cifra, recalcula con los datos del paquete en lugar de fiarte de la guía original: varias cifras de la guía anterior (12 cabezas, 85 MB, 107 GMAC) correspondían al modelo sin podar.

## 9. Pendientes fuera de DES-2

- Mediciones de latencia del proxy (sin imágenes y con imagen) con el acelerador real, incluida la carga de CPU del HPS con el acelerador corriendo.
- Filtro de imágenes diminutas (`--min-side`) y caché de veredictos por contenido en el proxy: se decide con los tiempos reales.
- Volver a medir el preprocesamiento en la placa con squash (las mediciones guardadas son con recorte).
- Hacer push de los dos commits locales cuando el usuario lo pida.

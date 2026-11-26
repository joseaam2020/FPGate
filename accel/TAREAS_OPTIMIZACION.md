# Tareas de optimización (ejecutables por agentes)

Fuente: [PLAN_OPTIMIZACION.md](PLAN_OPTIMIZACION.md). Alcance acordado: **F0–F4 completo**. Ejecución **mixta por
modelo**. Cambios de sistema permitidos: limpieza del `.qsys`/top, árbitro propio, módulo de kernel/DTB y proxy.
Compilaciones de Quartus: **las necesarias**, sin reintentos a ciegas.

## Cómo ejecutar gastando pocos tokens

1. **Una sesión nueva por tarea**, con el modelo que indica la tarea: `claude --model haiku|sonnet|opus`, o pídeme
   "lanza T10 con un subagente sonnet". Pega el bloque **Contexto común** y después el bloque de la tarea. Nada más.
   Si coordina un agente (recomendado: desde una sesión nueva y corta), basta con decirle al subagente: "Lee el
   Contexto común y el bloque ### T10 de accel/TAREAS_OPTIMIZACION.md y ejecútalo". Él lee los dos bloques, y así
   el coordinador no tiene que copiar el texto.
2. El agente **no explora el repositorio** ni lee `DECISIONES.md`, `CONTEXTO.md` ni los planes. Lee solo los
   archivos y líneas que lista la tarea. Si algo no cuadra con lo descrito, se para y lo dice: no improvisa.
3. **Respuesta final del agente: como mucho 10 líneas** (archivos cambiados, resultado de las pruebas, commit).
   Sin resúmenes del código.
4. **Tareas de placa (`Ejecuta: tú`):** las corres tú con el script indicado y pegas solo las líneas `RESULT` en la
   siguiente sesión que las necesite.
5. Si una tarea falla dos veces en lo mismo, se para y se sube de modelo (haiku → sonnet → opus). No hay que
   insistir con el mismo modelo.

---

## Contexto común (pegar al inicio de cada tarea)

```
Proyecto FPGate: acelerador INT8 de un ViT (EVA02 podado "r7") en una DE10-Standard (Cyclone V SoC).
Repo: c:\Users\josea\Documents\FPGate (Windows; Git Bash y PowerShell disponibles). Rama JoseArroyo-Accelerator.
- accel/hps/src: fpgate-accel en C (model.c, ops.c, accel_drv.c/.h, accel_sim.c, fpga_pack.c/.h, main.c).
- accel/rtl: SystemVerilog del core. accel/tb: testbenches ModelSim. accel/quartus_hps: proyecto Quartus 18.1 Lite.
- proxy/: proxy HTTPS en C (OpenSSL 1.0.1p en la placa).
Builds y pruebas (desde la raíz del repo):
  docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make native        (también: arm, test-kernels, test-sim, test-proxy)
  powershell -File accel/tools/sim_tb.ps1 tb_a [tb_b ...]                                  (ModelSim; imprime PASS/FAIL)
  proxy: docker run --rm -v "${PWD}:/repo" -w /repo/proxy fpgate-proxy-dev make test
REGLAS DURAS:
1. Exactitud bit a bit: los logits deben ser idénticos al backend de CPU. Nunca uses -ffast-math, FMA ni
   reordenes operaciones float. Si un cambio no puede probarse exacto, no lo hagas.
2. Todo cambio de software va detrás de una opción de línea de comandos que restaura la ruta anterior
   (se indica el nombre). La ruta antigua no se borra.
3. No toques la placa: no ejecutes ssh, scp ni nada contra el hardware. Si hace falta, escribe un script y para.
4. Código y comentarios en inglés. Commit solo si la tarea lo pide: conventional commits con scope, en inglés,
   una sola línea, sin cuerpo y SIN línea Co-Authored-By; añade solo los archivos de la tarea (git add <archivos>).
5. No leas DECISIONES.md, CONTEXTO.md ni PLAN_OPTIMIZACION.md. De accel/TAREAS_OPTIMIZACION.md lee solo este
   bloque y el de tu tarea (localízalo con grep "^### <ID> " y lee hasta el siguiente "^### "). Fuera de eso, lee
   solo lo que la tarea indica.
6. Respuesta final: como mucho 10 líneas (archivos, pruebas con su resultado, hash del commit).
```

---

## Orden y dependencias

| ID | Fase | Modelo | Depende de | Ejecuta | Verificación |
|---|---|---|---|---|---|
| T00 | F0 | haiku | – | agente + tú | script de información de la placa |
| T01 | F0 | haiku | – | agente | logits de referencia de CPU + script de compuerta |
| T02 | F0 | sonnet | – | agente | `--trace` (Gantt) |
| T03 | F0 | sonnet | – | agente | simulador: `RAW32`, entrada sin signo, unidad de softmax concurrente |
| T04 | F0 | sonnet | – | agente | driver: softmax asíncrono |
| T05 | F0 | sonnet | T04 | agente + tú | prueba de concurrencia y ancho de banda en la placa |
| T06 | F0 | haiku | – | agente + tú | microbancos del ARM (`hps_bench`) |
| T07 | F0 | sonnet | – | agente | contadores de ciclos en simulación (sin tocar RTL) |
| T08 | F0 | haiku | – | agente | informe de hold sobre la base de datos de la #4 |
| T09 | F0 | – | T00–T08 | **tú** | medición F0 en la placa → líneas `RESULT` |
| T10 | F1 | sonnet | T03, T04, T09 | agente | atención solapada por cabezas (S1) |
| T11 | F1 | sonnet | T03 | agente | `patch_embed` en el motor (P) |
| T12 | F1 | sonnet | – | agente | cuantizar con NEON (B1) |
| T13 | F1 | – | T10–T12 | **tú** | compuerta F1 en la placa |
| T20 | F2 | **opus** | T13 | agente | atención por trozos dentro de la tubería (C) |
| T21 | F2 | sonnet | T09 | agente | LayerNorm: división exacta más rápida (B2) |
| T22 | F2 | sonnet | T00 | agente + tú | región con caché (B3), **solo si T00 lo permite** |
| T23 | F2 | sonnet | – | agente | proxy: caché de veredictos + tamaño mínimo |
| T24 | F2 | – | T20–T23 | **tú** | compuerta F2 + Gantt → decide el orden F3/F4 |
| T25 | F2 | **opus** | T22 (hecha) | agente + tú | cola del MLP fusionada por grupos de 16 filas (−1 s est.) |
| T26 | F2 | sonnet | T25 | agente + tú | residual + LayerNorm + preparación fusionados (−0.5 s est.) |
| T27 | F2 | sonnet | T25 | agente + tú | el motor arranca la siguiente operación sin esperar al paso del ARM |
| T30 | F3 | haiku | T00 | agente | limpieza del `.qsys` y del top |
| T31 | F3 | sonnet | – | agente | árbitro propio de `f2h_sdram0` |
| T32 | F3 | sonnet | – | agente | valla de direcciones en los DMA |
| T33 | F3 | sonnet | T08 | agente | arreglos de timing + hold |
| T34 | F3 | **opus** | T07, T37 | agente | acumulador sombra: vaciado solapado (−0.3 s en `S`, ≈ −0.2 s en lineales, est.) |
| T37 | **F1 (puente)** | sonnet | – | agente + tú | FIFO de salida con crédito + aborto sin bloquear el puerto; compilación a 120 MHz → `h5b` |
| T35 | F3 | haiku | T30–T34, T37 | agente | síntesis, compilación y `.rbf` (bucle de frecuencia) |
| T36 | F3 | – | T35 | **tú** | bitstream nuevo en la placa + compuerta |
| T40 | F4 | haiku | – | agente | ¿tablas de RoPE separables? |
| T41 | F4 | sonnet | T40, T36 | agente | RTL: escritor de formatos + RoPE en el epílogo |
| T42 | F4 | sonnet | T36 | agente | RTL: SwiGLU fusionado en fc1 |
| T43 | F4 | sonnet | T41, T42 | agente | software para T41/T42 |
| T44 | F4 | haiku | T41–T43 | agente | compilación (como T35) |
| T45 | F4 | – | T44 | **tú** | placa + compuerta final |

Etiquetas git (las crea el agente de la última tarea de cada fase, o tú):

- antes de F1: `accel-opt-f0`;
- antes de F2: `accel-opt-f1`;
- antes de F3: `accel-opt-f2`;
- antes de F4: `accel-pre-f4`;
- tras cada bitstream bueno: `accel-bit-<MHz>`.

Bitstream de retorno: `accel/rbf_out/h5_120MHz/soc_system.rbf`.

---

## F0 — Medir

### T00 · Información de la placa (haiku)
**Leer:** nada.
**Hacer:** crear `accel/tools/board/board_info.sh`, en POSIX sh y solo de lectura. Imprime, cada uno con su línea
`RESULT <clave>: ...`:
- `uname -a`;
- `zcat /proc/config.gz | grep -E 'STRICT_DEVMEM|CONFIG_CMA|CONFIG_DMA_CMA|CONFIG_MODULES|CONFIG_UIO'` (o `NO_CONFIG_GZ` si no existe);
- `cat /proc/iomem`;
- `ls /lib/modules/$(uname -r) 2>&1 | head`;
- los nodos del árbol de dispositivos bajo `/proc/device-tree` cuyo nombre contenga `ff2` o `bridge` (nombre + `compatible` + `reg` en hex con `hexdump -C`);
- `cat /proc/cmdline`;
- `fw_printenv 2>/dev/null | head -50` (o `NO_FW_PRINTENV`).

**No hacer:** escribir nada en la placa.
**Aceptación:** `sh -n board_info.sh` sin errores. Commit `feat(accel): board info script`.
**Tú:** `scp -O accel/tools/board/board_info.sh root@socfpga:/home/root/ && ssh root@socfpga sh board_info.sh > board_info.txt`.
Guarda `board_info.txt` en el PC (sin commit) y pásalo a T22 y T30.

### T01 · Referencia de logits + script de compuerta (haiku)
**Leer:** `accel/hps/src/main.c` (solo el parseo de opciones: busca `--selftest`, `--logits`, `--fpga`).
**Hacer:**
1. Genera `accel/hps/tests/ref_logits_20.txt` con el backend de CPU en nativo (Docker), con `--selftest 20 --logits`
   y el paquete de `PKG` del `Makefile`. Guarda solo las líneas de logits.
2. Crea `accel/tools/board/gate.sh <flags extra>`. Ejecuta en la placa
   `./fpgate-accel --pkg pkg --index e1_r7.idx --selftest 20 --logits --fpga --fpga-attn <flags>`, extrae los logits
   con el mismo filtro, los compara con `diff` contra `ref_logits_20.txt` (que se copia junto al script) e imprime:
   - `RESULT gate: PASS|FAIL (<n> líneas distintas)`;
   - `RESULT time: <min>-<max> s/img`;
   - la línea de resumen de tiempos que ya imprime el programa.

**Aceptación:** el filtro funciona en nativo: correr `--selftest 20 --logits` otra vez y compararlo con la referencia
da PASS. Commit `test(accel): reference logits and board gate script`.

### T02 · Traza de eventos `--trace FILE` (sonnet)
**Leer:** `accel/hps/src/model.c` líneas 40–80 (struct), 571–645 (`pl_run`), 709–870 (`model_forward`), 912–983
(`attention_fpga`), y el parseo de opciones de `main.c`.
**Hacer:**
- Buffer estático de eventos `{double t0, t1; char kind[8]; char name[24]; int block, chunk_or_head; uint64_t cycles;}`
  (máx. 65 536; si se llena, se deja de registrar y se avisa una vez).
- Registrar:
  - cada operación del motor (inicio = `accel_hw_start`, fin = cuando `accel_hw_busy` pasa a 0 la primera vez);
  - cada softmax;
  - cada paso de `pl_arm` (tipo de paso);
  - el empaquetado de cada cabeza;
  - `patch_embed`, la cabeza final y la espera activa (`idle`).
- Al terminar cada imagen, escribir un CSV `t0,t1,kind,name,block,idx,cycles`, con tiempos relativos al inicio de
  la imagen.
- Opción `--trace FILE`. Sin ella, coste cero: una sola comprobación de puntero nulo.

**Aceptación:**
- `make native`;
- con `--sim --trace /tmp/t.csv` sobre 1 imagen, el CSV tiene eventos de los tipos `eng`, `arm` e `idle`;
- `make test-sim` en PASS;
- los logits no cambian con `--trace`.

Commit `feat(accel): per-image event trace`.

### T03 · Simulador: `RAW32`, entrada sin signo y unidad de softmax (sonnet)
**Leer:** `accel/hps/src/accel_sim.c` (completo, 205 líneas), `accel_drv.h` (completo), `ops.c` líneas 208–264
(`softmax_row_int`, `attention_head_ref`), `fpga_pack.h` (completo).
**Hacer:**
1. El motor simulado soporta `raw32` (acumuladores INT32 con `stride_out`), `in_unsigned` (P en UQ0.16, dos
   planos) y `stride_out` en todas las salidas.
2. Una **segunda hebra** simula la unidad de softmax:
   - lee `S` (INT32, `s_stride`) desde la región;
   - aplica `softmax_row_int` fila a fila;
   - escribe `P` en el formato de activaciones de 16 bits (dos planos, KP palabras, relleno a cero; mismo formato que
     espera `att_check_stages` en `model.c` 896–901);
   - escribe de forma progresiva y con retardo, como el motor simulado.

   Como `softmax_row_int` necesita el bloque, añade a `accel_sm_op_t` un campo `const void *sim_ctx` que el
   hardware ignora (el modelo pasará `pkg`+`block`).
3. Motor y softmax deben poder estar en vuelo **a la vez** (hebras y estados separados).

**Aceptación:**
- prueba nueva `tests/test_sim_attn.c` (añadir a `test-kernels`, solo en nativo): una cabeza real (bloque 0, cabeza 0
  del paquete) por el simulador da el mismo `ctx` que `attention_head_ref`;
- control negativo: leer `P` antes del final da diferencias;
- `make test-sim` en PASS.

Commit `test(accel): software model of RAW32, unsigned input and the softmax unit`.

### T04 · Driver: softmax asíncrono (sonnet)
**Leer:** `accel/hps/src/accel_drv.c` 140–269, `accel_drv.h`.
**Hacer:**
- `int accel_hw_sm_start(h, op)`, `int accel_hw_sm_busy(h)`, `int accel_hw_sm_wait(h, op, st, timeout)`.
- Misma secuencia de registros que `accel_hw_run_sm`. Campo `sm_in_flight` independiente de `in_flight`.
- `sm_start` falla si ya hay un softmax en vuelo.
- `accel_hw_run_sm` pasa a ser `start` + `wait`.
- Con `h->sim`, delega en el simulador de T03 (si T03 aún no existe, deja el *hook* con un TODO y que falle).

**Aceptación:** `make native arm test-sim` en PASS; las secuencias de registros de `accel_hw_run_sm` antes y después
son idénticas (revisión del diff). Commit `feat(accel): asynchronous softmax unit API`.

### T05 · Prueba de concurrencia en la placa (sonnet)
**Leer:** `accel/tools/board/accel_test.c` (estructura de subcomandos y una prueba existente como plantilla),
`accel_drv.h`.
**Hacer:** subcomando `accel_test concurrent [rows]` (por defecto 1025):
1. Prepara datos deterministas en la región, todos dentro de `0x30000000 + [0x6000000, 0x8000000)`:
   - Q y Kᵀ aleatorios de una cabeza (hd = 64);
   - `S` sintética para el softmax en un buffer aparte;
   - constantes `m0`/`sh` del bloque 0 (o fijas y documentadas).
2. Ejecuta, con `CYCLES`/`BYTES_*` de cada unidad: (a) `S` sola; (b) softmax sola; (c) las dos a la vez (inicia las
   dos y espera a ambas); (d) igual que (c) más una hebra que hace `memcpy` de 4 MB a la región.
3. Comprueba que las salidas de (c) y (d) son idénticas byte a byte a las de (a) y (b).
4. Imprime `RESULT conc: S <ciclos> alone / <ciclos> conc; SM <ciclos> alone / <ciclos> conc; MB/s ...; outputs identical|DIFFER`.
5. Con `rows` = 32 primero (modo humo).

**No hacer:** tocar direcciones fuera de esa ventana.
**Aceptación:** compila con el toolchain ARM del Makefile de `tools/board` (o el comando de su README).
Commit `test(accel): engine and softmax concurrency test`.

### T06 · Microbancos del ARM (haiku)
**Leer:** `accel/tools/board/hps_bench.c` (completo), `accel/hps/src/ops.c` 130–166 (`layer_norm`).
**Hacer:** añadir a `hps_bench`:
- lectura de la región con `ldm` de 8 registros (asm en línea) y con `vld1.8 {d0-d7}` en bucle (128 B por iteración);
- LayerNorm de 1025 × 768 partida en tres fases cronometradas por separado: sumas, varianza, bucle de normalización
  con división.

Imprime `RESULT bench: <nombre> <MB/s o ms>`.
**Aceptación:** compila en ARM. Commit `test(accel): extra HPS microbenchmarks`.

### T07 · Desglose de ciclos en simulación, sin tocar RTL (sonnet)
**Leer:** `accel/tb/tb_cycles.sv`, `accel/tb/tb_attn_core.sv`, `accel/rtl/attn_softmax.sv` líneas 55–70 (estados),
`accel/rtl/tile_sequencer.sv` (solo la declaración de estados).
**Hacer:** en los testbenches (no en `rtl/`), contadores por referencia jerárquica:
- ciclos en cada estado de `u_softmax.state` (`S_RD`, `S_MAXD`, `S_P2`, `S_P2D`, `S_RCP`, `S_P3`, `S_P3D`, `S_OUT`, …);
- en el secuenciador, ciclos por estado, más ciclos con el arreglo calculando frente a vaciando.

Ejecutar:
- `S` (RAW32, M = 64, N = 64, K = 64);
- `P·V` (16 b sin signo, M = 112, N = 64, K = 1026, spp = 7);
- softmax de 32 filas.

Imprimir `RESULT cyc: <op> <estado>=<ciclos> ...`.
**Aceptación:** los testbenches siguen en PASS; las líneas `RESULT` aparecen. Commit `test(accel): per-state cycle breakdown in simulation`.

### T08 · Informe de hold de la compilación #4 (haiku)
**Leer:** nada.
**Hacer:** `accel/quartus_hps/report_hold.tcl` para `quartus_sta -t`:
- abre el proyecto `DE10_Standard_GHRD`;
- `create_timing_netlist -model slow -temperature 0` (y luego `fast`), `read_sdc`, `update_timing_netlist`;
- `report_timing -hold -npaths 5 -detail full_path -to_clock [get_clocks *pll_0*divclk*] -file hold_paths.txt`.

Ejecútalo (`C:\intelFPGA_lite\18.1\quartus\bin64\quartus_sta.exe -t report_hold.tcl` desde `accel/quartus_hps`).
Responde en 5 líneas como mucho: origen, destino, si el camino atraviesa `chainin`/`scanin`/`chainout` de DSP, y el
nombre exacto del registro duplicado (`~DUPLICATE`).
**Aceptación:** `hold_paths.txt` generado. Commit solo del `.tcl`: `chore(accel): hold path report script`.

### T09 · Medición F0 en la placa (**tú**)

Copia `fpgate-accel` (build ARM con T02), `accel_test`, `hps_bench`, `gate.sh` y `ref_logits_20.txt`. Luego, en orden:

1. `sh board_info.sh` (si no lo hiciste en T00).
2. `./accel_test concurrent 32` y, si está bien, `./accel_test concurrent`.
3. `./hps_bench`.
4. `sh gate.sh --trace /home/root/t256.csv`, y repetir con `--chunk 128`, `--chunk 192` y `--chunk 320` (los CSV
   con otro nombre).
5. Trae los CSV al PC.

Pega en la sesión de T10 y T20 solo las líneas `RESULT` y la línea de resumen de cada ejecución.
**Compuerta:** `gate` en PASS y `outputs identical` en la prueba de concurrencia.
Si la concurrencia cuelga la placa o da `DIFFER`: **para**. T10 cambia a la variante "solo empaquetado solapado"
(ver T10) y se investiga el interconnect.

---

## F1 — Software rápido

### T10 · Atención solapada por cabezas (S1) (sonnet)
**Leer:** `model.c` 14–30 (mapa de la región), 876–983 (`att_check_stages`, `attention_fpga`), 985–1058
(`model_attach_fpga`); `accel_drv.h`. Datos de T09: líneas `RESULT conc`.
**Hacer:**
1. `FPGA_REGION_LEN` → `0x8000000` (comprueba con grep dónde se usa para mapear). Juegos de buffers de atención por
   paridad `p ∈ {0,1}` en `ATT2_OFF = 0x6000000 + p·0xA00000`: Q +0, K +0x20000, V +0x40000, S +0x60000,
   P +0x560000. El contexto sigue en el `ATT_C` actual.
2. `attention_fpga_ovl()`, cola del motor en este orden: `S0, S1, PV0, S2, PV1, …, S(H−1), PV(H−2), PV(H−1)`.
   El softmax va en orden 0..H−1. Dependencias (el buffer de paridad `h%2` se reutiliza cada 2 cabezas):
   - pack(h) requiere PV(h−2) terminado;
   - S(h) requiere pack(h) y SM(h−2) terminados;
   - SM(h) requiere S(h) y PV(h−2) terminados;
   - PV(h) requiere SM(h) terminado.
3. Bucle de sondeo sin bloquear (`accel_hw_busy`, `accel_hw_sm_busy`):
   - recoger lo terminado;
   - lanzar en cada unidad lo siguiente de su cola si se cumplen sus dependencias;
   - si no, empaquetar la siguiente cabeza permitida;
   - si no, espera activa.

   Tiempo límite de 60 s por operación, como ahora.
4. `att_check_stages` recibe las direcciones de `S`/`P` de la paridad de la cabeza 0.
5. Opción `--no-att-overlap`: usa el `attention_fpga` actual sin cambios.
6. Variante "solo empaquetado solapado", para el caso en que T09 diera fallo de concurrencia: opción
   `--att-overlap-pack-only`, que nunca tiene el softmax y el motor en vuelo a la vez.
7. **Seguridad con el bitstream h5.** T09 vio que, con el softmax en vuelo, `S` puede desbordar su FIFO de salida
   (`ERR_CODE` 5). Tras ese error, el puerto de la DDR3 queda bloqueado hasta reiniciar la placa, así que **no hay
   reintento posible**.
   - Por defecto, el solapamiento completo (softmax ∥ motor) **solo** se activa si `VERSION` del motor es la del
     bitstream con T37 (`hw->version >= 0x…0002`; lee la constante en `accel_drv.c`/`csr.sv` y documéntala).
   - Con cualquier otro bitstream se usa `--att-overlap-pack-only` automáticamente, con un aviso por `stderr`.
   - `--att-overlap-force` fuerza el modo completo (solo para pruebas).

**Aceptación:**
- `make native arm test-sim`;
- con `--sim --fpga-attn --check --blocks 2`: "identical" en todas las comprobaciones;
- logits idénticos a 9 cifras con y sin `--no-att-overlap` (2 imágenes, `--sim`).

Commit `perf(accel): overlap softmax, engine and packing across attention heads`.

### T11 · `patch_embed` en el motor (P) (sonnet)
**Leer:** `model.c` 709–750 (`patch_embed`), 150–210 (`linear_fpga_seq`, como modelo de operación), 985–1023 (pesos
empaquetados y `w_off`); `accel_drv.h` (`accel_op_t`).
**Hacer:** si `m->hw` y `w_off[patch_embed]` existe (si no existe, empaquétalo igual que las demás capas; se cumple
`fpga_supports`: N = 768, K = 588):
1. Desplegar y cuantizar como ahora (`m->cols`, `quantize_f32` → `m->a8`).
2. `fpga_pack_acts8` → staging → `memcpy` a `FPGA_A_OFF`.
3. Operación con `M = NP (1024), N = 768, K = 588, in_bits = 8, raw32 = 1, out_bits = 32, stride_out = 768·4, spp = 16`,
   salida en `FPGA_O_OFF`.
4. Copiar los INT32 a `m->acc` y seguir con **el mismo bucle** de bias y descuantización de las líneas 732–739.
5. Poner `m->a_ok = 0`.

Opción `--cpu-patch` para la ruta actual.
**Aceptación:**
- `make native arm test-sim` (el simulador ya hace RAW32 desde T03);
- `patch_out` idéntico (añade comparación con `--check`: calcula también en CPU y cuenta diferencias);
- logits iguales con y sin `--cpu-patch`.

Commit `perf(accel): patch embedding on the engine with RAW32 output`.

### T12 · Cuantizar con NEON, exacto (B1) (sonnet)
**Leer:** `ops.c` 1–44 (`quantize_f32`), `tests/test_quant.c`.
**Hacer:** ruta NEON en `quantize_f32` bajo `#ifdef __ARM_NEON`, 4 floats por iteración:
- `t = x·inv` (`vmulq_f32`);
- `r = (t + magic) − magic`;
- `d = |t − r|`;
- candidato a empate si `|d − 0.5| < 1e-5·|t| + 1e-6`. Es la misma condición que la escalar, porque
  `|t − round(t)| = min(f, 1 − f)` con `f = |t − trunc(t)|`.

Si algún carril es candidato, o `|t| ≥ 2^22`, recalcular **esos** elementos con el código escalar actual. Después,
`vcvtq_s32_f32`, saturación y estrechamiento (`vqmovn`) a 8 o 16 bits. Mantener el `omp parallel for` por bloques
de 4. Justificación de la exactitud (ponla en un comentario): NEON redondea al más cercano en números normales; si
`t` sale denormal, el resultado es 0 en las dos rutas; todo caso dudoso cae a la ruta escalar exacta.
**Aceptación:** `make test-kernels`. `test_quant` debe:
- comparar la ruta NEON (qemu-arm) contra `rintf(x/sc)` en los 226 M valores existentes;
- añadir 50 M valores aleatorios con escalas de 1e-4 a 10, valores cercanos a ±0.5·k y saturación;
- dar 0 diferencias.

Commit `perf(accel): NEON quantisation, exact`.

### T13 · Compuerta F1 (**tú**)

1. `sh gate.sh --trace f1.csv`.
2. `sh gate.sh --no-att-overlap --cpu-patch` (debe dar los mismos logits).
3. `make test-proxy` en el PC y una prueba real con el proxy.

**Paso:** PASS en todo y ≤ 6.0 s. Etiqueta `accel-opt-f1`.

---

## F2 — Software estructural y extras

### T20 · Atención por trozos de filas de consulta dentro de la tubería (C) (**opus**) — **aplazada**
No lanzar hasta que el ARM de la tubería baje de los 2.6 s de trabajo real del motor (tras T25–T27 y F4): mientras
el ARM sea el límite, esconder el motor de la atención bajo el trabajo del ARM no ahorra nada.
**Leer:** `model.c` 360–707 (tubería completa), 912–983 (atención), el CSV `f1.csv` de T13 (solo las 200 primeras
líneas) y la función de T10.
**Objetivo:** que la atención del bloque b deje de ser una barrera para el ARM. El tramo siguiente empieza con el
trozo 0 mientras el motor y el softmax hacen la atención del trozo 1.
**Diseño exigido:**
- Pasos nuevos en `pl_run`: `PS_ATT_PACK` (ARM; empaquetar Kᵀ/V de todas las cabezas una vez por bloque y Q del trozo
  por cabeza), `PS_ATT_S` (motor), `PS_ATT_SM` (unidad de softmax) y `PS_ATT_PV` (motor; escribe las filas del trozo
  en `ATT_C`).
- La unidad de softmax es un **segundo recurso** con su propia ocupación.
- `P` y `S` por trozo y cabeza, con huecos en la región por encima de `0x6000000` (calcula el tamaño y compruébalo).
- La barrera real se mantiene: los trozos del bloque b+1 no empiezan la atención hasta que **todos** los trozos
  terminaron q, k, v y RoPE del bloque b+1.
- Prioridad del motor: la operación del trozo de menor índice.
- Si algo no cabe, se vuelve solo a la ruta de T10, con aviso por `stderr`.
- Requiere el bitstream con T37 (comprobación de `VERSION` como en T10, punto 7). Con h5, `--no-att-pipeline`
  automático y aviso.
- Opción `--no-att-pipeline`.

**Aceptación:**
- `make native arm test-sim test-proxy`;
- `--sim --check --blocks 2`: "identical";
- logits idénticos a 9 cifras frente a `--no-att-pipeline` en 2 imágenes;
- control negativo: forzar que un trozo empiece la atención antes de la barrera debe dar diferencias con `--check`
  (y después se quita).

Commit `perf(accel): attention in query-row chunks inside the pipeline`.

### T21 · LayerNorm: división exacta más rápida (B2) (sonnet)
**Condición:** hazla solo si T09 (`RESULT bench` de T06) muestra que el bucle con división es más del 40 % de la
LayerNorm.
**Leer:** `ops.c` 130–166, `tests/test_ln.c`.
**Hacer:** en el bucle final, `t = (xr[k] − mu)·rden` con `rden = 1/den` (float). Comprobación exacta en double:
- `e = (double)(xr[k]−mu) − (double)t·(double)den`. El producto es exacto (24 × 24 bits).
- `t` es el cociente correctamente redondeado si `|e| ≤ 0.5·ulp(t)·|den|`, comparando de forma estricta lejos del
  empate.
- En caso contrario, o cerca del empate, usar la división real.

Opción de compilación `-DLN_EXACT_DIV` para volver a la división siempre.
**Aceptación:** `test_ln` ampliado con 20 M filas aleatorias (incluidas las de varianza pequeña y valores grandes):
0 diferencias en nativo y en qemu-arm. Commit `perf(accel): faster exact LayerNorm division`.

### T22 · Región con caché (B3) — **hecha** (664f88c)
Módulo propio `accel/tools/kmod/fpgate_mem.ko` (sin CMA no sirve u-dma-buf): mapea la ventana 0x30000000 con caché
y ofrece `ioctl` de limpieza e invalidación. `accel_drv.c` lo usa si existe `/dev/fpgate_mem`; `model.c` llama a
`accel_hw_clean` tras cada copia a la región y a `accel_hw_inval` antes de cada lectura. Medido: 6.55 → 6.06 s/img,
PASS bit a bit. Ver `accel/tools/kmod/README.md`.

### Lo que enseñó la placa (léelo antes de T25–T27)
Traza `accel/board_runs/f1/f1d.csv` (h5b, región con caché, chunk de 192 filas). Pasos del ARM de **un chunk**
en ms: `read` 1.3 · `res` 1.7 · `ln` 3.5 · `prep` 5.5 · `read` 1.7 · `read` 1.6 · `swiglu` 6.5 · `ln`(3072) 5.7 ·
`prep`(fc2, 16 bits) 11.2 · `read` 1.3 · `res` 1.7 · `ln` 3.6 · `prep` 5.5 · `read` ×3 ≈ 2.6 · `rope` 4.6 ≈ 58 ms;
72 chunks ≈ 3.9 s, igual que el tiempo de pared del motor (3.9 s), cuyo trabajo real son 2.6 s.
El ARM no está limitado por cálculo sino por **bytes**: cada paso recorre la matriz entera del chunk (590 KB en
float, 1.18 MB en los int16 de fc1) y la escribe en otro buffer para el paso siguiente. NEON (T12) y la división
rápida (T21) no dieron nada por eso. La regla para T25–T27: **eliminar pasadas y buffers intermedios**, con cada
grupo de 16 filas tratado de principio a fin mientras cabe en la caché. Exactitud: las operaciones por elemento y
el orden de las sumas de cada fila no cambian, así que el resultado es idéntico bit a bit.

### T25 · Cola del MLP fusionada por grupos de 16 filas (**opus**)
**Leer:** `model.c`: `pstep_t`, `pl_arm` (casos `PS_READ`, `PS_SWIGLU`, `PS_LN`, `PS_PREP`), `pl_run`, `pl_block_body`
(donde se añaden `fc1g`, `fc1x`, `PS_SWIGLU`, `PS_LN` de `mlpn` y `fc2`), `pl_alloc` (buffers `pl_n/pl_a16/pl_stage`);
`ops.c`: `layer_norm`, `quantize_f32`; `fpga_pack.c`: `fpga_pack_acts16`, `pack16_sub_neon`;
`accel_drv.h`: `accel_hw_clean/inval`.
**Hoy, por chunk:** `READ(fc1g)` 1.7 + `READ(fc1x)` 1.6 + `SWIGLU` 6.5 + `LN(mlpn)` 5.7 + `PREP(fc2)` 11.2 = **26.7 ms**:
copia g y u de la región a `g16/u16` (2 × 1.18 MB), SwiGLU escribe `h16` y `hh` (2.36 MB float), LN lee `hh` y
escribe `pl_n` (2.36 MB), cuantiza a `pl_a16` (1.18 MB), empaqueta a `pl_stage` (1.18 MB) y copia a la región.
**Hacer:**
1. Nuevo paso `PS_MLP` que sustituye a esa secuencia de 5 pasos en `pl_block_body` cuando `!m->check` (con `--check`
   se conservan los 5 pasos: la verificación por capas no se toca).
2. `ops.c`: extrae de `layer_norm` una `layer_norm_row(const float *xr, int d, g, b, epsf, float *yr)` con el mismo
   código (los 4 acumuladores, el mismo orden) y haz que `layer_norm` la llame. Así las dos rutas son idénticas.
3. `PS_MLP`, por grupo de 16 filas del chunk (el último puede ser parcial):
   - `accel_hw_inval` de las filas de g y u que va a leer (ver el punto 4), y lectura **directa desde la región**
     (int16 en orden de filas, como hace `PS_READ`), sin pasar por `g16/u16`;
   - por fila (OpenMP sobre las 16 filas, 2 núcleos): SwiGLU con la misma fórmula de `PS_SWIGLU` a un buffer de fila
     `hh_row[3072]` (float, local), **sin escribir `m->h16` ni `m->hh`** (en la tubería nadie más los lee: compruébalo
     con grep y dilo); `layer_norm_row` → `n_row[3072]`; `quantize_f32(n_row, 3072, fc2->s_x, 16, a16_grp + fila)`;
   - con el grupo completo, `pack16_sub_neon`/el bucle escalar de `fpga_pack_acts16` **directamente en la región**:
     el subtile `s` va a `FPGA_A_OFF + slot·pl_slot + s·2·K·16` (`hi`) y `+ K·16` (`lo`), con ceros en las filas que
     falten del último grupo;
   - al final del chunk, un solo `accel_hw_clean` del área A del slot (`fpga_acts_bytes(nr, K, 2)`).
4. Las dos salidas de fc1 se escriben hoy en el **mismo** slot de salida (`FPGA_O_OFF + slot·pl_slot`), por eso
   `READ(fc1g)` ocurre antes de `RUN(fc1x)`. Elige y justifica una de las dos:
   a) `fc1g` escribe en un segundo slot de salida (hay que comprobar que `FPGA_O_OFF..FPGA_W_OFF` tiene sitio para
      `PL_MAX` chunks × 2 × `pl_slot` o reducir `PL_MAX`; si no cabe, descarta);
   b) se conserva `READ(fc1g)` → `g16` (1.7 ms) y `PS_MLP` lee g de `g16` y u de la región.
5. Traza: `tr_add("arm", "mlp", …)`; contadores: el tiempo del paso va a `pl_swiglu` (el resumen por etapas lo
   enseña como SwiGLU; anótalo en el mensaje del commit).
6. `make native arm test-sim` (todo PASS) y, a mano, `build/native/fpgate-accel --selftest 2 --fpga --sim --check`
   seguido de la misma orden sin `--check`: los logits deben ser idénticos entre ambas (una usa los 5 pasos y la
   otra `PS_MLP`).

**Tú:** `scp` del binario, `gate.sh --trace /home/root/f2a.csv`, y pega `RESULT`, la línea `total` y la salida de
`awk -F, 'NR>1 && $3=="arm" && $5==3 && $6==0 {printf "%s %.2f\n",$4,($2-$1)*1000}' f2a.csv`.
**Aceptación:** PASS (0 líneas distintas); en la traza, `mlp` ≤ **12 ms** por chunk de 192 filas (hoy 26.7 ms en 5
pasos); s/img ≤ 5.4 (est.). Commit `perf(accel): fused SwiGLU, LayerNorm and fc2 packing by 16-row groups`.

### T26 · Residual + LayerNorm + preparación fusionados (sonnet)
**Depende de T25** (usa `layer_norm_row` y el mismo patrón). **Leer:** lo mismo que T25 más `fpga_pack_acts8`,
`pack8_sub_neon`, `PS_RES`, `pl_block_head`.
**Hoy, por chunk, dos veces** (tras `proj` y tras `fc2`): `READ` 1.3 + `RES` 1.7 + `LN` 3.5 + `PREP`(8 bits) 5.5 =
**12 ms**: copia y → `y16`, `x += y·sc`, LN de `x` → `pl_n`, cuantiza → `pl_a8`, empaqueta → `pl_stage`, copia.
**Hacer:** paso `PS_RLP` (cuando `!m->check`) que, por grupo de 16 filas: invalida y lee y directamente de la región,
actualiza `m->x` fila a fila (sigue siendo el flujo residual: **sí** se escribe), `layer_norm_row` → `n_row[768]`,
`quantize_f32(…, 8, …)` al grupo `a8_grp`, `pack8_sub_neon` directamente en el área A del slot; un `accel_hw_clean`
al final. El de `fc2` cruza al bloque siguiente (`pl_block_head`): la LN es la `n1` del bloque `bi+1`. Traza
`"rlp"`; contadores a `pl_ln`.
**Tú:** igual que T25 con `f2b.csv`. **Aceptación:** PASS; `rlp` ≤ **5 ms** por chunk (hoy 12 ms en 4 pasos).
Commit `perf(accel): fused residual, LayerNorm and activation packing by 16-row groups`.

### T27 · El motor arranca la siguiente operación sin esperar al paso del ARM (sonnet)
**Leer:** `model.c` `pl_run` (los 4 puntos del bucle) y los pasos fusionados de T25/T26.
**Hoy:** el bucle de `pl_run` solo mira si el motor terminó **entre** pasos del ARM (2–11 ms), así que el motor
espera de media medio paso antes de recibir la siguiente operación: tiempo de pared del motor 3.9 s frente a 2.6 s
de ciclos (`f1d.csv`).
**Hacer:** saca los puntos 1 y 2 del bucle a `pl_engine_tick(m, …)` y llámala también desde los pasos fusionados
entre grupo y grupo de 16 filas (≈ 1 ms). Nada de hilos ni `usleep`. Cuidado con los slots: la operación que se
arranca no puede ser la del chunk cuyo paso del ARM está en curso (ya lo excluye `j == inflight`; mantenlo).
**Tú:** `f2c.csv`. **Aceptación:** PASS; en la traza, Σ(pared de `eng` no‑`att_`) − Σ(`cycles`)/120 MHz ≤ **0.4 s**
(hoy 1.3 s). Commit `perf(accel): poll the engine between row groups so it never waits for an ARM step`.

### T23 · Proxy: caché de veredictos y tamaño mínimo (sonnet)
**Leer:** `proxy/src/inspect.c` 780–870, `proxy/src/accel_client.h` (completo), `proxy/src/settings.h` y `settings.c`
(cómo se declara un ajuste).
**Hacer:**
1. Antes de `accel_classify` (inspect.c ~860): `SHA256(rgb, len)` de OpenSSL. Tabla LRU de tamaño fijo con ajuste
   `verdict_cache_entries` (por defecto 4096; 0 = apagado) que guarda `{hash, score}` **solo** de respuestas OK.
   - En un acierto, llamar al mismo camino que `on_accel_done` con el resultado guardado, y poner a cero y liberar
     `rgb` igual que haría `accel_classify`.
   - Contadores `cache_hits`/`cache_misses` en las estadísticas del panel, si es trivial; si no, solo en el log.
2. Ajuste `min_judge_px` (por defecto **0 = apagado**). Si `src_w < min || src_h < min`, la imagen pasa sin juzgar,
   con razón `small` en el log.

**Aceptación:** `make test` del proxy en PASS. Pruebas nuevas: misma imagen dos veces → una sola petición al
acelerador; con `min_judge_px = 64`, una imagen de 32 × 32 no llega al acelerador.
Commit `feat(proxy): verdict cache by image hash and minimum size to judge`.

### T24 · Compuerta F2 y decisión (**tú** + yo)

1. `gate.sh --trace f2.csv` y prueba con el proxy.
2. Pásame `f2.csv` (o sus 300 primeras líneas) y el resumen. Con eso decido:
   - si el motor está ocupado más del 85 % del tiempo de pared → F3 primero;
   - si no → F4 primero (o T22, si fue viable).

Etiqueta `accel-opt-f2`.

---

## F3 — Tanda de RTL nº 1 (cierre de tiempos)

Reglas de la tanda:
- todo se integra y se simula **antes** de compilar;
- el mapa de CSR y los formatos no cambian (el software de F1/F2 vale para los dos bitstreams);
- `VERSION` del motor sube a `…0002` y del softmax a `0x534D0002`, para distinguirlos.

### T30 · Limpieza del `.qsys` y del top (haiku)
**Leer:** `accel/quartus_hps/build_system.tcl` (completo), `DE10_Standard_GHRD.v` líneas 210–400,
`DE10_Standard_GHRD.qsf` líneas 1005–1020, y `board_info.txt` (nodos `ff2*` del DTB).
**Condición:** si el DTB tiene nodos con `compatible` de `altr,pio`, `altr,juart` o `altr,sysid` en `0xFF20xxxx`,
**para y avisa**. Si no, sigue.
**Hacer:**
- En `build_system.tcl`, tras `load_system`: `remove_instance` de `hps_only_master`, `fpga_only_master`,
  `f2sdram_only_master`, `ILC`, `jtag_uart`, `led_pio`, `button_pio`, `dipsw_pio` y `mm_bridge_0`, y quitar la línea
  `set_connection_parameter_value ... mm_bridge_0.s0 ...`. **Conservar:** `hps_0`, `clk_0`, `pll_0`,
  `fpgate_accel`, `sysid_qsys`.
- En el top:
  - quitar `debounce_inst`, `hps_reset_inst`, los 3 `altera_edge_detector` y las conexiones a los PIO quitados;
  - atar `hps_0_f2h_cold_reset_req_reset_n`, `..._warm_...` y `..._debug_...` a `1'b1`;
  - atar `stm_hw_events` a 0;
  - conservar `hps_fpga_reset_n` (h2f_reset) y los LED/pines como salidas fijas, sin borrar asignaciones de pines.
- Quitar del `.qsf` los `VERILOG_FILE` que ya no se usen.
- Ejecutar:
  - `qsys-script --script=build_system.tcl --search-path="ip/**/*,$"`;
  - `qsys-generate soc_system.qsys --synthesis=VERILOG`;
  - `quartus_map DE10_Standard_GHRD`.

**Aceptación:** `quartus_map` sin errores. Informa ALMs estimados y M10K (de `output_files/*.map.summary`).
Commit `chore(accel): remove unused GHRD peripherals and debug masters`.

### T31 · Árbitro propio de `f2h_sdram0` (sonnet)
**Leer:** `accel/rtl/fpgate_accel_core.sv` (completo), `accel/rtl/dma_rd.sv` y `dma_wr.sv` (interfaz y ráfagas;
BURST = 16, sin cruzar 4 KiB), `accel/tb/avalon_mem_model.sv`, `accel/quartus_hps/build_system.tcl`, y el
`_hw.tcl` en `accel/quartus_hps/ip/fpgate_accel_core/`.
**Hacer:**
1. `rtl/mem_arbiter.sv`: 4 maestros Avalon-MM de 256 bits (rd, wr, rd2, wr2) → 1 maestro.
   - Rotación (round-robin) por **ráfaga completa**: una escritura se queda el bus hasta su último beat.
   - Lecturas con hasta 16 ráfagas pendientes y FIFO de IDs (2 bits × 16) para encaminar `readdatavalid`. Los
     maestros nunca frenan `readdata`.
   - Registra todas las salidas hacia el puerto del HPS.
2. `fpgate_accel_core` expone un único maestro `mem_*` (32 bits de dirección, `burstcount` de 8 bits). Actualiza el
   `_hw.tcl`.
3. `build_system.tcl`: en lugar de las 4 conexiones a `hps_0.f2h_sdram0_data`, exportar los dos lados
   (`add_interface accel_mem avalon master` + `set_interface_property accel_mem EXPORT_OF fpgate_accel.mem`;
   `add_interface f2h_sdram0 avalon slave` + `EXPORT_OF hps_0.f2h_sdram0_data`) y unirlos en el top con cables
   directos.

   **Comprueba en el `soc_system.v` generado que no existe `mm_interconnect_3`** (ni ninguna FIFO de 2048). Si
   Platform Designer no deja exportar el esclavo del HPS: **para y dilo**.
4. `tb/tb_mem_arbiter.sv`: los 4 maestros con tráfico aleatorio y `avalon_mem_model` con esperas y latencia; los
   datos leídos y escritos deben ser correctos y respetar el orden por maestro.

**Aceptación:** `sim_tb.ps1 tb_mem_arbiter tb_core tb_attn_core tb_cycles` en PASS; `quartus_map` sin errores,
informando M10K (debe bajar unas 52). Commit `feat(accel): own arbiter for the HPS SDRAM port`.

### T32 · Valla de direcciones (sonnet)
**Leer:** `rtl/dma_rd.sv`, `rtl/dma_wr.sv`, y los `ERR_CODE` existentes en `rtl/csr.sv` y `rtl/csr_sm.sv`.
**Hacer:** parámetros `ADDR_LO = 32'h3000_0000` y `ADDR_HI = 32'h4000_0000`. Toda ráfaga cuyo inicio o fin caiga
fuera no se emite: el DMA levanta `err` → `ERR_CODE` nuevo (siguiente libre, documentado en el `.sv`) y la operación
termina con `ERROR`. Comprobación registrada (sin lógica combinacional nueva delante del puerto).
**Aceptación:** testbench que pide una dirección `0x2FFF_FFE0` → `ERROR` con el código nuevo, sin escrituras en la
memoria del modelo; los testbenches existentes siguen en PASS. Commit `feat(accel): DMA address window guard`.

### T33 · Arreglos de timing y hold (sonnet)
**Leer:** respuesta de T08 (5 líneas); `rtl/tile_buffers.sv` (escritura de la RAM A, `WR_PIPE`, `a_waddr_p`);
`rtl/attn_softmax.sv` (escritura a `softmax_rowram` ×16 y comparación del máximo a la entrada del `S_RD`);
`accel/fpgate-accel.sdc` o el `.sdc` del GHRD que se use (búscalo en el `.qsf`).
**Hacer:**
1. RAM A: `WR_PIPE` +1 y dirección y dato de escritura replicados por grupo de bancos, con `(* preserve *)`, para
   que cada copia alimente como mucho 8 M10K.
2. Softmax: registrar la difusión hacia las 16 RAM de fila en un árbol de 2 niveles (4 × 4) y registrar el dato
   leído antes de la comparación del máximo. Ajusta las latencias internas para que el resultado no cambie.
3. Hold, según T08:
   - si el camino es un registro `~DUPLICATE` → `set_instance_assignment -name ADV_NETLIST_OPT_ALLOWED "NEVER ALLOW" -to <registro original>` en el `.qsf`;
   - si pasa por `chainin`/`scanin` → registro explícito en el PE que rompa la cadena para ese caso.
4. **No** añadas `multicycle` (queda para otra compilación si hace falta).

**Aceptación:** los 14 testbenches existentes en PASS, sin cambiar sus resultados. Commit `fix(accel): pipeline the RAM A write and softmax row fan-out, hold fix`.

### T34 · Acumulador sombra: vaciado solapado con el cálculo (**opus**)
**Datos medidos (`tb_attn_core`, `RESULT cyc`):**
- `S` con 32 filas, 65 tiles y K = 64: `seq.RUN = 21 320` ciclos para 130 subtiles, es decir, **164 ciclos por
  subtile frente a 64 de cálculo**. `S_RUN` espera a `row_last_seen`, así que incluye llenado (≈ 32) y vaciado de 16
  filas a 4 ciclos por fila (64).
- En las capas lineales con K = 768, el mismo vaciado sin solape cuesta ≈ 11 % de cada subtile.
- En `P·V`, `LOADA_WAIT` = 17 % (fuera de esta tarea).

**Leer:** `rtl/pe.sv`, `rtl/gemm_array.sv` (contrato `busy`/`drain_ready`/`drain_last`), `rtl/linear_tile.sv`,
`rtl/epilogue_row.sv`, `rtl/tile_sequencer.sv` líneas 225–250, y la parte de `gemm_engine.sv` que retiene el segundo
beat de `RAW32`.
**Hacer:**
1. Cada PE copia su acumulador a un registro sombra al terminar el subtile. El vaciado (y el epílogo) lee las
   sombras, y el arreglo queda libre para el siguiente subtile.
2. `tile_sequencer` pasa de `S_RUN` a `S_DRAIN`/`S_SSTART` cuando el arreglo ha copiado a las sombras, no cuando sale
   la última fila. La FIFO de salida sigue protegida por el crédito de T37 (si T37 no está, intégrala aquí).
3. En `RAW32`, sacar un beat por ciclo (2 ciclos por fila) en lugar de 4 ciclos por fila.
4. **Cuidado con el timing:** son 256 × 32 bits nuevos junto a los multiplicadores (DSP al 100 %). Registra la copia,
   evita multiplexores anchos delante del vaciado (usa una cadena de desplazamiento por columna si sale más barato) y
   no metas lógica nueva en el camino del producto.

**Aceptación:**
- todos los testbenches en PASS, bit a bit;
- `tb_attn_core` con `S_32rows seq.RUN` ≤ 70 ciclos por subtile (antes 164), impreso antes y después;
- `tb_cycles` de `b0.q_w` (8 b, K = 768) con menos ciclos que antes;
- `quartus_map` con los ALMs nuevos (aviso si son más de 4 k).

Commit `perf(accel): shadow accumulators so the array computes while it drains`.

### T37 · FIFO de salida: control de crédito + abortar sin bloquear el puerto (sonnet) — **compilación puente, antes de T10**
Va sola en una compilación a **120 MHz** con la receta actual (sin limpieza ni cambios de reloj), para desbloquear
T10/T20 cuanto antes. Después, T35/T44 la incluyen en las tandas grandes.
**Leer:** `accel/rtl/gemm_engine.sv` líneas 160–275 (FIFO de salida `OUT_FIFO = 32`, `push`, `raw_hold`,
`overflow` → error 5); `accel/rtl/tile_sequencer.sv` líneas 40–60 (puertos) y 225–250 (`S_SSTART`, `S_DRAIN`);
`accel/rtl/dma_wr.sv` (completo); `accel/rtl/attn_softmax.sv`, solo cómo aborta (`abort`/`S_ERR`) y su `dma_wr`;
`accel/rtl/csr.sv` y `csr_sm.sv` (constante `VERSION`).
**Segundo problema (T09):** tras un error 5, la placa queda con el motor **y** el softmax sin poder transferir
(operaciones de ≈ 2²⁰ ciclos con 0 bytes) hasta reiniciar. Hipótesis: el aborto corta una ráfaga de escritura a
medias y el interconnect de `f2h_sdram0` se queda esperando los beats que faltan.
**Hacer también:**
- en `dma_wr` (los dos, motor y softmax), un aborto o reset suave con una ráfaga empezada **completa los beats que
  faltan con `byteenable = 0`** antes de volver a reposo; ídem para las lecturas pendientes: se consumen los
  `readdatavalid` que falten;
- `VERSION` del motor y del softmax suben en 1 (lo usa T10 para activar la concurrencia).

**Aceptación adicional:** testbench que provoca un error a mitad de una ráfaga de escritura: el modelo de memoria
recibe exactamente `burstcount` beats, y la siguiente operación termina bien y bit-exacta.

**Compilación (en la misma tarea, si todo pasa):** `ACCEL_MHZ` sigue en 120;
`quartus_sh --flow compile DE10_Standard_GHRD`. Se acepta si setup ≥ 0 en todas las esquinas y hold igual o mejor
que la #4 (−0.108 ns). `.sof` como `h5b_120MHz_DE10_Standard_GHRD.sof` y `.rbf` en `accel/rbf_out/h5b_120MHz/`.
**Problema:** en `RAW32` un subtile mete 32 entradas (16 filas × 2 beats) en una FIFO de 32. `S_DRAIN` →
`S_SSTART` arranca el siguiente subtile sin mirar si hay sitio. Si `f2h_sdram0` acepta escrituras despacio
(contención), la FIFO se desborda y la operación termina con `ERR_CODE` 5. En modo normal (16 entradas por subtile,
K ≥ 512) no pasa.
**Hacer:**
- `gemm_engine` exporta `fifo_free` (entradas libres, registrado).
- `tile_sequencer`, en `S_SSTART`: no levanta `sub_start`/`fd_start` hasta que `fifo_free ≥ (raw32 ? 32 : 16)`
  (estado de espera; el watchdog `wd` no debe saltar mientras `fifo_free` crece).
- No cambiar `OUT_FIFO` ni nada más.

**Aceptación:** testbench nuevo o variante de `tb_raw32`: modelo de memoria con `waitrequest` alto durante 200
ciclos cada 300 → antes del cambio da error 5 (compruébalo); después termina con la salida bit a bit correcta.
`tb_gemm_engine`, `tb_linear_tile`, `tb_attn_core` y `tb_cycles` en PASS, con ciclos sin cambios (±0.5 %) cuando no
hay esperas. Commit `fix(accel): do not start a subtile without output FIFO room`.

### T35 · Síntesis, compilación y `.rbf` (haiku)
**Leer:** nada más que este bloque.
**Hacer** (en `accel/quartus_hps`, Quartus en `C:\intelFPGA_lite\18.1`):
1. `quartus_map` → si M10K > 80 % o ALMs > 72 %, **para** y devuelve las cifras.
2. `build_system.tcl`: `ACCEL_MHZ 140` → `qsys-script` → `qsys-generate` → `quartus_sh --flow compile DE10_Standard_GHRD`
   (en segundo plano; tarda unos 60 min).
3. Leer `output_files/DE10_Standard_GHRD.sta.summary`.
4. Regla de frecuencia:
   - setup y hold ≥ 0 en las 4 esquinas → listo;
   - peor setup del reloj del PLL entre −0.3 y 0 ns → repetir a 135, y luego a 130;
   - setup < −0.3 ns, o hold < 0 → **para** y devuelve los 3 peores caminos (`report_timing` como en T08, pero de setup).
   - No probar semillas.
5. Con el bitstream bueno: copiar el `.sof` a `h6_<MHz>MHz_DE10_Standard_GHRD.sof`, generar
   `accel/rbf_out/h6_<MHz>MHz/soc_system.rbf` con `quartus_cpf -c -o bitstream_compression=on <sof> soc_system.rbf`,
   y guardar los dos `.summary`.

**Aceptación:** `.rbf` generado y tabla de slack de las 4 esquinas en la respuesta.
Commit `build(accel): h6 bitstream at <MHz> MHz` (solo `build_system.tcl`, los `.summary` y los archivos que ya se
versionen así).

### T36 · Bitstream nuevo en la placa (**tú**)

1. Haz copia de seguridad del `.rbf` actual de la SD como `soc_system_h5.rbf`.
2. Copia el nuevo `.rbf` (`mount /dev/mmcblk0p1 /mnt/fat && cp ... && sync`) y reinicia en frío.
3. Comprueba:
   - `cat /sys/class/fpga_manager/fpga0/state` = `operating`;
   - `devmem 0xFF200000` y `0xFF200100` (VERSION nuevos);
   - `./accel_test regs` (el reloj debe ser el nuevo);
   - `./accel_test full`, o el subcomando de capas completas que uséis;
   - `./accel_test concurrent`;
   - `sh gate.sh --trace f3.csv`;
   - prueba con el proxy.

**Si algo falla:** vuelve a copiar `soc_system_h5.rbf` como `soc_system.rbf` y reinicia. Pásame el error.
**Paso:** etiqueta `accel-bit-<MHz>`.

---

## F4 — Tanda de RTL nº 2 (descargar el ARM)

### T40 · ¿Tablas de RoPE separables? (haiku)
**Leer:** `accel/hps/src/pkg.h` (dónde están `cos_tab`/`sin_tab` del bloque) y `model.c`, función `rope_int_range`
en `ops.c` 176–206.
**Hacer:** un programa C pequeño (o Python con numpy en el contenedor `model-torch`) que carga el paquete r7 y
comprueba si, para el token `t = 1 + 32·y + x`, las columnas pares e impares de `cos_tab[t−1]` dependen solo de `y`
en una mitad de `hd` y solo de `x` en la otra (o si hay otra estructura). Responde: `separable: sí/no`, el patrón,
y el tamaño mínimo de tabla en bytes.
**Sin commit.**

### T41 · Escritor de formatos + RoPE en el epílogo (sonnet)
**Leer:** respuesta de T40; `rtl/epilogue_row.sv`, `rtl/requant_lane.sv`, `rtl/gemm_engine.sv` (camino de salida),
`rtl/dma_wr.sv`, `rtl/csr.sv` (campos de `OP`); `fpga_pack.h` (formatos Q, Kᵀ y V, que son el contrato que hay que
reproducir).
**Hacer:** campo nuevo de `OP` (bits libres, documentados) `OUTFMT`:

| Valor | Salida |
|---|---|
| 0 | fila-mayor (actual) |
| 1 | activaciones tile-major (Q de una cabeza, y ctx para `proj`) |
| 2 | Kᵀ: tiles de 16 claves, saltando los 8 beats de constantes que el ARM escribe una vez |
| 3 | V: tiles de 16 canales, con K = KP palabras |

Más un registro `HEAD_DIM`/`HEADS` para repartir por cabezas. Transpositor 16 × 16 de bytes con doble buffer en
registros o MLAB.

RoPE opcional (bit `ROPE` en `OP`), solo si T40 dio una tabla de 16 KiB o menos:
- tabla en M10K cargada por un puerto de escritura de CSR (como la tabla SiLU de `swiglu_lane`);
- fórmula exacta de `rope_int_range` (interleaved; token 0 solo con desplazamiento; saturación a ±127);
- índice de token = fila global de la salida (registro `ROW0`).

`tb_outfmt.sv` con vectores reales: generar q/k/v de una cabeza con `fpgate-accel --dump-attn` (ya existe) y
comparar byte a byte con `fpga_pack_*`.
**Aceptación:** `tb_outfmt` y los testbenches existentes en PASS; `quartus_map` con los recursos nuevos.
Commit `feat(accel): output layout writer and RoPE in the epilogue`.

### T42 · SwiGLU fusionado en fc1 (sonnet)
**Leer:** `rtl/swiglu_lane.sv` y `tb/tb_swiglu_lane.sv`; `rtl/tile_sequencer.sv` (orden tile/subtile);
`rtl/epilogue_row.sv`.
**Hacer:** modo `OP.SWIGLU`. Los pesos llegan con tiles intercalados gate(j), up(j). Para cada panel:
- la salida requantizada INT16 del tile gate(j) se guarda en un buffer de 16 subtiles × 16 filas × 16 × 16 bits
  (M10K);
- con el tile up(j), `swiglu_lane` (4 carriles; tabla SiLU cargada por CSR) produce `h` en INT16;
- se escribe solo `h` (N = neurons).

**Aceptación:** `tb_swiglu_fused.sv` con datos reales de fc1 (bloque 0, `tools/dump_swiglu.py` ya existe) bit a bit
contra `swiglu_i`; los testbenches existentes en PASS; `quartus_map` con recursos. Commit `feat(accel): fused SwiGLU on fc1`.

### T43 · Software de T41/T42 (sonnet)
**Leer:** `model.c` 360–707 y lo añadido en T20; `fpga_pack.c` (`pack_weights`); interfaz de CSR nueva de T41/T42
(solo la cabecera de `csr.sv` con el mapa).
**Hacer:** detectar la `VERSION` que lo soporta.
- **q/k/v:** con `OUTFMT` 1/2/3 (+ `ROPE`), quitar los pasos `PS_READ` de q/k/v, `PS_ROPE` y el empaquetado de la
  atención.
- **`P·V`:** escribir ctx en formato 1 directo para `proj` (sin `PS_READ`/`PREP`).
- **fc1:** una sola operación SwiGLU (pesos intercalados empaquetados al arrancar) en lugar de fc1g + fc1x + `PS_SWIGLU`.
- Opciones `--no-outfmt` y `--no-fused-swiglu`.
- Extender `accel_sim` a los modos nuevos.

**Aceptación:** `make native arm test-sim test-proxy`, `--check` "identical", logits iguales con y sin las opciones.
Commit `perf(accel): use the layout writer and fused SwiGLU`.

### T44 · Compilación F4 (haiku)
Igual que T35, con la frecuencia de partida = la de T36, y el nombre `h7_<MHz>`.

### T45 · Placa y compuerta final (**tú**)
Igual que T36, con `gate.sh --trace f4.csv` y la prueba con el proxy. Etiqueta `accel-bit-final`.

---

## Qué esperar (est., del plan)

| Tras | s/imagen |
|---|---|
| hoy | 7.6 |
| F1 | 5.1–5.8 |
| F2 | 4.2–4.6 |
| F3 | 3.4–3.8 |
| F4 | 3.3–3.5 |

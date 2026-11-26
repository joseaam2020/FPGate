# Prompt para un modelo de mayor capacidad: plan de optimización del acelerador FPGA

> Cómo usarlo: copia desde "PROMPT" hasta el final en una conversación nueva. Si el modelo tiene acceso al
> repositorio, las rutas citadas existen (rama `JoseArroyo-Accelerator`, etiqueta `accel-pre-recompilacion`).
> Si no lo tiene, el prompt lleva los números necesarios.

---

## PROMPT

Eres un ingeniero senior de arquitectura de FPGA y de inferencia de redes neuronales cuantizadas. Necesito un
**plan de optimización sofisticado, priorizado y verificable** para un acelerador que ya funciona. Quiero que
cuestiones mis supuestos, que cuantifiques cada propuesta (ganancia en segundos por imagen, coste en recursos,
riesgo de timing, esfuerzo) y que señales qué falta medir antes de decidir. No escribas RTL todavía: entrega
el plan.

### 0. Objetivo principal

**Meta central: que cada inferencia (el score NSFW de una imagen) se calcule lo más rápido posible**, es
decir, minimizar la latencia por imagen de extremo a extremo (hoy 7.5–7.7 s; el proxy retiene la imagen hasta
tener el veredicto, así que esa latencia es lo que el usuario nota al navegar). El rendimiento se mide en
segundos por imagen con una imagen a la vez; el throughput con colas largas es secundario. Todo lo demás
(limpieza del diseño, recursos, timing) es un medio para bajar esa latencia, no un fin. Si una propuesta no
reduce la latencia por imagen, dilo y justifica por qué aun así merece la pena (por ejemplo, porque libera
recursos o margen de timing para otra mejora que sí la reduce). La exactitud bit a bit sigue siendo
innegociable: velocidad sin cambiar la decisión ni los logits.

### 1. Qué es el sistema

- **Tarea:** clasificar una imagen (448x448 RGB8) como NSFW/SFW con un ViT EVA02 podado ("r7"), cuantizado a
  INT8, dentro de un proxy HTTPS. El proxy retiene cada imagen hasta tener el veredicto y descarta el
  acelerador si tarda más de ~10 s por imagen (configurable, `--accel-timeout-ms`).
- **Placa:** Terasic DE10-Standard, Cyclone V SoC `5CSXFC6D6F31C6` (HPS: 2x ARM Cortex-A9, DDR3 compartida).
  Quartus Prime 18.1 Lite (sin licencias extra). La FPGA se configura desde U-Boot (SD, MSEL 01010).
- **Modelo r7:** 12 bloques, 8 cabezas, hd = 64, 1025 tokens (CLS + 32x32). 84 capas lineales en el
  acelerador (qkv/proj/fc1/fc2/SwiGLU, etc.), atención `S=Q·Kᵀ -> softmax entero -> P·V` por cabeza.
  Todo es **aritmética entera exacta**: el resultado de la FPGA es bit a bit el del modelo de referencia
  (verificado con 20 imágenes y capa por capa). **Esta exactitud es un requisito duro:** cualquier propuesta
  que la rompa debe decirlo y justificarse.
- **Reparto actual HPS/FPGA:**
  - FPGA (motor GEMM 16x16 + unidad de softmax): capas lineales, `S`, softmax (tablas EXP de 1025 entradas
    UQ0.16 y recíproco de 256), `P·V`, requantizado de salida.
  - ARM (C, 2 hilos OpenMP, NEON en GEMM y empaquetado): `patch_embed`, cuantizar entradas a INT8,
    LayerNorm, SwiGLU (la parte elemento a elemento; hay un `swiglu_lane.sv` en el RTL), RoPE, sumas
    residuales, empaquetar activaciones al formato del motor, leer resultados de la región compartida.
  - La región compartida con la FPGA es **sin caché desde el HPS** (medido: lectura 33–155 MB/s, escritura
    75–125 MB/s). Por eso el softmax tuvo que ir a la FPGA (`S` pesa ~34 MB por bloque).

### 2. Diseño de hardware actual (`accel/rtl`, ~4300 líneas SV)

- `gemm_array.sv` (arreglo 16x16 de PEs), `pe.sv`, `operand_feeder.sv`, `tile_buffers.sv` (RAM A de 256 KiB y
  RAM de pesos), `tile_sequencer.sv`, `gemm_engine.sv`, `epilogue_row.sv` / `requant_lane.sv`
  (requantizado), `linear_tile.sv`, `dma_rd.sv` / `dma_wr.sv` (DMA a DDR3 por `f2h_sdram0`, 256 bits),
  `csr.sv` / `csr_sm.sv` (registros), `attn_softmax.sv` + `softmax_*` (unidad de softmax con
  `SM_LANES = 4`, 16 RAM de fila para `P`), `fpgate_accel_core.sv` (parámetros `A_BEATS=8192`,
  `W_BEATS=1024`, `LANES=4`, `SM_LANES=4`). Hay también ficheros de prueba de hitos antiguos
  (`dsp_probe_top.sv`, `engine_probe_top.sv`, `lin_probe_top.sv`, `h0_dummy_engine.sv`,
  `fpgate_accel_top.sv`) que no deberían estar en el diseño final.
- Cuatro maestros Avalon-MM (motor lectura/escritura, softmax lectura/escritura) arbitrados por Platform
  Designer hacia un único puerto `f2h_sdram0` de 256 bits. Registros del acelerador en `0xFF200000` (puente
  ligero HPS->FPGA).
- Reloj del acelerador: **120 MHz** (PLL; el mismo PLL alimenta el reloj del puerto `f2h_sdram0`).

### 3. Estado de Quartus (compilación #4, la que corre en la placa)

| Recurso | Uso |
|---|---|
| ALMs | 27 340 / 41 910 (65 %) |
| Registros | 38 780 |
| Bloques M10K | 474 / 553 (86 %); 3.51 Mbit de 5.66 Mbit (62 %) |
| DSP | **112 / 112 (100 %)** |
| PLL | 1 / 15 |
| Pines | 338 / 499 |

Timing a 120 MHz (modelo lento 85 °C): setup +0.697 ns (reloj del PLL), hold **+0.012 ns**. En las dos
esquinas frías hay **un solo camino** con hold negativo (−0.108 / −0.011 ns: de un PE a un multiplicador de
otro, por un registro duplicado por la síntesis física); no se pudo eliminar con 3 compilaciones más (semillas,
síntesis física off). Receta del `.qsf`: `PHYSICAL_SYNTHESIS_*` en EXTRA, `OPTIMIZE_HOLD_TIMING "ALL PATHS"`,
`OPTIMIZE_MULTI_CORNER_TIMING ON`, sin `SEED`. **Cada compilación completa tarda 50–70 min.**

Intentos a 150 MHz (con la unidad de softmax): no cierra. Setup −1.6 ns, con caminos en el motor
(`a_waddr_p` -> RAM A) por congestión de M10K, y en el softmax (normalización -> 16 RAM de fila, FIFO del
interconnect -> comparaciones de máximo). Fmax estimada ≈ 121 MHz con el diseño actual. En una versión previa
(sin softmax) sí cerró a 150 MHz con +0.198 ns.

Uso de DSP: ningún multiplicador nuevo cabe en DSP; la lógica nueva usa `multstyle = "logic"` (ALMs).

### 4. Qué hay en Platform Designer (`accel/quartus_hps/soc_system.qsys`)

Heredado del GHRD de Terasic: `hps_0`, `led_pio`, `button_pio`, `dipsw_pio`, `ILC` (agrupa las IRQ de los
PIO y del jtag_uart), `jtag_uart`, `mm_bridge_0` (periféricos pequeños, movido a `0xFF240000`), `sysid_qsys`,
`hps_only_master`, `fpga_only_master`, `f2sdram_only_master` (maestros JTAG->Avalon de depuración; el último
cuelga de `f2h_sdram0_data`). Añadidos por nosotros: `fpgate_accel` (el core), `pll_0` (ya del GHRD, reconfigurado como reloj del
acelerador). En el top (`DE10_Standard_GHRD.v`) quedan también `debounce`, `edge_detect` y `altsource_probe`
para los resets del HPS. Mi hipótesis (no medida): quitar todo esto libera muy poco (decenas de ALMs, un par de
M10K). **Quiero que cuantifiques lo que realmente se gana**, qué es seguro quitar sin romper el arranque de
Linux/U-Boot (por ejemplo, `sysid_qsys` lo puede comprobar el software; los resets del HPS los usa el GHRD) y
si el verdadero beneficio sería de timing (menos congestión en `f2h_sdram0`) o de recursos, o ninguno.

### 5. Rendimiento medido en la placa (120 MHz, 1 imagen, ya con todo el software optimizado)

Total: **7.5–7.7 s por imagen** (era 63.2 s con solo capas lineales en FPGA; 14.2 s con atención en FPGA
antes de optimizar el ARM). Modos: `--no-overlap` 10.0 s, `--no-pipeline` 8.8 s, pipeline 7.7 s.

Desglose del pipeline (tramo entre atenciones, por imagen):

| Concepto | Tiempo |
|---|---|
| Tramo completo (pared) | 4.93 s |
| ARM ocupado | 4.47 s: LayerNorm 0.84, cuantizar 1.18 (sin NEON), empaquetar 0.65, leer 0.83, SwiGLU 0.45, RoPE 0.30, otros 0.22 |
| ARM esperando al motor | 0.46 s |
| Motor GEMM (suma) | 2.62 s (313 M ciclos / 120 MHz) |
| Atención | 2.22 s (FPGA 1.92 s: 230 M ciclos; 96 cabezas: 707 K ciclos `S`, 932 K softmax (~910 por fila), 759 K `P·V` por cabeza) + empaquetar Q/K/V 0.34 s |
| `patch_embed` (ARM) | 0.49 s |

Conclusión provisional: en el tramo lineal **el ARM es el cuello de botella** (ocupado 4.47 de 4.93 s), el
motor está ocioso casi la mitad del tiempo. La atención es secuencial por cabeza con el ARM esperando.

Eficiencia del motor (H4, 150 MHz): `b0.q_w` 209 MAC/ciclo (pico 256), `b0.fc2_w` 116 MAC/ciclo (probable
limitado por ancho de banda/latencia de la DDR3). Cycles RTL vs simulador: ±3 %.

### 6. Lista inicial de optimizaciones (mía; ampliar, criticar, descartar)

A. **Limpieza del diseño** (la que más me importa para empezar): sacar del `.qsys`/top lo no usado en
   ejecución (PIOs, ILC, jtag_uart, maestros JTAG, debounce, ficheros `*_probe_top`/`h0_dummy` del proyecto),
   sin romper el arranque. Ganancia esperada: recursos y quizá timing; **cuantificar**.
B. **Software ARM (sin RTL, exacto):** cuantizar con NEON + OpenMP (≈ −0.8 s), afinar LayerNorm/pack/read,
   lecturas más anchas de la región sin caché.
C. **Atención por trozos de filas de consulta** para solapar ARM y motor (≈ −1.5 a −2 s estimado).
D. **Mover a FPGA** LayerNorm, SwiGLU, RoPE y/o la cuantización (hasta ≈ −2 s), limitado por recursos
   (35 % ALMs, 14 % M10K, 0 DSP) y por el hold.
E. **Frecuencia:** volver a 150 MHz solo ahorraría ≈ 0.5 s del motor y no cierra; ¿qué cuesta cerrar a
   135–140 MHz (segmentar el camino del softmax, bajar la congestión de M10K)?
F. **Softmax:** 8 carriles en lugar de 4 (≈ 910 ciclos/fila hoy); ritmo de filas de `RAW32`; más subtiles por
   panel en `P·V`.
G. **Motor GEMM:** utilización 116–209 MAC/ciclo de 256; prefetch/doble buffer de pesos, mejor reparto entre
   los 4 maestros del puerto `f2h_sdram0`, ráfagas más largas.
H. **A nivel del proxy (no del acelerador):** tamaño mínimo para juzgar, caché de veredictos por hash.

### 7. Restricciones que el plan debe respetar

- Exactitud bit a bit (decisión y logits idénticos al modelo de referencia; hay tests por capa y de 20
  imágenes, simulación ModelSim ASE con unos 14 testbenches en `tb/`, tests por Docker/qemu del software ARM).
- El usuario ejecuta todo lo que toca la placa; un fallo puede colgarla (requiere reinicio). El plan debe
  ordenar el trabajo por etapas con compuertas y decir qué se puede verificar sin placa.
- Compilaciones de Quartus de ~1 h: minimizar su número; agrupar cambios de RTL.
- Hay un bitstream estable guardado (`accel/quartus_hps/h5_120MHz_*.sof`, `accel/rbf_out/h5_120MHz/`, etiqueta
  git `accel-pre-recompilacion`); cualquier plan debe poder volver a él.
- Presupuesto: el proxy descarta si tarda >10 s; hoy 7.7 s. **Objetivo: la latencia por imagen más baja
  posible**; meta intermedia de ≈ 5 s, sin considerarla techo.

### 8. Qué quiero que entregues

1. Una **tabla priorizada por reducción de latencia por imagen** de optimizaciones (incluidas las mías y las que yo no haya visto), con: ganancia
   estimada en s/imagen y cómo la calculas, recursos que consume/libera (ALM, M10K, DSP), riesgo de timing,
   esfuerzo, dependencias, y si se verifica en simulación, en Docker o solo en placa.
2. Un **orden de ejecución** por fases con criterios de paso/fallo y puntos de retorno (etiquetas git).
3. Un análisis específico de la **limpieza del .qsys y del top** (qué se puede quitar con seguridad y qué
   ganancia real tiene), distinguiendo lo que afecta a recursos, a timing o a nada.
4. Un **análisis del cuello de botella real**: con los números de arriba, ¿dónde está el límite teórico del
   sistema (ancho de banda DDR3/`f2h_sdram0`, ocupación del motor, ARM, latencia del softmax)? ¿Qué
   mediciones adicionales harías antes de gastar compilaciones de Quartus (por ejemplo, contadores de
   ocupación del motor, ancho de banda efectivo del puerto SDRAM)?
5. Las **preguntas abiertas** y los supuestos que no puedas comprobar con estos datos. No inventes cifras:
   marca lo que sea estimación.

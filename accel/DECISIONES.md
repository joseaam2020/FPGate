# Decisiones del acelerador (DES-2)

Fuente: `CONTEXTO.md` y `guia_DES-2_acelerador.md`. Este archivo se mantiene al día con cada decisión.

## Entorno

- Quartus Prime 18.1 Lite, ModelSim Intel FPGA (ASE) 10.5b en `C:\intelFPGA_lite\18.1`. No hay Verilator.
- Simulación: `accel\sim.ps1 [tb_name]` (sin argumentos corre todos los `tb/tb_*.sv`). Devuelve 0 solo si todos imprimen PASS.

## Fijadas (diseño y E1)

Un solo motor GEMM, arreglo 16×16 output-stationary, INT8×INT8 en DSP 9×9, reloj único (objetivo 150 MHz), CSR en `0xFF200000`, DMA Avalon-MM, probabilidad de atención UQ0.16, requantización con `M0` de 31 bits y desplazamiento por canal.

## Tomadas el 2026-11-08

| # | Decisión |
|---|---|
| C | Entradas de 16 bits en dos pasadas INT8×INT8 (byte alto, byte bajo) con acumulador desplazable |
| D | Epílogo bit-exacto con el modelo (`M0` completo) |
| E | Verificación bit-exacta etapa por etapa contra `e1_r7` |
| F | r7 es el objetivo; todo parametrizado |
| – | `ACC_W = 32` |

## Decisiones de implementación (H1)

- **Interfaz del arreglo sin sesgar.** `gemm_array` recibe un paso k por ciclo (`a_vec`, `w_vec`) y hace él mismo el escalonado: fila *i* retrasada *i* ciclos, columna *j* retrasada *j*. Las etiquetas `valid/first/shift` viajan con la activación.
- **Vaciado por multiplexor de filas**, una fila por ciclo, empezando `ROWS+COLS-1` ciclos después de `in_last`. El acumulador de cada PE es único (sin registro sombra), así que el productor no debe iniciar otro tile mientras `busy` esté en alto. Costo por tile ≈ `K + ROWS + COLS + ROWS`. Optimizar solapando tiles solo si Quartus deja margen.
- **Reset de las etiquetas `valid`** en PE y en el escalonado, para que ningún X llegue al acumulador.
- El acumulador envuelve módulo 2^32; el modelo de referencia nunca se acerca a ese rango, y el testbench compara módulo 2^32 en los casos extremos.

## Estado de hitos

| Hito | Estado |
|---|---|
| H0 CSR + VERSION + heartbeat + Platform Designer | **HECHO y verificado en la placa (2026-11-09)**: `csr_test` 24/24 PASS (VERSION, HEARTBEAT, registros, START/BUSY/DONE, errores, reinicio suave). El reloj de la fábrica en este sistema es de 50 MHz (CLOCK_50); para 150 MHz hace falta un PLL |
| H1 `pe` + arreglo 16×16 + vaciado | **simulación PASS** (`tb_gemm_array`, 34 tiles: INT8, INT16 con signo, UINT16, K = 1…2048, extremos). Quartus: cabe y cierra a 150 MHz, pero usa los 112 DSP (ver abajo) |
| H3 epílogo + modo 16 bits | **hecho en simulación**: `requant_lane`, `swiglu_lane`, `epilogue_row`, `linear_tile`, todos bit-exactos con capas reales. Conjunto medido en Quartus: cierra a 150 MHz. Falta: SwiGLU en varios carriles |
| H2 memorias, secuenciador y DMA | **COMPLETA** (2026-11-09): simulación en PASS (10 de 10 testbenches; 5 capas reales de r7 bit-exactas por el camino DMA) y Quartus cierra 150 MHz (ALMs 42 %, M10K 58 %, DSP 100 %, setup +0.21 ns) |
| H4 motor real en la placa | **casi completa (2026-11-10)**: 5 rebanadas y 2 capas completas de r7 bit-exactas por el DMA en la placa a 150 MHz; fallos inyectados 7/7; ciclos contra simulación a iguales dimensiones dentro de ±3 % (criterio ±5 % cumplido). Falta: `faults --bad-addr` y el costo de empaquetar en el HPS (ver "H4: resultados en la placa") |
| H5–H6 | pendientes |

## Primera compilación completa (`rtl/dsp_probe_top.sv`, 2026-11-08)

Top de prueba (generador pseudoaleatorio + arreglo 16×16 + 8 LEDs) sintetizado y ajustado con `quartus_map` y `quartus_fit`, reloj de 150 MHz.

| Recurso | Resultado | Disponible |
|---|---|---|
| Bloques DSP | **112** (los 256 multiplicadores en modo "Independent 9x9") | 112 (100 %) |
| ALMs | 11 241 (27 %) | 41 910 |
| Registros | 13 751 | – |
| M10K | 12 (el escalonado de activaciones se infiere como memoria) | 553 |
| Timing 150 MHz | **cierra**: holgura de setup +0.63 ns (modelo lento), hold +0.03 ns | – |

**Conclusión:** el arreglo 16×16 cabe y cierra a 150 MHz, pero el Fitter usa **todos** los DSP en vez de los ≈ 86 de la cuenta de la guía (el empaquetado es peor que 3 por bloque). Hasta el Fitter la cuenta de síntesis (256) era engañosa. Queda **cero margen** para el epílogo (decisión 10) y la atención. Opciones, a decidir después: (a) instanciar `altera_mult_add` en modo 9×9 explícito para forzar 3 por bloque; (b) arreglo más pequeño (p. ej. 16×12 o 14×14); (c) epílogo sin DSP (más lento, en ALMs). Probar (a) primero: es barato.

### Pruebas de margen de DSP (mismo top de prueba)

| Prueba | DSP en síntesis | DSP tras el Fitter | Timing 150 MHz |
|---|---|---|---|
| Arreglo 16×16 | 256 | 112 | cierra (+0.63 ns) |
| Arreglo 12×16 | 192 | 112 | cierra (+0.13 ns) |
| 16×16 + 4 productos 32×31 (carga tipo epílogo) | 268 | 112 | **no cierra (−0.50 ns)** |

- El Fitter solo empaqueta más de lo necesario cuando no cabe, así que "112/112" no mide el margen real. La prueba con la carga del epílogo demuestra que **sí cabe** (el Fitter aprieta hasta 3 por bloque), pero con el empaquetado forzado el reloj se degrada.
- `altmult_add` (3 multiplicadores): descartado; Quartus lo mapea en 18×18 con sumador (2 bloques y 178 celdas lógicas).
- Ruta crítica del caso −0.50 ns: `first_d` de un PE → acumulador de otro PE de la misma antidiagonal. Quartus fusionó los registros de control idénticos (misma demora de escalonado), así que una sola señal alimenta PEs lejanos. Es el problema de fanout de control que anticipa la guía (§8, punto 2). Arreglo propuesto: impedir la fusión (`(* preserve *)`) o registrar el control una vez más por grupo de PEs.
- **Arreglo aplicado (2026-11-08):** `(* dont_merge *)` en `valid_d`, `first_d` y `shift_d` de `pe.sv`. Resultado con el mismo top de prueba (16×16 + 4 productos 32×31): **cierra a 150 MHz**, setup +0.135 ns (modelo lento) / +0.205 ns, hold +0.057 ns; 11 319 ALMs, 14 504 registros, 112/112 DSP. `tb_gemm_array` sigue en PASS.
- Las 4 multiplicaciones 32×31 de la prueba tienen una sola etapa de registro; el epílogo real irá segmentado.

## Epílogo: un carril (`rtl/requant_lane.sv`, H3 parcial)

- Bit-exacto con `golden.requantize`: suma de sesgo, producto 32×31 (63 bits), redondeo mitad hacia +∞, desplazamiento aritmético 1…62, saturación simétrica a ±127 / ±32767. Segmentado en 6 etapas, un valor por ciclo.
- `tb_requant_lane`: PASS con 5 capas reales de la imagen 0 (b0.q_w, b0.v_w, b0.fc1g_w, b0.fc2_w, b1.fc2_w; 5 120 valores) y 20 000 valores sintéticos (todos los desplazamientos, saturación, empates de redondeo).
- Costo medido por carril ANTES de partir el producto (versión de 6 etapas; Quartus, proyecto aparte, 150 MHz): **3 bloques DSP**, 280 ALMs, 312 registros, Fmax 156.5 MHz (modelo lento, 85 °C). Con eso, **decisión 10**: n carriles = 3n DSP y un valor por ciclo cada uno; con 4 carriles (12 DSP) el vaciado de un tile de 16 filas × 16 columnas tarda ≈ 64 ciclos. Falta decidir cuántos carriles y si el vaciado se solapa con el siguiente tile.
- Las pruebas del arreglo con carga de epílogo (arriba) usaban 4 productos de una sola etapa; el carril real es más profundo y llegó a 156 MHz solo.

## Epílogo: SwiGLU (`rtl/swiglu_lane.sv`)

- Bit-exacto con `export_check.run`: `sg = silu[gi+127]` (tabla de 255 × INT16, una por bloque, cargada por un puerto de escritura), `h = sat16((sg·ui·M0 + 2^(s-1)) >> s)`. La requantización final reutiliza `requant_lane`. Latencia 3 + 6 ciclos, un valor por ciclo.
- `tb_swiglu_lane`: PASS con 2 048 valores reales (bloques 0 y 11 de la imagen 0, comparados con el `swiglu_i` guardado en el paquete) y 20 003 sintéticos (tabla aleatoria, desplazamientos 20…62, saturación).
- La tabla es memoria síncrona (se infiere como M10K). Un puerto de lectura por carril: una M10K de doble puerto atiende 2 carriles; 16 carriles necesitarían 8 copias de la tabla (≈ 4 KiB cada una).

## Capa lineal completa (`rtl/epilogue_row.sv`, `rtl/linear_tile.sv`)

- `epilogue_row`: `LANES` carriles `requant_lane` en paralelo; procesa una fila de 16 acumuladores en `16/LANES` pasos y entrega la fila ya requantizada. Acepta una fila cada `16/LANES` ciclos.
- `gemm_array` ahora tiene `drain_ready` (frena el vaciado: los acumuladores esperan quietos) y `drain_last`.
- `linear_tile` = arreglo + epílogo. `tb_linear_tile`: **PASS**, 20 tiles 16×16 de capas reales (b0.q_w, b0.v_w, b0.fc1g_w, b0.fc2_w, b1.fc2_w: entradas de 8 y 16 bits en dos pasadas, K = 768, 1232 y 2048, salidas de 8 y 16 bits), bit a bit contra la salida del modelo, con 4 y 16 carriles al mismo tiempo.
- Falta medir recursos y reloj del conjunto (arreglo + epílogo de 4 carriles) en Quartus: `linear_tile` tiene demasiados pines para sintetizarlo suelto, así que se medirá dentro de un top de prueba.

## Medición del conjunto arreglo 16×16 + epílogo de 4 carriles (`rtl/lin_probe_top.sv`)

Tres iteraciones con el mismo top de prueba (estímulo pseudoaleatorio, 8 salidas), 150 MHz:

| Iteración | Setup (modelo lento) | Causa de las rutas críticas |
|---|---|---|
| 1. epílogo inicial | −1.49 ns | selector de columnas (`beat`) alimentando suma, multiplicador y vaciado |
| 2. + registro tras el selector, `in_ready` registrado, contador de filas replicado | −0.55 ns | copia de 512 bits `drain_data → row_q`; multiplicador 32×31 sin segmentar |
| 3. + epílogo lee la fila en sitio (sin `row_q`), producto partido en 4 de 16×16 | **+0.592 ns** (modelo 0 °C: +0.904) | – |

Resultado final: **cierra a 150 MHz**; 12 847 ALMs (31 %), 18 051 registros, 12 M10K, DSP 112/112 tras el empaquetado del Fitter. Hold +0.092 ns. El holgura es pequeña: cualquier circuito nuevo (DMA, secuenciador, buffers) puede volver a apretarla.

Cambios de diseño que salieron de esto: `requant_lane` pasa a 7 etapas (`RQ_LATENCY` en `fpgate_pkg`); `epilogue_row` ya no copia la fila, la lee del arreglo mientras `in_ready` está bajo; `gemm_array` replica el contador de filas del vaciado.

## H0: registros de control (`rtl/csr.sv`, `rtl/fpgate_accel_top.sv`)

- `csr`: esclavo Avalon-MM de 32 bits, latencia de lectura 1, mapa de la guía §6.7 con `VERSION` = `0x46500001`, `CTRL` (START, RESET_SOFT, IRQ_EN, CLEAR_ERR), `STATUS` (BUSY, DONE, ERROR), `ERR_CODE`, parámetros de la operación, `CYCLES` de 64 bits, `BYTES_RD/WR`, `HEARTBEAT` y un `SCRATCH` de prueba. Respeta `byteenable`. START con BUSY alto da error 1.
- Desviación del mapa de la guía (a acordar con `fpgate-accel`, decisión 7): `STRIDE_OUT` va en `0x4C`, porque la guía pone dos registros en `0x34`; `SCRATCH` en `0x50`.
- `h0_dummy_engine`: motor de relleno que solo espera `DIM_M` ciclos (1000 si es 0) y marca DONE, o ERROR con código `0x10` si `OP[31]` está en 1. Sirve para probar el flujo desde Linux. Se reemplaza en el H2.
- `tb_csr`: PASS (36 comprobaciones: versión, registros de solo lectura, `byteenable`, heartbeat, START/BUSY/DONE con contador de ciclos, START con BUSY, error del motor con IRQ, CLEAR_ERR, RESET_SOFT).
- Falta de H0: el sistema en Platform Designer con el HPS (necesita el proyecto de referencia de la placa) y leer `VERSION` desde Linux.

## H0: sistema con el HPS (`quartus_hps/`)

- Base: el GHRD de Terasic (`DE10-Standard_v.1.4.0_SystemCD/Demonstration/SoC_FPGA/DE10_Standard_GHRD`), copiado sin los archivos generados. El original del CD no se toca. El GHRD se hizo con Quartus 16.1.2 Standard; Qsys 18.1 Lite lo abrió y generó sin errores. Ya trae el HPS con la DDR3 de la placa, los pines, el puente ligero y el puerto `f2h_sdram0` activo (sirve para el DMA del H2).
- `ip/fpgate_accel/fpgate_accel_hw.tcl`: componente de Platform Designer. Esclavo Avalon de 64 palabras de 32 bits, latencia de lectura 1. Usa los archivos de `accel/rtl` (`fpgate_accel_top`, `csr`, `h0_dummy_engine`).
- `add_accel.tcl`: script que agrega el componente al sistema (`qsys-script`). Los registros quedan en **0xFF200000** como pide el diseño. El puente `mm_bridge_0` del GHRD (LEDs, interruptores, sysid) tuvo que pasar a **0xFF240000** (su tamaño de 0x40000 bytes solo admite 0x0 o 0x40000).
- La interrupción del acelerador queda sin conectar: en el H0 el software consulta `STATUS`.
- Regenerar el sistema: `qsys-generate soc_system.qsys --synthesis=VERILOG --part=5CSXFC6D6F31C6 --search-path="ip/**/*,$"`. Compilar: `quartus_sh --flow compile DE10_Standard_GHRD`.

- **Compilación completa (2026-11-09):** `quartus_sh --flow compile DE10_Standard_GHRD` termina sin errores y genera `quartus_hps/output_files/DE10_Standard_GHRD.sof` (7.4 MB). 3 421 ALMs (8 %), 6 M10K, 0 DSP (todavía solo CSR y motor de relleno). Timing con todos los relojes en positivo (peor: hold +0.143 ns en `afi_clk_write_clk`, setup +1.73 ns).
- Para compilar hubo que quitar el atributo de solo lectura de las carpetas copiadas del CD (`ip/`, `soc_system/` y subcarpetas), que Quartus interpreta como falta de permiso de escritura (error 297009). `Get-ChildItem -Recurse -Directory | % { $_.Attributes = $_.Attributes -band -bnot [IO.FileAttributes]::ReadOnly }`.
- Falta: cargarlo en la placa y leer `VERSION` desde Linux (el HPS necesita los puentes habilitados).

- **Prueba en la placa (2026-11-09):** `.sof` programado por USB-Blaster con Linux ya arrancado; `tools/board/csr_test` pasa 24 de 24 comprobaciones. No hizo falta habilitar los puentes a mano. `HEARTBEAT` mide 50.0 MHz: el reloj de la fábrica es `CLOCK_50`, así que para el objetivo de 150 MHz hay que agregar un PLL (decisión 5) y repetir el cierre de tiempos con ese reloj. El acceso a registros funciona con el puente ligero en `0xFF200000`.

## H2: formatos en la DDR3, presupuesto de memoria y flujo (decididos 2026-11-09)

Puerto de datos: `f2h_sdram0`, Avalon-MM de 256 bits (32 bytes por beat). Direcciones **en bytes**, alineadas a 32. Todo lo de abajo está parametrizado.

**Activaciones A ("tile-major", las escribe el HPS ya agrupadas — cambio de contrato con `fpgate-accel`):**
- La matriz de M filas se parte en subtiles de 16 filas, contiguos desde `ADDR_A`. Un subtile contiene, para cada paso `k = 0..K-1`, una palabra de 16 bytes (fila `i` en el byte `i`). Entrada de 8 bits: K palabras. Entrada de 16 bits: dos planos seguidos (byte alto, luego byte bajo), K palabras cada uno. Tamaño de un subtile: `K·16` o `2·K·16` bytes.
- Las filas sobrantes del último subtile (M no múltiplo de 16) pueden contener cualquier cosa; no se escriben.
- K debe ser par (un beat = 2 palabras).

**Pesos W y constantes (los genera `tools/pack_layer.py` / `pack_weights`, a partir de `e1_r7`):**
- Un tile por cada 16 canales de salida, contiguos desde `ADDR_W`. Cada tile: 8 beats de constantes (16 entradas de 16 bytes: sesgo INT32 en bytes 0-3, `M0` en 4-7, `shift` en el byte 8, resto 0) seguidos de K palabras de 16 bytes (byte `j` = peso del canal `j`). Tamaño de tile: `256 + 16·K` bytes.
- `ADDR_BIAS` y `ADDR_SCALE` del CSR quedan reservados (las constantes viajan con el tile).

**Salida:** por cada fila de cada tile, un beat con 16 valores. Con salida de 16 bits son 32 bytes en `ADDR_OUT + fila·STRIDE_OUT + tile·32`; con 8 bits son 16 bytes en `... + tile·16` (con `byteenable`). Así el HPS la lee en el formato fila-mayor de siempre.

**Campos de `OP`:** `[0]` entrada de 16 bits, `[1]` entrada con signo (fc2 sí, P no), `[2]` salida de 16 bits, `[15:8]` subtiles por panel (lo calcula el HPS; el hardware da error si no caben en la RAM de A), `[31]` reservado para fallo del motor de relleno (H0).

**Presupuesto de memoria en chip (M10K = 691 KiB):** RAM de A 256 KiB (peor caso: 64 filas × K=2048 × 16 bits, o 128 filas × K=2048 × 8 bits) + dos buffers de W de 32 KiB (K=2048) + FIFO de salida (32 filas × 32 B = 1 KiB) ≈ 322 KiB (≈ 47 %). Decisión 9: panel de 4 subtiles (64 filas) en operaciones de 16 bits, 8 subtiles (128 filas) en las de 8 bits.

**Flujo del secuenciador:** para cada panel: cargar el panel de A; para cada tile de 16 canales: tener el tile de W cargado (se precarga el siguiente durante el cómputo), y para cada subtile del panel: alimentar el arreglo (K pasos, o 2K en 16 bits), recoger las 16 filas del epílogo en la FIFO y escribirlas con `dma_wr`. El arreglo no se reutiliza hasta que `busy` baja (contrato de `gemm_array`).

## H2: motor completo (resultados en simulación, 2026-11-09)

Módulos nuevos: `dma_rd`, `dma_wr`, `tile_buffers`, `operand_feeder`, `tile_sequencer`, `gemm_engine` (une todo con `linear_tile`) y `fpgate_accel_core` (CSR + motor, dos maestros Avalon). `fpgate_accel_top` del H0 no se tocó (sigue siendo el de la placa). Herramientas: `tools/pack_layer.py` (empaca una capa real al formato de la DDR3 y se autoverifica) y `tb/avalon_mem_model.sv` (memoria con esperas aleatorias). Desviación del plan: no hay `tb_sequencer` ni `tb_tile_buffers` aparte; el secuenciador y los buffers se prueban a través del motor (resultados y contadores exactos), que cubre lo mismo.

| Testbench | Qué prueba | Resultado |
|---|---|---|
| `tb_dma_rd` | ráfagas, cruce de 4 KiB, comandos vacíos, timeout y recuperación | 69 comandos PASS |
| `tb_dma_wr` | escrituras de 32 y 16 bytes con byte enables al azar | 400 beats PASS |
| `tb_gemm_engine` | formas aleatorias (M 1…70, N 16…64, K par 2…128, 8/16 bits con y sin signo, salida 8/16 bits, 1…8 subtiles por panel), filas de relleno intactas, `BYTES_RD/WR` exactos, parámetros inválidos (código 2), timeout de lectura (3), escritura bloqueada (4/6) y recuperación | 40 operaciones PASS |
| `tb_engine_real` | las 5 capas reales de r7 (`b0.q_w`, `b0.v_w`, `b0.fc1g_w`, `b0.fc2_w`, `b1.fc2_w`) por todo el camino DMA, con 2 subtiles/panel y con 1 | 10 corridas PASS, bit a bit contra `y.hex` |
| `tb_core` | lo mismo solo por registros (ADDR_*, DIM_*, OP, START, STATUS), contadores `BYTES_RD/WR/CYCLES`, error por K impar, siguiente operación | 25 comprobaciones PASS |

Más los 5 testbenches anteriores (`tb_gemm_array`, `tb_linear_tile`, `tb_requant_lane`, `tb_swiglu_lane`, `tb_csr`), todos en PASS tras los cambios.

**Sobrecosto medido (T13):** entre el arranque de un subtile y el siguiente dentro de un tile hay `planes·K + 110` ciclos (110 = arreglo ≈ 31 + vaciado de 16 filas con 4 carriles = 64 + latencia del epílogo y control). Pesa 14 % con K = 768, 4.5 % en `fc2` (K = 1232, dos pasadas) y 2.7 % con K = 2048. Para una imagen (≈ 22 500 corridas de subtile por bloque en los GEMM, 12 bloques) son ≈ 30 M ciclos ≈ 0.2 s a 150 MHz sobre los 2.29 s ideales, sin contar la atención. **Decisión 10 (provisional): 4 carriles** (12 DSP). Con 8 carriles el hueco bajaría a ≈ 78 ciclos (≈ 0.06 s menos) pero cuesta 12 DSP más, que no sobran con el Fitter al 100 %.

**Fallos encontrados y corregidos durante H2:** la precarga de pesos perdía el pulso `rd_done` si terminaba mientras se calculaba (el secuenciador esperaba para siempre; ahora se guarda el evento); la detección de errores del secuenciador estaba antes del `case` y podía perderse; anchos de `w_waddr`.

**Quartus, motor completo (`rtl/engine_probe_top.sv`: `fpgate_accel_core` con LANES = 4 y un respondedor sintético en lugar de la DDR3; 150 MHz; el motor solo se alcanza por los registros, así que los parámetros siguen siendo señales):**

| Recurso | Estimado (guía/plan) | Real | Disponible |
|---|---|---|---|
| ALMs | ≲ 70 % | **17 561 (42 %)** | 41 910 |
| Registros | – | 20 476 | – |
| M10K | ≈ 47–52 % (322 KiB + RAM) | **322 de 553 (58 %)**, 2.6 Mbit de datos | 553 |
| DSP | 112 tras empaquetar | **112 de 112 (100 %)** (268 antes del Fitter) | 112 |
| Setup a 150 MHz | ≥ 0 | **+0.212 ns** (modelo lento 85 °C); +0.217 ns (0 °C) | – |
| Hold | ≥ 0 | +0.053 ns (peor) | – |

El uso de M10K supera la cuenta por la fragmentación de las RAM de 256 bits (el estimado era 47 %). Queda 42 % de margen de M10K para los buffers de la atención (H5). El margen de tiempo es pequeño (0.21 ns): al integrar con el HPS (otro reloj, PLL, puertos de memoria) hay que repetir este cierre. El Fitter vuelve a usar los 112 DSP y no queda ninguno libre para la atención (decisión pendiente para H5).

**H2 queda completa** frente al criterio acordado: operación completa por registros, bit-exacta con datos aleatorios y reales (5 capas de r7), contadores y errores correctos, 10 de 10 testbenches en PASS, y Quartus cierra 150 MHz dentro de recursos. Pendiente del plan fuera de H2: conexión de los dos maestros al puerto `f2h_sdram0` y PLL (H4).

## H4: sistema con el motor real, PLL y DMA a la DDR3 (2026-11-09)

- **Sistema:** `quartus_hps/build_system.tcl` reconstruye `soc_system.qsys` desde el GHRD original (`ghrd_original.qsys`): PLL 50→150 MHz (`pll_0`), `fpgate_accel_core` (componente en `ip/fpgate_accel_core/`) con registros en 0xFF200000, el puente `mm_bridge_0` del GHRD en 0xFF240000, y los dos maestros (lectura y escritura) en `hps_0.f2h_sdram0_data`, cuyo reloj es ahora el del PLL (150 MHz). El `.sof` del H0 se conserva en `quartus_hps/h0_DE10_Standard_GHRD.sof`.
- **Recursos del sistema completo:** ALMs 21 399 (51 %), registros 30 683, M10K 386 de 553 (70 %), DSP 112 de 112 (100 %).
- **Cierre de tiempos a 150 MHz, tres iteraciones:** (1) −0.427 ns: escritura de las RAM grandes sin registros (dirección y dato del DMA y del interconnect hacia ~250 M10K) y longitud de ráfaga del DMA combinacional hacia el puerto del HPS; (2) con dos etapas de registro en la escritura de `tile_buffers` (`WR_PIPE`), el estado `S_PREP` en `dma_rd` y el retraso de tres ciclos del aviso `rd_done` hacia el secuenciador: −0.149 ns de setup y −0.001 ns de hold; (3) con la optimización física de Quartus (`PHYSICAL_SYNTHESIS_*` y `OPTIMIZE_HOLD_TIMING "ALL PATHS"` en el `.qsf` del GHRD): **setup +0.198 ns y hold +0.161 ns en el reloj del PLL (modelo lento 85 °C); todos los relojes y esquinas en positivo.** Margen pequeño: cualquier cambio de RTL obliga a repetir la compilación. El `.sof` final se guardó como `quartus_hps/h4_DE10_Standard_GHRD.sof`.
- **Región de DDR3:** no hay que reservarla: la imagen de Linux de Terasic ya limita su memoria a 768 MB (`/proc/iomem`: `00000000-2fffffff : System RAM`, U-Boot `DRAM: 1 GiB`), así que `0x30000000`…`0x3FFFFFFF` queda libre. `accel_test mem` pasó en la placa (escritura/lectura de 64 KiB por `/dev/mem`).
- **Herramientas H4:** `tools/dump_full_layer.py` (capas completas en binario; `b0.q_w` 1025×512×768 y `b0.fc2_w` 1025×768×1232; `tools/check_layer_bins.py` las recalcula desde los bytes: 0 diferencias), `tools/pack_layer.py` (también `.bin` para las rebanadas) y `tools/board/accel_test.c`.
- **Pendiente de H4:** ver la sección siguiente.

## H4: resultados en la placa (2026-11-10)

**Cómo hay que cargar la FPGA (lección importante).** Programar el `.sof` por JTAG con Linux ya arrancado sirve para los registros (puente ligero), pero **el DMA por `f2h_sdram0` congeló todo Linux** (probado varias veces; además una lectura simple del maestro JTAG hacia la DDR3 no terminó en 60 s). Lo que funciona es el flujo estándar de Terasic:
1. `SW10` (MSEL) en **`01010`** (FPPx32, "FPGA configurada desde el HPS"); el valor de fábrica `10010` (AS) no sirve. Con otra posición, `fpga load` de U-Boot falló con `altera_load: Failed with error code -8` (la FPGA se quedaba en fase de inicialización) y el gestor del kernel con `-22`.
2. `soc_system.rbf` en la partición FAT de la SD, junto a `u-boot.scr`, `zImage` y `socfpga.dtb`. Se genera con `quartus_cpf -c -o bitstream_compression=on h4_DE10_Standard_GHRD.sof soc_system.rbf` (sin `--hps`: "Device 5CSXFC6D6F31 does not support HPS IOCSR"). Copia en `accel/rbf_out/`.
3. U-Boot ejecuta `fatload`, `fpga load` y `bridge_enable_handoff` (suelta `fpgaportrst` = `0x1FF` y aplica la configuración). Comprobación: `cat /sys/class/fpga_manager/fpga0/state` = `operating`.
4. Con la FPGA sin configurar **no** se debe ejecutar `accel_test`: leer los registros por el puente cuelga el sistema.

**Preloader regenerado (2026-11-10).** El de la SD original tenía el controlador SDRAM configurado sin puertos de FPGA (`cportwidth` = `0x44FFF`, etc., `fpgaportrst` = 0). Se regeneró con el SoC EDS 18.1 desde nuestro `hps_isw_handoff` (`preloader_build/`, receta abajo) y se grabó en los primeros 256 KiB de `/dev/mmcblk0p3` (tipo `a2`); `u-boot.img` queda intacto desde 256 KiB. Respaldo del original: `accel/preloader_backup.bin` (MD5 `62a08546…`). **No se aisló si el preloader nuevo era imprescindible**: el cambio decisivo fue cargar la FPGA con el flujo estándar (MSEL + `.rbf`); ambas cosas están puestas ahora.
- Receta: `bsp-create-settings --type spl --bsp-dir spl_bsp --preloader-settings-dir hps_isw_handoff/soc_system_hps_0 --settings spl_bsp/settings.bsp --set spl.boot.WATCHDOG_ENABLE false`, y `make CROSS_COMPILE=arm-linux-gnueabihf-` dentro del *Embedded Command Shell* (este EDS no trae `arm-altera-eabi-`). Hizo falta: poner `ds-5/sw/gcc/bin` y `host_tools/altera/mkpimage` en `PATH`, reemplazar por copias los tres enlaces simbólicos de `uboot-socfpga` (`include/asm`, `arch/arm/include/asm/arch` y `proc`; el compilador es de Windows) y un enlace temporal `C:\cygdrive\c` → `C:\` (borrar al terminar).

**Resultados** (`accel_test`, FPGA a 150 MHz, bit-exactos con el modelo en todos los casos):

| Prueba | Ciclos | Tiempo | MAC/ciclo | Lectura / escritura |
|---|---|---|---|---|
| `b0_q_w` 32×32×768, 8 b → 16 b | 8 062 | 0.08 ms | 97.5 | 49 664 B / 2 048 B |
| `b0_v_w` 32×32×768, 8 b | 7 476 | 0.06 ms | 105.2 | 49 664 B / 1 024 B |
| `b0_fc1g_w` 32×32×768, 8 b | 7 405 | 0.05 ms | 106.2 | 49 664 B / 1 024 B |
| `b0_fc2_w` 32×32×1232, 16 b | 20 584 | 0.14 ms | 61.3 | 118 784 B / 2 048 B |
| `b1_fc2_w` 32×32×2048, 16 b | 33 859 | 0.23 ms | 61.9 | 197 120 B / 2 048 B |
| **`full_b0_q_w` 1025×512×768, 8 b → 16 b** | **1 929 640** | **12.87 ms** | **208.9** | 2.81 MB / 1.05 MB |
| **`full_b0_fc2_w` 1025×768×1232, 16 b** | **8 334 689** | **55.57 ms** | **116.4** | 18.86 MB / 1.57 MB |

- Fallos inyectados (`faults vectors/b0_q_w`): 7/7 PASS (START con BUSY → ERROR código 1; la operación en curso termina bien y su salida es correcta; RESET_SOFT a mitad de una operación deja el motor en reposo y la siguiente operación sale bit-exacta). **Falta** la prueba de dirección inválida (`--bad-addr`, se corre al final y con cuidado).
- **Eficiencia sobre el máximo** (256 MAC/ciclo en 8 bits, 128 en 16 bits por las dos pasadas): 82 % en `q_w`, 91 % en `fc2_w`.
- **Contraste con la simulación a iguales dimensiones (criterio ±5 %: CUMPLIDO).** `tb/tb_cycles.sv` + `tools/sim_cycles.ps1` simulan la operación completa por registros con los mismos M, N, K, bits y subtiles por panel que `accel_test`, con memoria `ddr` (10 % de esperas, 10 % de huecos, 30 ciclos de latencia). Los bytes leídos y escritos coinciden exactos con los de la placa. Los ciclos dependen linealmente del número de tiles de salida, así que las capas completas (demasiado largas para ModelSim ASE, ≈ 700 ciclos/s) se simularon con 1, 2 y 4 tiles y se extrapoló:

  | Caso | Placa | Simulación | Diferencia |
  |---|---|---|---|
  | `b0_v_w` / `b0_fc1g_w` 32×32×768, 8 b | 7 476 / 7 405 (típico 7 450) | 7 245 | +2.8 % |
  | `b0_q_w` 32×32×768 (repetida; la primera corrida tras arrancar dio 8 062 por acceso en frío) | 7 374–7 446 | 7 245 | +2.3 % |
  | `b0_fc2_w` 32×32×1232, 16 b | 20 584 | 20 124 | +2.3 % |
  | `b1_fc2_w` 32×32×2048, 16 b | 33 859 | 33 168 | +2.1 % |
  | `full_b0_q_w` 1025×512×768 (extrapolado: 142 883 + 31 × 57 477) | 1 929 640 | 1 924 659 | **+0.26 %** |
  | `full_b0_fc2_w` 1025×768×1232 (extrapolado: 455 947 + 47 × 167 378) | 8 334 689 | 8 322 713 | **+0.14 %** |

  Con memoria ideal (sin esperas, latencia 4) las rebanadas dan 5 211 (8 b), 14 763 y 24 227 (16 b): la latencia de la DDR3 pesa en las capas cortas y casi nada en las completas. La linealidad en N se comprobó con tres puntos en ambas capas (N = 16, 32, 64): pendiente por tile 57 483 y 57 473 en `q_w`; 167 378 y 167 356 en `fc2_w` (`fc2_w` con N = 64: 958 037 ciclos; la extrapolación con los tres puntos da 8 322 024, +0.15 % contra la placa).
- **Ancho de banda:** 218 MB/s de lectura en `q_w` y 339 MB/s en `fc2_w`, menos del 10 % de los 4.8 GB/s del puerto de 256 bits a 150 MHz: el motor está limitado por cálculo, no por la DDR3.
- La medición del reloj por sondeo en capas cortas (~98 a 147 MHz) es ruido del sondeo (la capa dura 0.05–0.2 ms); en las capas completas da 150 MHz.

## Datos de prueba reales (`tools/dump_vectors.py`)

`tools/dump_swiglu.py` solo lee las etapas ya guardadas en el paquete (`stages_img_00.npz`). Para las capas lineales, `tools/dump_vectors.py` no hay modelo nuevo: intercepta `golden.int_matmul_acc` y `golden.requantize` mientras `export_check.run` procesa la imagen 0, y escribe tiles de 32×32 en `accel/vectors/<capa>/*.hex`. Se ejecuta sin GPU:

```
docker run --rm -v "<repo>/model:/workspace" -v "<repo>/accel:/accel" -w /workspace model-torch:latest /accel/tools/dump_vectors.py --pkg quant/release/e1_r7 --out /accel/vectors
```

(`docker compose run` falla aquí porque pide GPU NVIDIA.) Los `.hex` se pueden regenerar; decidir al hacer commit si se versionan (≈ 2 MB), porque el paquete r7 no está en git.

## Pendientes (guía §4)

4 puerto de datos · 5 reloj final · 7 mapa de CSR · 8 acumulador en ALMs vs 2×18×19 · **9 `PANEL_M` en operaciones de 16 bits** · **10 DSP del epílogo**.

## Medición del HPS para repartir el trabajo (2026-11-10, `tools/board/hps_bench.c`)

Cortex-A9 a 925 MHz, región de DDR3 de la FPGA mapeada por `/dev/mem`.

| Medida | Resultado |
|---|---|
| Lectura de la región, 32 bits / NEON 4×16 B / `memcpy` a heap | 33 / 104 / 155 MB/s |
| Escritura a la región, 32 bits / NEON / `memcpy` desde heap | 75 / 125 / 111 MB/s |
| Referencia: `memcpy` heap → heap | 280 MB/s |
| GEMM INT8 con NEON (un núcleo, bloque 2×2) | 1 156 MMAC/s → las capas lineales de una imagen (≈ 54 GMAC) serían ≈ 47 s |
| `patch_embed` (0.46 GMAC) en el HPS | ≈ 0.40 s |
| LayerNorm float32 escalar (1025 × 768) | 34 ms → ≈ 1.2 s por imagen (36 LayerNorm) |
| Softmax entero escalar (1025 × 1025, una cabeza) | 55.7 ms, 49 ciclos por elemento → ≈ 5.4 s por imagen en un núcleo |

- El "mapeo con caché" dio exactamente lo mismo que el de `O_SYNC`: la región queda fuera de la RAM de Linux (`mem` truncada a 768 MB), así que `/dev/mem` la mapea siempre sin caché. No hay lecturas rápidas de la región desde el HPS.
- **El softmax no puede ir al HPS.** `S = Q·Kᵀ` pesa ≈ 34 MB por bloque en INT32 (8 cabezas): leerlo cuesta ≈ 0.22 s por bloque, la `P` de 16 bits de vuelta ≈ 0.15 s, y el cálculo ≈ 0.2 s por bloque con dos núcleos. Son ≈ 7 s por imagen solo de softmax, contra un presupuesto de 10 s del proxy. **El softmax y la cadena `S → P → P·V` deben quedar dentro de la FPGA (H5)**, sin que `S` ni `P` salgan al HPS.
- **SwiGLU sí puede quedarse en el HPS:** su salida (INT16) ya viaja al HPS para la LayerNorm interna del MLP, y la tabla SiLU cuesta ≈ 15 M búsquedas por imagen. Costo extra pequeño.
- El tráfico por la región sin caché (empaquetar entradas y leer salidas de las capas lineales, ≈ 15 MB por bloque) suma ≈ 1.3 s por imagen, y la LayerNorm ≈ 1.2 s. Esos dos son los costos del HPS que hay que optimizar (NEON, dos núcleos).

## fpgate-accel, fase 1: backend de CPU (2026-11-10, `accel/hps/`)

Programa en C (decisión del usuario: C, igual que el proxy y el mock) que sigue el contrato de `proxy/src/accel_proto.h` y el comportamiento de `mock_accel.c`. La pasada del modelo sigue `model/quant/export_check.py` (que reconstruye la inferencia solo con el paquete). El paquete se lee por un índice de texto (`tools/gen_pkg_index.py`), sin tocar `model/`.

**Verificación (nativo, x86-64, `accel/hps/README.md` explica cómo repetirla):**

| Comprobación | Resultado |
|---|---|
| 20 imágenes de referencia: decisión (umbral 0.5) | **20/20 idénticas** |
| Diferencia máxima de logits / de puntaje frente a la referencia | 0.075 (rango ≈ ±10) / 0.008 |
| Hash SHA-256 de `patch_embed` (etapa exacta: cuantización + GEMM INT8 + de-cuantización) | coincide |
| Normalización `uint8 RGB → float32 CHW` (lo que hace el proxy → `fpgate-accel`) | 0 de 602 112 valores difieren |
| Kernels enteros con las entradas de referencia, bloques 0 y 11: RoPE + softmax + `P·V`; GEMM de `proj` con epílogo; SwiGLU | **0 diferencias** en 524 800, 787 200 y 1 262 800 valores |
| Protocolo (`tests/test_protocol.py`): `STATUS`, `classify`, `request_id`, `latency_us`, `UNSUPPORTED` (tamaño, formato, canales, longitud, versión), sigue sirviendo tras los errores | PASS |

- Las etapas intermedias del bloque 0 difieren de 0.004 % (`v_w`) a 3 % (`fc2_w`) de los valores, casi todos en 1 LSB: es el efecto de LayerNorm y suma residual en flotante, que no pueden reproducir el patrón de bits de PyTorch (otro orden de suma, otra CPU/GPU). **Criterio adoptado:** enteros bit a bit con entradas de referencia; flotantes con tolerancia, decisión idéntica en las 20 imágenes.
- Tiempo en el PC (varios hilos con OpenMP): 2.6 a 10 s por imagen, 84 s las 20.
- Compila para ARM (estático, NEON) pero en la placa corre en un solo núcleo y con GEMM sin optimizar: es la referencia de corrección. Las decisiones de reparto (softmax dentro de la FPGA, SwiGLU en el HPS) están en "Medición del HPS" más arriba.

## fpgate-accel, fase 2: capas lineales en la FPGA (2026-11-10, `accel/hps/`)

`fpgate-accel --fpga` empaqueta los pesos de las 84 capas lineales en la región de DDR3 (53.3 MiB, 1.0 s al arrancar) y manda cada GEMM INT soportado (N multiple de 16, K par) al motor. `patch_embed` (necesita salida sin reescalar) y la cabeza (N = 4) siguen en CPU, igual que la atencion. Modulos: `src/fpga_pack.c` (formatos de DDR3), `src/accel_drv.c` (registros y region, misma secuencia que `accel_test`; se niega a tocar la FPGA si `fpga0/state` no es `operating`), integracion en `src/model.c`. `--check` calcula ademas cada capa en CPU y cuenta diferencias; `--blocks N` corre solo los primeros N bloques.

**Verificacion:**
- Empaquetadores en C contra los archivos de Python que ya valido la placa (`tests/test_pack.c`): pesos de `b0.q_w` y `b0.fc2_w` completos y activaciones de 4 rebanadas reales (8 y 16 bits): **0 bytes distintos**.
- En la placa, bloque 0 completo con `--check`: las 7 capas (`q`, `k`, `v`, `proj`, `fc1g`, `fc1x`, `fc2`, M = 1025) **identicas a la CPU**. Formas nuevas para el motor en la placa: K = 512 y N = 1232.
- En la placa, imagen 0 completa: logits **identicos al backend de CPU** (diferencia maxima con la referencia 0.02312, la misma), decision igual.

**Medidas en la placa (imagen 0, 63.2 s en total):**

| Etapa | Segundos |
|---|---|
| Atencion (CPU, un nucleo) | 51.70 |
| Cuantizar entradas | 2.40 |
| LayerNorm | 1.59 |
| `patch_embed` | 1.08 |
| SwiGLU | 1.11 |
| RoPE | 0.54 |
| Resto (residual, conversiones) | 0.66 |
| Camino de la FPGA: 84 operaciones, 313 242 678 ciclos | 2.09 en el motor + 0.99 empaquetar + 1.06 leer = 4.14 |

- La extrapolacion de ciclos hecha con un solo bloque (314 M) acerto: 313.2 M medidos.
- **Sin la atencion quedan 11.5 s** (4.14 de camino FPGA + 7.38 del ARM sin optimizar). Con la atencion en la FPGA (estimada ≈ 0.6 s) serian ≈ 12 s por imagen, sobre los 10 s por defecto del proxy (`--accel-timeout-ms`, configurable). Hacen falta dos cosas: la atencion en la FPGA (H5) y optimizar el ARM (dos nucleos, NEON): hoy todo corre en un nucleo y en C escalar.
- Cuantizar cuesta ≈ 59 ns por elemento (division en coma flotante y `rintf`): se puede acelerar sin perder exactitud calculando rapido y recalculando exacto solo cuando el valor cae cerca de un .5.

## H5: la atención dentro de la FPGA (en curso, 2026-11-10)

Plan aprobado (modelo híbrido, por etapas): `accel/..` ver el plan en la sesión; arquitectura de la etapa 1: `Q·Kᵀ` en el motor con salida INT32 cruda → unidad de softmax nueva (DDR3 → DDR3) → `P·V` en el motor. `S` y `P` no pasan por la CPU. Etapas 2 y 3 (encadenar `S → softmax` y `softmax → P·V` dentro del chip) quedan para después de medir recursos y tiempos.

**Hecho en simulación (todo bit-exacto con datos reales de r7, bloques 0 y 11):**
- Oráculo: `fpgate-accel --dump-attn` (en `accel/hps`) escribe, de las etapas de referencia, `Q8/K8/V8`, `S`, `P` y `ctx` por cabeza (`accel/vectors/attn_b0_h0`, `_b0_h5`, `_b11_h0`, `_b11_h5`); el contexto recalculado coincide con `b<N>.ctx_i` (0 de 65 600 distintos en las 4 cabezas). Datos reales: `S` queda entre −25 000 y 13 000 (teórico ±1 M), se mantiene INT32 por seguridad.
- **`RAW32`** (OP bit 3): el arreglo entrega los acumuladores sin epílogo; cada fila de un tile son 64 bytes (dos beats) en `ADDR_OUT + fila·stride + tile·64`. Cambios pequeños en `epilogue_row`, `linear_tile`, `tile_sequencer` (tile de salida de 64 bytes) y `gemm_engine` (segundo beat de la fila por un registro de retención). Una fila por 4 ciclos (mismo ritmo que el epílogo; se puede acelerar después). `tb_raw32`: **25 comprobaciones PASS** (`S` de dos cabezas reales, 3 formas cada una); `tb_linear_tile` y `tb_gemm_engine` siguen en PASS.
- **`attn_softmax`** (`rtl/attn_softmax.sv`, `softmax_idx_lane.sv`, `softmax_p3_lane.sv`, ROM generadas por `tools/gen_luts.py`): máximo mientras llega la fila (enmascarando las claves de relleno, columnas ≥ 1025), índice `((max − s)·M0 + 2^(sh−1)) >> sh` con multiplicador radix-4 en lógica (sin DSP; los 112 están usados), tabla EXP (ROM de M10K de dos puertos), suma, recíproco por tabla, normalización, 16 filas en RAM de filas y lectura transpuesta al formato de activaciones del motor (`[subtile][plano hi/lo][k][16 B]`, K = 1026). 4 carriles en el paso del exponente. `tb_attn_softmax`: **PASS**, `P` idéntica a la referencia en 32 filas (bloque 0) y 20 filas (bloque 11, grupo parcial: filas sin usar y relleno en cero); 24 493 ciclos para 32 filas (≈ 765 por fila con la escritura). Error encontrado y corregido en la simulación: el último dígito radix-4 leía un bit inexistente de `d0` (X) y sumaba `3·M0`.
- Integración en `fpgate_accel_core`: dirección de registros de 6 a 7 bits; **`0x000–0x0FF` motor (sin cambios), `0x100–0x1FF` unidad de softmax** (`rtl/csr_sm.sv`: VERSION `0x534D0001`, CTRL, STATUS, ERR_CODE, ROWS, COLS, KP, ADDR_S, STRIDE_S, ADDR_P, M0, SHIFT, CYCLES, BYTES_RD/WR); dos maestros Avalon nuevos (`rd2`, `wr2`) hacia `f2h_sdram0_data` (4 maestros en total); `ip/fpgate_accel_core/fpgate_accel_core_hw.tcl` y `build_system.tcl` actualizados. `tb_attn_core`: **cabeza completa por registros, contexto idéntico a la referencia** (bloque 0 cabeza 0 y bloque 11 cabeza 5, 32 filas × 64 valores): 21 992 ciclos las puntuaciones, 24 501 el softmax y 21 942 el contexto para 32 filas.
- Software (`accel/hps`): empaquetadores de Q (activaciones), Kᵀ y V (pesos con las constantes `av_requant`) **idénticos byte a byte** a los de Python de la simulación (`tests/test_pack.c`); `accel_drv` con `raw32`, entrada sin signo y el bloque de registros del softmax; `model.c` con `--fpga-attn` (por cabeza: empaquetar, `S`, softmax, `P·V`; `--check` compara `S` y `P` de 17 filas de la cabeza 0 y el contexto del bloque entero contra la CPU).
- Estimación con los ciclos medidos en simulación (32 filas): por cabeza de 1025 filas ≈ 2.1 M ciclos (14 ms), por imagen (12 bloques × 8 cabezas) ≈ 1.35 s a 150 MHz, sin solapar. El softmax solo ≈ 0.5 s; mejoras ya identificadas: ritmo de filas en `RAW32` (hoy 1 fila cada 4 ciclos), 8 carriles, solape entre filas y más subtiles por panel en `P·V`.

**Pendiente:** compilación de Quartus (recursos y tiempos; reloj gradual 145/140/… si no cierra a 150 MHz), prueba en la placa por etapas, medición de tiempos y repartición final.

**H5, Quartus (2026-11-10), cuatro compilaciones del sistema con la unidad de softmax (cuatro maestros a `f2h_sdram0`):**

| # | Reloj | Resultado |
|---|---|---|
| 1 | 150 MHz | No cabía: las 16 RAM de filas de `P` se infirieron como registros (308 mil registros, 133 144 nodos combinacionales contra 83 820). Arreglo: `softmax_rowram.sv` (módulo propio con `ramstyle = "M10K, no_rw_check"`) y `multstyle = "logic"` en los carriles (0 DSP nuevos) |
| 2 | 150 MHz | Cabe (ALMs 66 %, M10K 85 %, DSP 112); setup −1.481 ns, hold −0.430 ns. Caminos en la lógica nueva: carril de normalización → 16 RAM de filas, y FIFO del interconnect → comparaciones del máximo |
| 3 | 150 MHz | Con segmentación en todas las fronteras de la unidad: hold +0.108 ns, setup −1.607 ns (ahora también caminos del motor, `a_waddr_p` → RAM A, por congestión: M10K al 86 %) |
| 4 | **120 MHz** | **Setup cierra en todas las esquinas (+0.697 ns a 85 °C, +0.770 a 0 °C)**; hold +0.012 ns a 85 °C y −0.108 / −0.011 ns en las dos esquinas frías por **un solo camino** (de un PE a un multiplicador de otro, por un registro duplicado). ALMs 27 340 (65 %), registros 38 780, bloques M10K 474 de 553 (86 %), 3.51 Mbit (62 %), DSP 112 de 112 |

`build_system.tcl` tiene la variable `ACCEL_MHZ` (el PLL y el reloj del puerto `f2h_sdram0` cambian juntos). La #4 está guardada como `quartus_hps/h5_120MHz_DE10_Standard_GHRD.sof` y `rbf_out/h5_120MHz/soc_system.rbf`. Estimación del peor camino a 150 MHz: necesitaba 6.667 + 1.607 = 8.27 ns ≈ 121 MHz máximo. A 120 MHz las capas lineales pasan de 2.09 s a 2.6 s por imagen (cálculo: 313 M ciclos / 120 MHz). Compilación #5: misma configuración con `SEED 2` para limpiar el hold.

**H5 en la placa (2026-11-11), bitstream de 120 MHz (compilación #4):**
- Carga: `fpga0/state` = `operating`, `fpgaportrst` = `0x1FF`, `devmem 0xFF200100` = `0x534D0001` (unidad de softmax presente), `accel_test regs` 9/9 con el reloj medido en 120.0 MHz, capas del motor siguen bit-exactas (`full_b0_q_w`: 1 906 931 ciclos = 15.9 ms; menos ciclos que a 150 MHz porque la latencia de la DDR3 pesa menos en ciclos).
- **Imagen 0 completa con las capas lineales y la atención en la FPGA: logits idénticos al backend de CPU (diferencia máxima con la referencia 0.02312, la misma), decisión igual.**

| Etapa (imagen completa) | CPU (antes) | FPGA lineal | FPGA lineal + atención |
|---|---|---|---|
| Atención | 51.70 s | 51.70 s | **2.30 s** (FPGA 1.92 s a 120 MHz + empaquetar Q/K/V 0.34 s + leer 0.04 s) |
| Capas lineales: motor | 47 s en ARM | 2.09 s (150 MHz) | 2.59 s (120 MHz) |
| Capas lineales: empaquetar y leer (HPS) | – | 2.05 s | 2.03 s |
| Cuantizar entradas | 2.40 s | 2.40 s | 2.40 s |
| LayerNorm | 1.59 s | 1.59 s | 1.59 s |
| `patch_embed` | 1.08 s | 1.08 s | 1.02 s |
| SwiGLU | 1.11 s | 1.11 s | 1.05 s |
| RoPE | 0.54 s | 0.54 s | 0.54 s |
| Resto | 0.66 s | 0.66 s | 0.66 s |
| **Total por imagen** | – | **63.2 s** | **14.2 s** |

Atención en la FPGA, por 96 cabezas (12 bloques × 8): 67.9 M ciclos las puntuaciones, 89.5 M el softmax y 72.9 M el contexto = 230 M ciclos = 1.92 s. Por cabeza: 707 K, 932 K y 759 K ciclos; el softmax (≈ 910 ciclos por fila) es el paso más lento y supera mi estimación (≈ 780).

**Pendiente para bajar de los 10 s del proxy:** el ARM sigue en ≈ 9.3 s (cuantizar 2.40, empaquetar/leer el motor 2.03, LayerNorm 1.59, SwiGLU 1.05, `patch_embed` 1.02, RoPE 0.54, resto 0.66; un solo núcleo y C escalar), más 4.9 s de FPGA. Mejoras del lado FPGA ya identificadas: ritmo de filas de `RAW32`, 8 carriles en el softmax, más subtiles por panel en `P·V`, y volver a más de 120 MHz si un cierre de tiempos lo permite.

**H5, Quartus: compilaciones 5 a 7 y decisión (2026-11-11).** Las tres a 120 MHz, mismos recursos (ALMs 65 %, M10K 86 %, DSP 112). El hold no depende de la frecuencia (es un camino del mismo reloj), así que bajar más los MHz no lo arregla.

| # | Cambio respecto a la #4 | Setup mínimo | Hold mínimo |
|---|---|---|---|
| 4 | (referencia) síntesis física ON/EXTRA, semilla por defecto | +0.697 ns | −0.108 ns (frío lento), −0.011 ns (frío rápido); +0.012 ns a 85 °C |
| 5 | `SEED 2` | +0.079 ns | −0.026 ns |
| 6 | `SEED 3` | +0.688 ns | −0.082 ns |
| 7 | síntesis física OFF, `FITTER_EFFORT "STANDARD FIT"`, `SEED 1` | +0.142 ns | **−0.123 a −0.363 ns en las cuatro esquinas** (peor) |

- El único camino con hold negativo en la #4 es de un PE a un multiplicador de otro PE del arreglo (por un registro que la síntesis física duplicó). Apagar la síntesis física (#7) empeoró el hold en todas las esquinas: esa opción estaba corrigiéndolo.
- **Decisión: el bitstream de referencia es el de la #4** (`quartus_hps/h5_120MHz_DE10_Standard_GHRD.sof`, `rbf_out/h5_120MHz/soc_system.rbf`), probado en la placa. Setup holgado; una sola violación de hold de ≤ 0.11 ns en una esquina fría.
- **Receta exacta de la #4:** RTL de `accel/rtl` sin cambios desde el 10-nov 20:45; `build_system.tcl` con `ACCEL_MHZ 120` (`qsys-script`, `qsys-generate`); `DE10_Standard_GHRD.qsf` con `PHYSICAL_SYNTHESIS_COMBO_LOGIC ON`, `..._REGISTER_DUPLICATION ON`, `..._REGISTER_RETIMING ON`, `..._EFFORT EXTRA`, `OPTIMIZE_HOLD_TIMING "ALL PATHS"`, `OPTIMIZE_MULTI_CORNER_TIMING ON` y **sin** líneas `SEED` ni `FITTER_EFFORT`; `quartus_sh --flow compile DE10_Standard_GHRD`. El `.qsf` del repositorio es ese. No se ha verificado que recompilar dé el mismo `.sof` bit a bit (Quartus suele ser determinista con las mismas entradas); el `.sof` y el `.rbf` de la #4 están guardados para no depender de eso.

## Optimización del ARM, parte 1 (2026-11-11, `accel/hps`)

Objetivo: bajar de los 10 s del proxy el tiempo de la imagen (14.2 s medidos, ≈ 9.3 s son del ARM). Todo lo de esta parte es **exacto**: no cambia ningún resultado respecto a la versión anterior.

- **Dos núcleos:** el build ARM no pasaba `-fopenmp`, así que las pragmas se ignoraban. Ahora el `Makefile` lo incluye y se paralelizan GEMM, LayerNorm, RoPE, SwiGLU, sumas residuales, `patch_embed` (unfold y des-cuantización), empaquetado y conversión de lectura.
- **Cuantización rápida:** `quantize_f32` usa `x·(1/sc)` con redondeo por número mágico y recalcula con la división exacta solo cerca de los empates. 226 M de valores contra `rintf(x/sc)`: 0 diferencias.
- **Sin ida y vuelta de `ctx`:** si `s_ctx` es igual a la escala de entrada de `proj`, se usa el INT8 directamente (sin pasar a float y cuantizar de nuevo); si no, el camino largo.
- **NEON:** GEMM INT8 (1 fila × 4 columnas, lo usa `patch_embed`) y transposición 16×16 en el empaquetado de activaciones de 8 y 16 bits.
- **LayerNorm:** 4 acumuladores `double` independientes en las sumas (rompen la cadena de latencia). Salida bit a bit igual a la anterior (0 de 1.5 M).
- **Verificación sin placa:** el contenedor `fpgate-proxy-dev` trae `qemu-arm`, así que el binario ARM real (NEON, OpenMP estático) corre contra el oráculo: bloques 0 y 11 bit-exactos, logit 0.02312, 20/20 decisiones (peor diferencia 0.07451, igual que antes). `make test-kernels` (en `accel/hps`) ejecuta cuatro pruebas de kernel (`test_quant`, `test_gemm`, `test_ln`, `test_pack_rand`) en nativo y en ARM bajo qemu; todas dan 0 diferencias. qemu no sirve para medir tiempos.
- **Sin medir en la placa todavía.** Referencia a comparar (14.2 s): cuantizar 2.40, empaquetar/leer 2.03, LayerNorm 1.59, SwiGLU 1.05, `patch_embed` 1.02, RoPE 0.54, resto 0.66.

### Punto de retorno antes del solapamiento ARM/FPGA

El siguiente paso (solapar el trabajo del ARM con el de la FPGA, con hilos y sincronización) es el más delicado. **La etiqueta git `accel-pre-solape` marca el último commit sin solapamiento.** Si el solapamiento falla en la placa (cuelgues, resultados distintos o más lento), se vuelve con `git checkout accel-pre-solape -- accel/hps` (o `git reset --hard accel-pre-solape` si se quiere descartar todo lo posterior; no hay cambios de FPGA en juego, el bitstream es el mismo). Regla del solapamiento: debe poder desactivarse con una opción de línea de comandos (`--no-overlap`) y el resultado con y sin él debe ser idéntico bit a bit.

## Solapamiento ARM/FPGA, paso 1: capas lineales por trozos (2026-11-11, `accel/hps`)

Empieza después de la etiqueta `accel-pre-solape`. Se vuelve a la versión sin solapamiento con `git checkout accel-pre-solape -- accel/hps`, o sin tocar git con `--no-overlap` (usa el mismo código secuencial de antes, `linear_fpga_seq`).

**Qué hace.** Cada capa lineal se parte en trozos de 256 filas (un panel completo del motor: 16 subtiles de 16 filas con entrada de 8 bits). Con el motor calculando el trozo *c*, el ARM empaqueta la entrada del trozo *c+1* y lee la salida del trozo *c−1*. Los trozos usan zonas distintas de los buffers de entrada y salida, así que no hay conflicto; solo hay una operación del motor a la vez. Los pesos se leen una vez por trozo, igual que antes por panel, así que los ciclos del motor casi no cambian (más un pequeño costo fijo por operación).

**Cambios.** `accel_drv`: `accel_hw_run` se divide en `accel_hw_start` y `accel_hw_wait` (la secuencia de registros es la misma de antes). `model.c`: `linear_fpga_chunked`, `pack_chunk`, `read_chunk`; la ruta secuencial queda intacta. Opciones: `--no-overlap`, `--chunk N` (múltiplo de 16; por defecto 256).

**Prueba sin placa: motor simulado (`--sim`, `src/accel_sim.c`).** Decodifica los mismos formatos empaquetados, calcula con los kernels de CPU y escribe la salida por rebanadas desde un hilo aparte, con retardo, sobre memoria normal. Sirve para comprobar la lógica de orden, no los tiempos (el hilo del "motor" compite por CPU con el ARM). Resultado: con trozos de 256, 64, 1000 y sin solapamiento, **todas las capas idénticas al backend de CPU** (`make test-sim`), también con el binario ARM bajo qemu. Control negativo: con un fallo introducido a propósito (leer un trozo mientras el motor lo escribe) el simulador marca ≈ 524 000 de 524 800 salidas distintas, es decir, detecta errores de orden. No cubre la atención ni el softmax (no se simulan): esa parte no cambió.

**Lo que no se sabe hasta probar en la placa:** el tiempo real. Posible freno: el ARM y la FPGA comparten la DDR3 y la región es no cacheable, así que empaquetar y leer a la vez que el motor lee y escribe puede hacer lento a cualquiera de los dos. Hay que mirar `FPGA: ... layers wall` frente a la suma de pack + wait + read.

**Siguiente idea (no hecha):** casi todo entre dos atenciones es por filas (LayerNorm, cuantizar, GEMM, RoPE, SwiGLU, sumas), así que se podría hacer una tubería por trozos de filas entre capas para que el ARM y la FPGA trabajen a la vez también en esas etapas. Es una reestructuración mayor; se decide con los tiempos medidos.

## Solapamiento ARM/FPGA, paso 2: tubería por trozos de filas entre atenciones (2026-11-11, `accel/hps`)

Empieza después de la etiqueta `accel-pre-tuberia` (último commit con el solapamiento solo dentro de cada capa). Volver: `git checkout accel-pre-tuberia -- accel/hps`, o `--no-pipeline` en la línea de comandos (usa el código del paso 1), o `--no-overlap` (el secuencial original).

**Idea.** Entre la atención de un bloque y la del siguiente todo es por filas: proj, suma residual, LayerNorm 2, fc1 (gate y up), SwiGLU, LayerNorm 3, fc2, suma residual, y luego LayerNorm 1, q, k, v y RoPE del bloque siguiente. Solo la atención necesita todas las filas. Las 1025 filas se parten en trozos de 256 (5 trozos) y cada trozo recorre la misma lista de pasos. Un paso es trabajo del ARM o una operación del motor. Mientras el motor trabaja con un trozo, el ARM ejecuta pasos listos de otros trozos. Antes, la FPGA esperaba en LayerNorm, cuantizar, SwiGLU y RoPE, y el ARM esperaba en cada capa: los dos tiempos se sumaban. En teoría el total pasa a ser el mayor de los dos.

**Diseño.**
- Planificador de un solo hilo (`pl_run` en `model.c`): una operación del motor a la vez; tras cada paso del ARM consulta sin bloquear el estado del motor (`accel_hw_busy`), recoge la operación terminada y lanza la siguiente de cualquier trozo que la espere (el de menor índice). Sin hilos nuevos ni cerrojos.
- Cada trozo tiene su hueco propio en las áreas de activaciones y de salidas de la región (1 MiB por trozo con 256 filas; hasta 8 trozos) y sus propios buffers de trabajo (salida de LayerNorm, entrada cuantizada, empaquetado). Los trozos solo tocan filas distintas de las matrices compartidas (`x`, `q16`, `g16`…), así que no hay conflictos de datos.
- Cada paso hace exactamente lo mismo que el código secuencial sobre sus filas; RoPE recibe la posición absoluta del trozo (`rope_int_range`). Si algo no encaja (una capa fuera de la FPGA, más de 8 trozos, buffers que no caben) el modelo vuelve solo al camino anterior.
- La región de la atención no se toca: la atención sigue siendo la barrera entre tramos.

**Opciones.** `--no-pipeline`, `--no-overlap`, `--chunk N`, `--logits` (imprime los logits con 9 cifras para comparar dos ejecuciones bit a bit), `--sim` (motor simulado). Con `--stages` la imagen 0 usa el camino sin tubería (la tabla de etapas necesita las matrices completas).

**Pruebas sin placa (`make test-sim`, motor simulado por hilo con retardo, y el binario ARM bajo qemu).**
- Logits idénticos bit a bit, en 2 imágenes, con: sin solapamiento, solape por capa, tubería con trozos de 256, 160, 512 y 48 (este último cae al solape por capa por falta de huecos). Cada trozo de cada capa es igual al resultado de la CPU (`--check`).
- 20 imágenes: los logits de la tubería con el motor simulado son idénticos bit a bit a los del backend de CPU; 20/20 decisiones.
- Control negativo: si todos los trozos escriben su salida en el mismo hueco (un error de reparto de memoria), `--check` marca ≈ 131 000 de 131 072 salidas distintas por capa.
- Los tiempos del simulador no significan nada: el "motor" gasta CPU del PC y el ARM espera por él.

**Lo que se sabe y lo que no.** Sabemos que la lógica de orden y de memoria es correcta. No sabemos el tiempo real: puede mejorar mucho (si ARM y FPGA se reparten bien) o poco (si compiten por la DDR3, que la región no cacheable agrava). El resumen de la placa trae una línea nueva `pipeline between attentions: wall X s; ARM busy: …; ARM waiting for the engine Y s`: si `ARM waiting` es grande, el motor es el cuello de botella (bien: queda lo mínimo); si es pequeño y `wall` es mayor que los ciclos del motor (`FPGA: … cycles (… s)`), el ARM es el cuello de botella.

# Plan de optimización de la latencia por imagen (acelerador r7, DE10-Standard)

Fecha: 2026-11-12. Punto de partida: bitstream #4 (120 MHz, `rbf_out/h5_120MHz/`), software en la etiqueta
`accel-7s-en-placa`, **7.5–7.7 s por imagen**, exacto bit a bit.

Convenciones: **(med.)** = medido en la placa o en un informe de Quartus; **(est.)** = estimación mía, con el
cálculo al lado; **(?)** = supuesto que no puedo comprobar con lo que hay en el repositorio. Las ganancias **no
se suman**: cada una depende del régimen (ARM o motor como cuello de botella) en el que se aplique. Por eso la
§4 da la proyección por fases.

---

## 0. Resumen ejecutivo

1. **Hoy la latencia es una suma de tres tramos en serie, no un máximo:** `patch_embed` 0.49 (ARM) +
   tubería entre atenciones 4.93 (limitada por el ARM) + atención 2.22 (FPGA en serie, ARM parado) = 7.64 s (med.).
2. **La mayor ganancia inmediata no necesita RTL ni Quartus.** La unidad de softmax tiene CSR y maestros propios,
   pero el software nunca la hace trabajar a la vez que el motor (`attention_fpga`, [model.c:920-966](hps/src/model.c#L920-L966):
   empaquetar → `S` → softmax → `P·V`, con un único juego de buffers). Si se solapan, la atención baja de 2.22 a
   ≈ 1.3 s (est.).
3. **`patch_embed` puede ir al motor sin RTL nuevo:** es un GEMM INT8 con acumulación entera y descuantización en
   float después ([model.c:730-738](hps/src/model.c#L730-L738)). En modo `RAW32` el motor da los mismos acumuladores:
   0.49 → ≈ 0.08 s (est.).
4. Con software solamente (fases 1 y 2) la proyección es **≈ 4.2–4.6 s** (est.), y el sistema pasa a estar limitado
   por el **motor**. A partir de ahí suben de valor la frecuencia y la eficiencia del motor, y con ellas la limpieza
   del `.qsys` como medio para cerrar tiempos. Con una tanda de RTL bien elegida, ≈ 3.3–3.6 s (est.).
5. Corrijo varios supuestos de la lista inicial (detalle en §1 y §3):
   - la limpieza libera **≈ 2.5 k ALMs**, no decenas;
   - el recurso que de verdad se puede recuperar es la **FIFO de 52 M10K** del interconnect de `f2h_sdram0`;
   - `fc2` **no** está limitado por la DDR3 (91 % del pico de 16 bits);
   - 8 carriles de softmax dan ≈ 0 s una vez solapado;
   - LayerNorm y cuantizar son float IEEE: llevarlos a la FPGA con exactitud bit a bit no compensa.

---

## 1. Análisis del cuello de botella real

### 1.1 Trabajo por imagen de cada recurso (lo que habría que hacer aunque todo se solapara)

| Recurso | Trabajo por imagen hoy | Cómo sale | Cota inferior teórica |
|---|---|---|---|
| ARM (2 núcleos) | **≈ 5.3 s** ocupado | tubería 4.47 + `patch_embed` 0.49 + empaquetar Q/K/V ≈ 0.3 + leer ctx 0.04 (med.) | depende de qué se le quite (§2) |
| Motor GEMM | **≈ 3.78 s** a 120 MHz | lineales 313 M ciclos + `S` 67.9 M + `P·V` 72.9 M = 453.8 M ciclos (med.) | lineales: 3.55 GMAC/256 + 0.97 GMAC(fc2, 16 b)/128 = 21.5 M ciclos/bloque × 12 = 258 M; atención 96 × (263 K + 526 K) = 75.7 M → **334 M ciclos = 2.78 s a 120 MHz, 2.39 s a 140, 2.23 s a 150** (est.) |
| Unidad de softmax | 0.75 s | 89.5 M ciclos (med.), ≈ 910 ciclos/fila | con 4 carriles y solape entre filas ≈ 260 ciclos/fila → 0.21 s (est.) |
| DDR3 (400 MHz × 32 bits = 3.2 GB/s teóricos, (med.) en `soc_system.qsys`) | ≈ 1.7 GB/imagen (est.) | lineales ≈ 40 MB/bloque (`fc2` relee los pesos en cada panel: ≈ 17 MB/bloque) + atención ≈ 13 MB/cabeza (`S` 4.2 MB escrito y leído, `P` 2.1 MB escrito y leído, Q/K/V) | ≥ 0.55 s de transferencia pura; **no es el límite**, pero en la atención solapada el pico sería ≈ 1.1 GB/s (est.), un tercio de la DDR3 compartida con el ARM: **hay que medirlo** (M2) |

**Conclusión.** Con el reparto actual y solape perfecto el límite sería max(ARM 5.3, motor 3.78) ≈ 5.3 s, es decir,
el ARM. Hoy se pierden ≈ 2.4 s porque los tramos van en serie: la atención (2.22 s) no se solapa con nada y el
`patch_embed` (0.49 s) tampoco. El orden lógico es:

1. Quitar la serialización (software): fases 1 y 2.
2. Bajar el trabajo del ARM por debajo del motor (software y, si hace falta, RTL de epílogo).
3. Solo entonces atacar el motor (frecuencia y eficiencia), que es cuando una compilación de 1 h compensa.

### 1.2 Eficiencia del motor (con los números del propio `DECISIONES.md`)

- `full_b0_q_w` (8 bits): 208.9 MAC/ciclo = **82 %** del pico de 256.
- `full_b0_fc2_w` (16 bits, dos pasadas): 116.4 MAC/ciclo = **91 % del pico de 128**. El ancho de banda medido es
  339 MB/s, menos del 10 % del puerto. **No está limitado por la DDR3**: el supuesto de la §5 del prompt es
  incorrecto. El techo es la doble pasada de 16 bits, que no se puede evitar: hacer 16×8 en una pasada necesita
  multiplicadores de 18×18, y 112 DSP dan como mucho 224, menos de los 256 PEs.
- Lineales en conjunto: 258 M ideales / 313 M medidos = **82 %**. Margen real pequeño: ≈ 0.1–0.2 s (est.).
- Atención en el motor: 75.7 M ideales / 140.8 M medidos = **54 %**. Aquí está el margen del motor:
  - `S`: 707 K ciclos por cabeza contra 263 K ideales, ≈ 167 ciclos por bloque de 16×16 con K = 64. La hipótesis
    (?) es que la salida `RAW32` a 1 fila cada 4 ciclos (64 ciclos/bloque) no se solapa con los 64 de cálculo.
  - `P·V`: 759 K contra 526 K (?). Posibles causas: la carga del panel de A (7 subtiles × 2052 × 16 B ≈ 7.2 K beats
    por panel, ≈ 11 %) no se solapa con el cálculo, y hay vaciados entre pasadas.
  - Se confirma en simulación (M4) antes de tocar RTL.

### 1.3 Mediciones a hacer antes de gastar compilaciones

Ninguna necesita Quartus. Todas son software o simulación sobre el bitstream actual.

| # | Medición | Cómo | Para decidir |
|---|---|---|---|
| **M1** | **Línea de tiempo por imagen** (diagrama de Gantt): inicio y fin de cada operación del motor, de cada softmax y de cada paso del ARM | marcas de `now_s()` en `pl_run`/`attention_fpga` a un buffer y volcado CSV con `--trace` | Explicar los 7.64 s con error < 3 %; ver de dónde salen los 0.46 s de "ARM esperando" (llenado y vaciado por tramo, o dependencias) |
| **M2** | **Concurrencia y ancho de banda real de `f2h_sdram0`** | (a) softmax de una cabeza sola; (b) `S` sola; (c) las dos a la vez, en buffers distintos; (d) igual con el ARM copiando a la región. Usar los contadores `CYCLES`/`BYTES_RD`/`BYTES_WR` que ya existen en los dos bloques de CSR | Viabilidad y ganancia real de la fase 1a. **Es la primera vez que los 4 maestros estarían activos a la vez**: empezar con 32 filas y luego una cabeza completa |
| **M3** | **ARM**: (a) reparto de los 0.84 s de LayerNorm entre sumas y bucle de división; (b) lectura sin caché con `ldm` de 8 registros y `vldm` de 128 B frente a `memcpy` (`hps_bench`); (c) `zcat /proc/config.gz \| grep -E 'STRICT_DEVMEM\|CMA\|MODULES'`, `uname -r`; (d) volcado del DTB: `dtc -I fs /proc/device-tree \| grep -B2 -A6 ff2` | microbancos y comandos de solo lectura | Decidir 1c, B2 y B3; ver qué nodos del DTB apuntan al puente ligero antes de la limpieza (§3) |
| **M4** | **Desglose de ciclos en simulación**: contadores por estado en `attn_softmax` (`S_RD`, `S_P2`, `S_P3`, `S_OUT`) y en `tile_sequencer` (cálculo / carga de A / espera de W / vaciado) para las formas de `S` (`RAW32`, K = 64) y de `P·V` (16 b, K = 1026) | `tb_cycles` / `tb_attn_core` con 32–64 filas (ModelSim ASE, minutos) | Si G1/G2 valen la pena y cuánto RTL cuestan |
| **M5** | Barrido de `--chunk` (128, 192, 256, 320) en la placa | ya existe la opción | Ajuste sin código; los 0.46 s de espera |
| **M6** | Recursos de cada cambio de RTL con **solo Analysis & Synthesis** (`quartus_map`) | minutos frente a 1 h | Que la tanda de RTL quepa (objetivo: M10K ≤ 80 %, ALMs ≤ 72 %) antes de la compilación completa |
| **M7** | Camino exacto de la violación de hold | `quartus_sta` sobre la base de datos de la #4 (sin recompilar): `report_timing -hold -npaths 5` en la esquina lenta de 0 °C | Si pasa por enrutado dedicado de DSP (`chainin`/`scanin`), el router no puede añadir retardo y hace falta RTL; si es el registro duplicado, basta `ADV_NETLIST_OPT_ALLOWED "NEVER ALLOW"` sobre ese registro (?) |

---

## 2. Tabla priorizada de optimizaciones

Esfuerzo: **S** ≤ 1 día, **M** 2–4 días, **L** 1–2 semanas (est.). Verificación: **D** = Docker/qemu
(`make test-kernels`, `make test-sim`), **Sim** = ModelSim, **P** = placa, **Q** = compilación de Quartus.
"Pared" = reducción de la latencia por imagen en el régimen indicado.

| Pri. | ID | Propuesta | Ganancia en pared (est.) y cálculo | Recursos | Riesgo de timing | Esfuerzo | Depende de | Verif. |
|---|---|---|---|---|---|---|---|---|
| 1 | **S1** | **Atención solapada por cabezas** (software). Softmax(h) a la vez que el motor hace `S`(h+1) y `P·V`(h−1); empaquetar Q/K/V(h+1) mientras corre h. Dos buffers de `S`, dos de `P`, dos de Q/K/V | **−0.8 a −0.9 s.** Por cabeza, motor = 707 K + 759 K = 1.47 M ciclos; el softmax (932 K) queda oculto. 96 × 1.47 M / 120 MHz = 1.17 s, más ≈ 0.15 s de llenado y vaciado por bloque (softmax de la última cabeza 7.8 ms + empaquetar la primera ≈ 3 ms, × 12) → ≈ 1.32 s frente a 2.22 | 0 (≈ 17 MB más de región) | ninguno | M | M2 (contención) | D (ampliar `accel_sim` con la unidad de softmax: `softmax_row_int` ya existe) + P |
| 2 | **B1** | **Cuantizar con NEON, fusionado en la salida de LayerNorm** (casi todas las cuantizaciones siguen a una LN: q/k/v, fc1, fc2). La fila de 3 KB queda en L1, así que no se escribe ni se relee `y` | **−0.6 a −0.9 s** mientras manda el ARM. ARM: 1.18 → 0.15–0.3 s (≈ 45 M elementos; el mismo truco de número mágico + comprobación de empates, en vectores de 4; el caso cercano al empate sigue con `rintf(x/sc)` escalar) | 0 | ninguno | S–M | – | D (repetir los 226 M valores de `test_quant` contra `rintf`) + P |
| 3 | **C** | **Atención por trozos de filas de consulta dentro de la tubería**: la atención pasa a ser una serie de pasos de `pl_run`, con el softmax como segundo recurso. El tramo siguiente arranca con el trozo 0 mientras el motor hace la atención del trozo 1 | **−0.8 a −1.2 s** después de S1 y B1. Total → patch + max(ARM ≈ 3.5, motor 3.78) + llenado ≈ 4.2–4.6 s. La barrera de K/V completas por bloque se mantiene (la atención del bloque b+1 necesita todas las filas de k y v) | 0 | ninguno | L | S1, M1 | D + P |
| 4 | **P** | **`patch_embed` en el motor** (`RAW32`; K = 588 es par, N = 768, M = 1024; \|acc\| ≤ 588·127² < 2³¹). El ARM despliega, cuantiza, empaqueta (0.6 MB) y convierte los acumuladores a float con el mismo código de hoy | **−0.40 s.** Motor: 0.46 GMAC / ≈ 209 = 2.2 M ciclos = 18 ms; ARM ≈ 40–60 ms (leer 3.1 MB de `RAW32` a ≈ 155 MB/s ≈ 20 ms) | pesos empaquetados ≈ 0.47 MB | ninguno | S | – | D + P |
| 5 | **E** | **Subir el reloj a 140 MHz** (fallback 135/130), en la tanda de RTL nº 1 con A′ y los arreglos de caminos (§4, fase 3) | **−0.5 s** si el motor es el cuello (3.78 × (1 − 120/140) = 0.54); hoy solo ≈ −0.3 s (la parte lineal está oculta tras el ARM) | – | **alto**; Fmax hoy ≈ 121 MHz (med.) | L | A′, F2 terminada | Sim + Q + P |
| 6 | **G1** | `RAW32` a 2 ciclos/fila (un beat por ciclo) y vaciado solapado con el cálculo del bloque siguiente | **−0.2 a −0.3 s** con el motor como cuello: `S` 707 K → ≈ 400 K (?) por cabeza, × 96 / 120 MHz | pocos ALMs | bajo–medio (camino de 512 bits de vaciado) | M | M4 | Sim + Q |
| 7 | **G2** | `P·V`: solapar la carga del panel de A con el último tile del panel anterior (A partida en dos mitades de 128 KiB, 3 subtiles por mitad) o más subtiles por panel | **−0.1 a −0.15 s** con el motor como cuello (≈ 11 % de 72.9 M ciclos) | 0 M10K (reparte la misma RAM) | medio (RAM A ya congestionada) | M | M4 | Sim + Q |
| 8 | **B3** | **Región con caché o con escritura combinada** (`u-dma-buf` o `reserved-memory` + módulo pequeño con `sync_for_cpu/device`). Hoy `/dev/mem` mapea fuera de la RAM de Linux como *strongly-ordered* | **−0.4 a −0.8 s** de ARM (leer 0.83 + empaquetar 0.65 + Q/K/V ≈ 0.3 son ancho de banda de la región; heap → heap da 280 MB/s frente a 33–155). En pared solo cuenta mientras mande el ARM | 0 | ninguno | M–L | M3c; poder compilar módulos para el kernel de Terasic (?) | P (arranque: se recupera con la SD) |
| 9 | **H-QKV** | **Epílogo de atención en RTL**: RoPE en el epílogo de q/k (es entrelazado, así que el par `d, d+1` está en el mismo tile de 16 canales, [ops.c:176-206](hps/src/ops.c#L176-L206)) + **escritor de formato** (transpositor 16×16 de bytes + direcciones) que escribe Q en formato de activaciones, k en formato Kᵀ, v en formato V y el ctx de `P·V` en formato de activaciones de `proj` | ARM **−0.8 a −0.9 s** (RoPE 0.30 + leer q/k/v ≈ 2.6 MB/bloque ≈ 0.25 + empaquetar Q/K/V ≈ 0.3 + ctx). En pared: de 0 a −0.8 según el régimen | transpositor ≈ 1–1.5 k ALMs (est.); RoPE 8 multiplicadores 16×16 en lógica ≈ 2 k ALMs (est.); tablas cos/sin: **4 KB si son separables por fila/columna de la rejilla 32×32, 256 KB si no (no caben)** (?) | medio | L | M3, decisión tras F2 | Sim + Q + P |
| 10 | **H-SG** | **SwiGLU en el epílogo** (`swiglu_lane.sv` ya verificado): tiles de gate y up intercalados y buffer de la salida gate de un trozo (256 filas × 16 × 16 b = 7 M10K) | ARM **−0.6 s** (SwiGLU 0.45 + leer `g16`/`u16`: 5.05 → 2.5 MB/bloque ≈ −0.19). En pared, igual que H-QKV | ≈ 3–5 k ALMs con 4 carriles sin DSP (est., medir con M6) + ≈ 9 M10K | medio | M–L | decisión tras F2 | Sim + Q + P |
| 11 | **B2** | LayerNorm: sustituir `vdiv.f32` (≈ 15 ciclos, no segmentada en el A9) por recíproco + comprobación exacta en double (`t·den` es exacto en 48 bits) + división real solo si falla. Mismo esquema que el cuantizado | **−0.1 a −0.25 s** de ARM (≈ 28 M divisiones/imagen ≈ 0.23 s con 2 núcleos (est.)) | 0 | ninguno | M | M3a | D (prueba exhaustiva como `test_ln`) |
| 12 | **A** | **Limpieza del `.qsys` y del top** (§3) | **0 s directos.** Libera ≈ 2.5 k ALMs, 5 M10K, ≈ 3.5 k registros, y quita los cruzadores de reloj y adaptadores del interconnect de `f2h_sdram0`. Facilita E | −2.5 k ALM | positivo pero pequeño | S | M3d | Q + P (arranque) |
| 13 | **A′** | **Árbitro propio para `f2h_sdram0`**: exportar el puerto del HPS y arbitrar los 4 maestros en RTL. Nuestros DMA nunca frenan `readdata`, así que basta una FIFO de IDs de ráfaga (16 × 2 bits). Desaparece el interconnect Merlin: `cmd_mux` 308 ALMs, `rsp_fifo` 342, traductores, agente y **la FIFO `rdata` de 2048 × 258 bits = 52 M10K** | **0 s directos; es la palanca de timing más fuerte.** M10K 474 → ≈ 417 (86 → 75 %); quita dos de los caminos que impidieron los 150 MHz (congestión de M10K y "FIFO del interconnect → máximo") | −52 M10K, ≈ −0.9 k ALMs (est.) | **positivo** | M | `qsys-generate` para confirmar que el puerto se puede exportar (?) | Sim (`avalon_mem_model`) + Q + P |
| 14 | **R** | **Valla de direcciones** en `dma_rd`/`dma_wr` (y en los del softmax): error si la dirección sale de `0x30000000–0x3FFFFFFF` | 0 s; **evita que un fallo de RTL o de software corrompa Linux y cuelgue la placa** | ≈ 50–100 ALMs | ninguno | S | – | Sim + P (`--bad-addr`, pendiente desde H4) |
| 15 | **T** | Arreglo de hold (según M7) + `set_multicycle_path` desde los CSR estáticos (M, N, K, direcciones; estables muchos ciclos antes de `START`) | 0 s; robustez y margen para E | – | positivo | S | M7 | Q |
| 16 | **H-proxy** | Caché de veredictos por SHA-256 de los bytes de la imagen; tamaño mínimo para juzgar | **0 s por imagen nueva.** Pero con una imagen a la vez, la imagen k de una página espera k × 7.7 s: lo que nota el usuario baja mucho. La caché no cambia ningún veredicto; el tamaño mínimo es una **decisión de política** (deja pasar imágenes sin juzgar) | 0 | – | S | – | D (pruebas del proxy) |
| – | F | Softmax a 8 carriles / solape de filas | **≈ 0 s después de S1**: el softmax por cabeza (932 K) queda por debajo del motor por cabeza (1.47 M, y ≥ 1.0 M aun con G1+G2). Cuesta ≈ +2 k ALMs (4 `softmax_idx_lane` × 510). **Aplazar** | | | | | |
| – | D-LN | LayerNorm y cuantizar en la FPGA | **Descartar.** Son float32 IEEE (con sumas en double, división y `sqrtf`). Ser bit-exacto exige reproducir cada operación con redondeo correcto y en el mismo orden, con 0 DSP: ≈ 6–10 k ALMs (est.) y verificación larga. Se gana más con B1+B2 en el ARM | | | | | |
| – | Fusión `S`→softmax en chip | Sin pasar `S` por la DDR3 | **Descartar.** El máximo necesita la fila entera; 16 filas × 1040 × 4 B = 66 KB y el orden actual (tile por fuera) obliga a todo el panel (≈ 525 KB). No cabe en las 79 M10K libres. S1 da casi todo el beneficio | | | | | |
| – | Último bloque solo con CLS | | **No aplica**: la cabeza promedia los 1024 parches ([model.c:846-852](hps/src/model.c#L846-L852)) | | | | | |
| – | 16 bits en una pasada | | **Imposible** con 112 DSP (§1.2) | | | | | |

---

## 3. Limpieza del `.qsys` y del top: qué se gana de verdad

Cifras del informe del Fitter de la compilación #4 (`output_files/DE10_Standard_GHRD.fit.rpt`, tabla
"Fitter Resource Utilization by Entity", columna "ALMs needed") (med.). Lo recuperado tras re-ajustar puede variar
±20 % (est.).

| Componente | ALMs | Registros | M10K | ¿Seguro quitarlo? | Efecto |
|---|---|---|---|---|---|
| `f2sdram_only_master` (JTAG → `f2h_sdram0`) | 347 | 513 | 1 | Sí (ya se vio que no sirve: colgó a los 60 s) | Recursos y **timing**: también se van los 2 cruzadores de reloj `clk_50`→PLL (53 + 36 ALMs) y los 2 adaptadores de ancho 32→256 (36 + 56) del interconnect del puerto, y el `cmd_mux` pasa de 5 a 4 entradas |
| `fpga_only_master` | 344 | 476 | 1 | Sí | Recursos (dominio `clk_50`) |
| `hps_only_master` + `mm_interconnect_2` (puente F2H AXI) | 363 + 367 | 502 + 707 | 1 | Sí. Deja el puente F2H configurado en el HPS y sin conectar: así no cambia el `hps_isw_handoff` (?) | Recursos |
| `interrupt_latency_counter` | 119 | 313 | 0 | Sí (solo mide latencia de IRQ) | Recursos |
| `jtag_uart` | 61 | 116 | 2 | Sí, **si el DTB no lo declara** (M3d) | Recursos |
| `led_pio`, `button_pio`, `dipsw_pio` | 41 | 83 | 0 | Sí, **si el DTB no los declara** (M3d) | Nada medible (dominio `clk_50`, lejos del camino crítico) |
| `mm_bridge_0` + `mm_interconnect_1` + su parte de `mm_interconnect_0` (adaptador de ráfaga 98, `rsp_fifo` 46, agente 15, un cruzador) | 42 + 217 + ≈ 175 | 138 + 174 + … | 0 | Sí, cuando ya no le quede ningún esclavo | Recursos |
| `debounce`, 3 × `altera_edge_detector`, `hps_reset` (`altsource_probe`) | 77 | 134 | 0 | Sí, atando `f2h_cold/warm/debug_rst_req_n` a 1 (inactivos) y `stm_hw_events` a 0. **No tocar `hps_fpga_reset_n` (h2f_reset)**: es el reset del sistema | Nada (son resets de depuración por JTAG) |
| `sld_hub` | 104 | 117 | 0 | Desaparece solo cuando no queda ningún nodo JTAG (vuelve si se usa SignalTap) | Recursos |
| `sysid_qsys` | ≈ 0 | – | 0 | Indiferente: no aparece en la tabla (constantes absorbidas en el multiplexor de lectura) | Nada. Mejor poner un ID de compilación en el registro `VERSION` del acelerador |
| **Total de la limpieza (A)** | **≈ 2.5 k (≈ 6 % del chip, 9 % de lo usado)** | ≈ 3.5 k | **5** | | Recursos: sí. Timing: algo en el puerto de la DDR3. **Latencia directa: 0** |
| **Interconnect de `f2h_sdram0` con árbitro propio (A′)** | ≈ −0.9 k neto | ≈ −2 k | **−52** | Se verifica en simulación | **Timing**: es el que ataca la congestión de M10K (86 % → ≈ 75 %) |

**Respuesta a la hipótesis del prompt.** Quitar lo heredado libera bastante más que "decenas de ALMs":
≈ 2.5 k ALMs, ≈ 3.5 k registros y 5 M10K. Aun así no reduce la latencia por sí mismo. Casi todo vive en el
dominio de 50 MHz, lejos de los caminos que fallan a 150 MHz. El beneficio de timing real está en el puerto
`f2h_sdram0`:

- **Con la limpieza:** fuera los cruzadores, los adaptadores de ancho y una entrada del multiplexor.
- **Sobre todo con A′:** la FIFO `rdata` del agente de esclavo es de 2048 × 258 bits y Platform Designer la
  dimensiona con los pendientes y la ráfaga máxima que declara el puerto del HPS, no con la ráfaga de 16 de
  `dma_rd`. Son 52 M10K en el centro del chip, y un camino de 256 bits que ya apareció como crítico.

**Antes de quitar nada (sin riesgo):**

1. M3d: volcar el árbol de dispositivos en marcha y buscar nodos en `0xFF20xxxx`. Como `mm_bridge_0` ya se movió a
   `0xFF240000` y Linux arranca, lo probable es que no haya drivers ligados a esos periféricos (?). Pero si un driver
   sondea una dirección sin esclavo, el puente ligero responde con error de decodificación y el kernel puede caer al
   arrancar.
2. Nada de la limpieza toca el controlador SDRAM ni los puertos FPGA→SDRAM, así que **no requiere regenerar el
   preloader**. A′ tampoco: el puerto sigue siendo uno de 256 bits; solo cambia quién lo maneja.
3. La limpieza y A′ van en la **misma compilación** que E (fase 3), para no gastar una hora solo en ellas.

---

## 4. Orden de ejecución por fases

Regla general, la misma que ya se usa: cada cambio de software va detrás de una opción para desactivarlo
(`--no-att-overlap`, `--cpu-patch`, …). Con y sin la opción, los logits deben ser idénticos a 9 cifras (`--logits`).
El mapa de CSR **no cambia** en la fase 3, para que el software de las fases 1–2 corra con los dos bitstreams.

**Compuerta de exactitud (todas las fases):**

1. `make test-kernels` y `make test-sim` (D);
2. `--check --blocks 1` en la placa ("every output identical");
3. `--selftest 20`: 20/20 decisiones y logits idénticos al backend de CPU;
4. `make test-proxy` y una prueba real con el proxy.

| Fase | Contenido | Dónde se verifica | Criterio de paso | Proyección (est.) | Punto de retorno |
|---|---|---|---|---|---|
| **F0 – Medir** | M1–M5 y M7 (instrumentación `--trace`, prueba de concurrencia, microbancos, DTB) | P (solo lectura y operaciones conocidas) + Sim | El Gantt explica 7.64 s con error < 3 %; la concurrencia softmax+motor no cuelga y su ralentización queda medida | 7.6 s | etiqueta `accel-opt-f0` |
| **F1 – Software rápido** | S1, P, B1, ajuste de `--chunk` | D, luego P | Compuerta de exactitud; ≤ 6.0 s; `ARM waiting` medido | **≈ 5.1–5.8 s**: atención 1.32 + patch 0.08 + tubería ≈ max(3.5 + esperas, 2.62) ≈ 3.9 | `accel-opt-f1`; las opciones vuelven a la ruta anterior sin git |
| **F2 – Software estructural** | C (y, en paralelo, investigar B3 y B2) | D (`accel_sim` con softmax), luego P | Compuerta de exactitud; Gantt con **motor ocupado ≥ 85 % de la pared** (o ARM, y entonces se sabe cuánto falta) | **≈ 4.2–4.6 s** | `accel-opt-f2` |
| **Decisión** | Con el Gantt de F2: si manda el motor → F3 primero; si manda el ARM → F4 (o B3) antes que F3 | | | | |
| **F3 – RTL nº 1 "cierre de tiempos"** | A + A′ + R + T + arreglos de caminos (replicar `a_waddr_p` por banco de RAM A y una etapa más en `WR_PIPE`; árbol de registros de 2 niveles para la difusión a las 16 RAM de fila del softmax; registrar la entrada de las comparaciones de máximo) + G1 (+ G2 si M4 lo justifica). PLL a **140 MHz** | Antes de Quartus: los 14 testbenches + nuevos (`tb_arbiter`, valla de direcciones) en PASS; M6 (`quartus_map`) con M10K ≤ 80 % y ALMs ≤ 72 %. Luego una compilación | Setup y hold ≥ 0 en las 4 esquinas. Placa: `accel_test regs` 9/9, capas completas bit-exactas, compuerta de exactitud | **≈ 3.4–3.8 s** si el ARM no vuelve a mandar | `accel-pre-f3` + `.sof`/`.rbf` guardados. Si falla el setup por < 0.3 ns: misma RTL a 135 y luego 130 MHz (`ACCEL_MHZ`, una compilación cada una) |
| **F4 – RTL nº 2 "descargar el ARM"** (solo si el Gantt lo pide) | H-QKV y/o H-SG | Sim con vectores reales (como `tb_swiglu_lane`) + M6 + una compilación | Igual que F3 | **≈ 3.3–3.5 s** (cota del motor a 140 MHz con G1/G2 ≈ 3.0 s + llenado) | `accel-pre-f4` |

**Cómo reducir el riesgo de colgar la placa:**

- **Volver a un bitstream conocido sin sacar la SD:** guardar en la FAT `soc_system_h5.rbf` junto al nuevo y,
  desde la consola de U-Boot, `fatload mmc 0:1 <dir> soc_system_h5.rbf; fpga load 0 <dir> <tamaño>` antes de
  `bridge_enable_handoff`. Comprobar los nombres reales en el `u-boot.scr` de Terasic (?).
- **Lo primero que hace el software con un bitstream nuevo:** leer `VERSION` en `0xFF200000` y `0xFF200100` y
  comparar con lo esperado. Nunca acceder al puente con la FPGA sin configurar (lección de H4).
- **Concurrencia (F1):** probar primero con operaciones de 32 filas y buffers separados, y después con cabezas
  completas.
- **La valla de direcciones (R)** convierte un error de direcciones en `ERR_CODE` en lugar de en un Linux corrupto.

**Menos horas de Quartus:**

- Solo dos tandas de RTL (F3, F4), cada una precedida de simulación y de `quartus_map` (minutos).
- No contar con compilación incremental en la edición Lite (?).
- Si la PC tiene memoria de sobra, lanzar en paralelo dos semillas de la misma tanda en copias del proyecto.
- El hold (T) se diagnostica sobre la base de datos de la #4 con `quartus_sta`, sin recompilar.

---

## 5. Preguntas abiertas y supuestos sin comprobar

1. **Contención en la DDR3** con softmax + motor + ARM a la vez (S1, C). Nunca se ha probado con los 4 maestros
   activos. La ganancia de S1 supone menos de un 10 % de ralentización (?). Lo resuelve M2.
2. **De dónde salen los 0.46 s de "ARM esperando"** (llenado y vaciado de cada tramo o dependencias) y cuánto se
   quedará después de C. Lo resuelve M1.
3. **Ineficiencia de `S` y `P·V`:** atribuirla a `RAW32` y a la carga del panel es una hipótesis. Lo resuelve M4.
4. **Viabilidad de B3:** `CONFIG_STRICT_DEVMEM`, CMA, versión del kernel de Terasic y si hay cabeceras para compilar
   un módulo. Lo resuelve M3c.
5. **Nodos del DTB en el puente ligero** antes de la limpieza. Lo resuelve M3d.
6. **Tablas de RoPE:** ¿`cos_tab`/`sin_tab` (1024 × 64) se separan por fila y columna de la rejilla de 32 × 32?
   Decide si H-QKV cabe (4 KB frente a 256 KB). Se comprueba en el paquete r7, sin placa.
7. **Exportar `hps_0.f2h_sdram0_data`** desde Platform Designer y manejarlo con RTL propia (A′). Supongo que sí; se
   confirma con `qsys-generate` y leyendo el `soc_system.v` generado, sin compilar.
8. **Por qué falla el hold** (DSP en cascada o registro duplicado). Lo resuelve M7.
9. **Fmax alcanzable** tras A + A′ + arreglos: 140 MHz es una apuesta razonable porque sin softmax ya cerraba a
   150 con +0.198 ns. Pero eso fue con 70 % de M10K, y con A′ se quedaría en ≈ 75 % (est.).
10. **Coste real en ALMs de H-SG y de RoPE** sin DSP: mis cifras son de orden de magnitud. Se resuelve con M6.
11. **Tráfico total por imagen** (≈ 1.7 GB, est.): se mide sumando `BYTES_RD/WR` de todas las operaciones en M1.
12. **El proxy comparte el ARM:** TLS y decodificación (75–125 ms por imagen, med.) compiten con la inferencia. Con una
    imagen a la vez no importa; con varias pestañas sí.

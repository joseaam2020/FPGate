# Contexto rápido para la siguiente sesión (2026-11-13)

**F4 en placa (2026-11-18): 3.94 s/img**, PASS, 20/20, peor diferencia 0.07451. `accel/rbf_out/f4_130MHz/` (seed 2,
setup +0.052 / +0.009, HEAD b936832 + T43 535533c), FEATURES 0x7, reloj 130 MHz. Por imagen: tubería 2.48 s (antes
3.18), atención 1.06, patch_embed 0.33, resto 0.07. En la SD: `soc_system.rbf` = F4, `soc_system_h6.rbf` = copia.
La compuerta se ejecuta desde `/home/root/fpgate` (ahí están `pkg/` y `e1_r7.idx`). Traza: `/home/root/f4.csv`.
Traza `f4` (1.ª imagen): tubería ARM 2.11 s (`rlp` 0.99, `mlp` 0.92 = LN 3072 + cuant., `read` 0.10) frente a motor
2.30 s de pared (2.16 de ciclos): ahora limita el motor dentro de la tubería (espera 0.37 s). Atención 1.03 s
aparte: motor 0.78, `softmax` ARM 0.75, `att_pack` 0.19. Total ARM ≈ 3.4 s frente a motor ≈ 3.0 s: el límite global
vuelve a ser el ARM (LayerNorm + softmax). Siguiente: acelerar LN/softmax en el ARM (exacto), luego T20; 140 MHz al final.

**Actualización 2026-11-17:** **5.15 s/img** (trazas `f2c`, `f2d`). Causa real de que T12 no rindiera: gcc convertía
`vcltq_f32` en comparaciones escalares con `vmrs` → comparaciones enteras (5a8c214) y una comprobación de máscara
por bloque de 64 (abf40fc), recalculando solo carriles dudosos (6897f10): **5.11 s/img** (`f2e`).
**T26 (edb8a86, `f3b`): 4.69 s/img**, PASS. `rlp` 8 ms/chunk (antes 8.7 en 4 pasos: LayerNorm es cálculo). ARM 2.88 s,
motor 2.34 s de ciclos / 2.63 de pared. Lo que queda en el ARM por imagen: `mlp` 1.18, `rlp` 0.99, `read` 0.30,
`rope` 0.30. Tras F4 (−RoPE −SwiGLU ≈ 0.75 s) el ARM quedaría por debajo del motor: entonces manda la frecuencia.

**h6 de prueba (2026-11-17, `f3a`): 4.86 s/img**, PASS ×2. `accel/rbf_out/h6_120MHz_test/` (T30–T34 sin 031310f,
setup −0.056 en slow 85C, como h5b). En la SD: `soc_system.rbf` = h6, `soc_system_h5b.rbf` = copia. Atención
1.43 → 1.10 s (S de 717 K a 293 K ciclos). Lineales 2.59 → 2.34 s de ciclos, pero la tubería sigue en 3.34 s porque
el ARM (2.98 s) vuelve a ser el límite. Siguiente: T26 (software) y F4 (T41/T42 + 031310f, 130–140 MHz).
Los hashes de commits de abajo cambiaron al reescribir la rama (p. ej. 6897f10 → 2ad8f64).

**Equilibrio anterior (f2e):** en la tubería el ARM trabaja 2.96 s y el motor 3.01 s de pared (2.59 s de ciclos): ya están
a la par. A partir de aquí el motor también limita: lo siguiente es h6 (T34, sombras), F4 (RoPE/SwiGLU en el
epílogo, sin ciclos extra del motor) y subir a 130–140 MHz con los 3 caminos de T35 arreglados.

**Estado anterior:** 7.6 → **5.95 s/img**, bit a bit exacto (compuerta `gate.sh` PASS, 20/20). Bitstream en la placa: h5b
(T37, VERSION 2) sin cerrar timing pero estable; h6 (T30–T34, 120 MHz) compilándose en la sesión coordinadora (T35).
Módulo `fpgate_mem.ko` cargado al arranque en la placa (región 0x30000000 con caché).

**Lo que enseñaron las trazas** (`accel/board_runs/f1/*.csv`, análisis en `accel/TAREAS_OPTIMIZACION.md`,
apartado "Lo que enseñó la placa"): la tubería la limita el ARM (3.78 s ocupado de 4.10 s; el motor trabaja 2.6 s).
`prep`/`read` eran bytes (arreglado con la caché); la cola del MLP (SwiGLU + LayerNorm 3072 + cuantización 16 bits,
paso `mlp` = 23 ms/chunk) es **cálculo** en el A9, no memoria: la fusión (T25) apenas ahorró.

**Commits de hoy:** 6e2ab6d (división LN exacta), 664f88c (módulo con caché), 56533bf (T25 fusión MLP), 1c71156
(T27 tick del motor + umbral NEON). Pendiente sin commit: `accel/tools/board/step_bench.c` (micro-banco).

**Siguiente paso inmediato:** el usuario ejecuta en la placa `OMP_NUM_THREADS=2 ./step_bench` y `=1` (binario ya
compilado, `scp accel/tools/board/step_bench`). Con esos `RESULT bench` decidir:
- si LayerNorm/SwiGLU dominan → van a la FPGA (F4: T41 RoPE, T42 SwiGLU; añadir LN 3072 si procede) y T26 se recorta;
- si `pack`/`quantize` dominan → reescribirlos (T26 tal cual).
Después: flashear h6 cuando T35 termine (procedimiento igual que h5b, en este archivo de tareas T36), medir con
`--trace`, y lanzar F4 con los caminos de 140 MHz.

**Reglas:** agentes leen solo "Contexto común" + su bloque de `TAREAS_OPTIMIZACION.md`; toda tarea de rendimiento del
ARM se acepta con medida en la placa (traza), nunca en x86/qemu; el usuario ejecuta todo lo de la placa; commits
convencionales en inglés, una línea, sin Co-Authored-By. Memoria: `project_arm_memory_bound.md`.

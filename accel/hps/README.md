# fpgate-accel (lado HPS del acelerador)

Programa en C que atiende al proxy por un socket Unix: recibe una imagen RGB de 448×448, corre el modelo r7 y devuelve `score_nsfw`. Sigue el protocolo de `proxy/src/accel_proto.h` (fuente de verdad, no se copia aquí) y el comportamiento del `mock_accel`: un cliente y una imagen a la vez, `UNSUPPORTED` para cualquier cosa que no sea 448×448 RGB8 HWC, `STATUS` responde `OK`.

**Estado:** backend de CPU completo y verificado (fase 1). Fase 2: con `--fpga` las 84 capas lineales corren en la FPGA, verificadas en la placa (identicas a la CPU); la atención sigue en el ARM (51.7 s por imagen), así que una imagen tarda ≈ 63 s. Mediciones y plan en `accel/DECISIONES.md`.

## Archivos

| Archivo | Qué hace |
|---|---|
| `src/main.c` | servidor del socket, y `--selftest` (20 vectores, etapas intermedias, normalización) |
| `src/model.c` | pasada completa del modelo, siguiendo `model/quant/export_check.py` (la especificación ejecutable del paquete) |
| `src/ops.c` | kernels: cuantización, GEMM INT8/INT16, epílogo de punto fijo, LayerNorm, RoPE entero, atención entera (softmax por tablas) |
| `src/pkg.c` | carga el paquete r7 por `mmap` y verifica SHA-256 |
| `tools/gen_pkg_index.py` | convierte `manifest.json` en un índice de texto plano (`pkg/e1_r7.idx`); solo lee el paquete |
| `tools/stages_to_raw.py` | vuelca las etapas intermedias de la imagen 0 (`stages_img_00.npz`) a archivos crudos para la autoprueba |
| `tests/test_protocol.py` | prueba del protocolo contra el binario real |

## Compilar y probar (en el PC, desde PowerShell en la raíz del repositorio)

```
docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make native
docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make arm        # estático, para la placa

python accel/hps/tools/gen_pkg_index.py model/quant/release/e1_r7 accel/hps/pkg/e1_r7.idx
docker run --rm -v "${PWD}/model:/model:ro" -v "${PWD}/accel/hps:/hps" --entrypoint python model-torch:latest /hps/tools/stages_to_raw.py /model/quant/release/e1_r7/vectors/stages_img_00.npz /hps/pkg/stages_img_00

# 20 imágenes de referencia + etapas intermedias + kernels enteros con entradas de referencia
docker run --rm -v "${PWD}:/repo:ro" -w /repo/accel/hps fpgate-proxy-dev build/native/fpgate-accel --selftest 20 --pkg /repo/model/quant/release/e1_r7 --index pkg/e1_r7.idx --stages pkg/stages_img_00

# protocolo
docker run --rm -v "${PWD}:/repo:ro" -w /repo/accel/hps fpgate-proxy-dev python3 tests/test_protocol.py build/native/fpgate-accel /repo/model/quant/release/e1_r7 pkg/e1_r7.idx
```

## Uso

```
fpgate-accel /run/fpgate-accel.sock --pkg <dir del paquete> --index <e1_r7.idx> [--verify] [--quiet]
```

`--verify` comprueba el SHA-256 de los cinco `.bin` al arrancar (lee ≈ 60 MB). El paquete (55 MB de pesos) se mapea con `mmap` y se queda en el caché de páginas; no hace falta copiarlo a la región de la FPGA hasta que exista el backend de FPGA.

## Qué se exige y qué no (criterio de exactitud)

- **Enteros (GEMM, epílogo, RoPE, softmax, `P·V`, SwiGLU): bit a bit.** Alimentados con las entradas de referencia de los bloques 0 y 11, dan 0 diferencias. El hash de `patch_embed` y la normalización `uint8 → float32` también coinciden exactos.
- **Partes en flotante (LayerNorm, residual): tolerancia.** PyTorch suma en otro orden y en otra CPU/GPU, así que no se puede reproducir el patrón de bits. Efecto: algunos valores cuantizados cambian 1 LSB y la diferencia se acumula por los 12 bloques.
- **Resultado:** en las 20 imágenes la decisión coincide (20/20); diferencia máxima de logits 0.075 (rango ≈ ±10), de puntaje 0.008.

## Límites conocidos

- Un solo núcleo en la placa (las pragmas de OpenMP solo se activan en el build nativo). En la placa esta versión es la referencia de corrección, no la de velocidad: los GEMM de las 12 capas en el ARM cuestan decenas de segundos por imagen (ver `accel/DECISIONES.md`, mediciones del HPS).
- `hw_cycles` se devuelve en 0 (no hay hardware en este backend).

## Con la FPGA (en la placa)

Requisitos: FPGA configurada por U-Boot (`accel/tools/board/README.md`: `SW10` = `01010`, `soc_system.rbf` en la SD) y `cat /sys/class/fpga_manager/fpga0/state` = `operating`. Copiar a la placa el binario ARM, `pkg/e1_r7.idx`, los cinco `.bin` del paquete en `pkg/` y `vectors/img_00.f32` en `pkg/vectors/`.

```
./fpgate-accel --selftest 1 --pkg pkg --index e1_r7.idx --fpga --check --verbose --blocks 1   # bloque 0, cada capa contra la CPU
./fpgate-accel --selftest 1 --pkg pkg --index e1_r7.idx --fpga                                # imagen completa, con desglose por etapa
./fpgate-accel /run/fpgate-accel.sock --pkg pkg --index e1_r7.idx --fpga                      # servidor para el proxy
```

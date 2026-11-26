# fpgate_mem: región DDR3 del acelerador con caché

El kernel de la placa (Terasic, `4.5.0-00200-g1ec9f95`) gestiona la RAM hasta 0x2FFFFFFF; la ventana del
acelerador (0x30000000, 128 MiB) queda fuera y `/dev/mem` la mapea sin caché (111 MB/s escritura,
155 MB/s lectura desde el Cortex-A9). Este módulo la mapea con caché de escritura diferida y ofrece
`ioctl` para limpiar o invalidar rangos (L1 por dirección virtual del kernel, L2 PL310 por dirección
física). `accel/hps/src/accel_drv.c` lo usa solo si existe `/dev/fpgate_mem` (`--cached` lo exige,
`--uncached` lo evita). `accel_test` fuerza el modo sin caché porque copia a la región por su cuenta.

Medido con la compuerta (h5b, 2026-11-13): 6.55 → 6.06 s por imagen, bit a bit idéntico.

## Compilar (PC, una vez)

```powershell
docker build -t fpgate-kmod accel/tools/kmod
docker run --rm -v fpgate-ksrc:/ksrc -v "${PWD}:/repo" fpgate-kmod sh /repo/accel/tools/kmod/fetch_kernel.sh
docker run --rm -v fpgate-ksrc:/ksrc -v "${PWD}:/repo" -w /repo/accel/tools/kmod fpgate-kmod sh build_module.sh
```

El fuente del kernel vive en el volumen Docker `fpgate-ksrc` (NTFS no admite `drivers/gpu/drm/nouveau/.../aux.c`).
`kconfig_4.5.txt` es el `/proc/config.gz` de la placa; `.scmversion` reproduce el `-00200-g1ec9f95` del
`vermagic`. El `.ko` queda en `out/` (sin versionar).

## Cargar en la placa

```sh
insmod /home/root/fpgate_mem.ko        # dmesg: "fpgate_mem: window 0x30000000+0x8000000 mapped cached"
ls -l /dev/fpgate_mem                  # lo crea devtmpfs; si no: mknod /dev/fpgate_mem c 10 $(awk '/fpgate_mem/{print $1}' /proc/misc)
```

Para cargarlo en cada arranque, añade la línea `insmod` a `/etc/rc.local` (o al script que arranca el proxy).

## Reglas de coherencia (ya aplicadas en `model.c`)

- Tras cada `memcpy` del ARM a la región que el acelerador leerá: `accel_hw_clean(hw, off, len)`.
- Antes de cada lectura del ARM de lo que el acelerador escribió: `accel_hw_inval(hw, off, len)`.
- El ARM nunca escribe en las zonas de salida del acelerador (así no hay líneas sucias que se escriban encima).
- No mezclar en el mismo proceso el mapeo con caché y el de `/dev/mem` sobre las mismas direcciones.

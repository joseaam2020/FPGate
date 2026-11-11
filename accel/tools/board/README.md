# Pruebas en la placa

Dos programas, ambos estáticos con musl y para correr como root en el HPS:

- `csr_test` (H0): comprueba los registros con el diseño del H0 (motor de relleno). Ya no aplica al diseño del H4.
- `accel_test` (H4): registros con el motor real, región de DDR3, capas reales del modelo r7 comparadas bit a bit, y pruebas de fallo.

## Compilar (en el PC, con la toolchain del proxy)

Desde la raíz del repositorio, en PowerShell:

```
docker run --rm -v "${PWD}/accel/tools/board:/src" -w /src fpgate-proxy-dev arm-linux-musleabihf-gcc -O2 -Wall -static accel_test.c -o accel_test
```

## Datos de prueba (en el PC)

```
# rebanadas 32x32 de 5 capas (ya generadas en accel/vectors/<capa>/ con dump_vectors.py + pack_layer.py)
# capas completas (1025 tokens): dump_full_layer.py las deja en accel/vectors/full_<capa>/
```

Copiar a la placa (ejemplo):

```
scp -O accel/tools/board/accel_test root@socfpga:/home/root/
scp -O -r accel/vectors/b0_q_w accel/vectors/b0_fc2_w accel/vectors/full_b0_q_w root@socfpga:/home/root/vectors/
```

## Región de DDR3 del acelerador

El acelerador usa los 256 MB superiores de la DDR3 (`0x30000000`…`0x3FFFFFFF`). **No hace falta reservarla:** la imagen de Linux de Terasic ya limita la memoria del sistema a 768 MB, así que esa zona queda libre. Se verifica en la placa (verificado el 2026-11-09, kernel 4.5.0-00200-g1ec9f95, U-Boot 2013.01.01 con `DRAM: 1 GiB`):

```
grep MemTotal /proc/meminfo          # unos 769000 kB
grep -i "system ram" /proc/iomem     # 00000000-2fffffff : System RAM
```

Si esa salida cambiara (otra imagen de Linux), habría que limitar la memoria con `mem=768M` en los argumentos de arranque (en U-Boot: `setenv mmcboot "setenv bootargs console=ttyS0,115200 root=${mmcroot} rw rootwait mem=768M; bootz ${loadaddr} - ${fdtaddr}"`, sin `saveenv`).

## Programar la FPGA

**Para `regs` basta el JTAG; para `layer` y `faults` NO.** Programar el `.sof` por USB-Blaster con Linux ya arrancado deja funcionar los registros (puente ligero), pero el DMA por el puerto SDRAM de la FPGA congela Linux. Para el motor completo hay que cargar la FPGA como lo hace Terasic:

1. `SW10` (MSEL) en `01010` (FPPx32). El valor de fábrica `10010` (AS) no sirve: U-Boot da `altera_load: Failed with error code -8`.
2. Copiar `soc_system.rbf` a la partición FAT de la SD (junto a `u-boot.scr`, `zImage`, `socfpga.dtb`):
   ```
   mount /dev/mmcblk0p1 /mnt/fat && cp soc_system.rbf /mnt/fat/ && sync && umount /mnt/fat
   ```
   El `.rbf` se genera en el PC con `quartus_cpf -c -o bitstream_compression=on h4_DE10_Standard_GHRD.sof soc_system.rbf` (sin `--hps`). Copia ya hecha en `accel/rbf_out/`.
3. Apagar y encender la placa (mejor en frío). En la consola de U-Boot debe verse `reading soc_system.rbf` sin `altera_load: Failed`.
4. Comprobar `cat /sys/class/fpga_manager/fpga0/state` = `operating` y `devmem 0xFFC25080 32` = `0x1FF`. **Con otro estado, no ejecutar `accel_test`**: leer los registros con la FPGA sin configurar cuelga el sistema.

Para cambiar el diseño: regenerar el `.rbf`, copiarlo a la SD y reiniciar (el JTAG sirve solo para pruebas de registros).

## Ejecutar

```
./accel_test regs
./accel_test mem
./accel_test layer vectors/b0_q_w vectors/b0_v_w vectors/b0_fc1g_w vectors/b0_fc2_w vectors/b1_fc2_w
./accel_test layer vectors/full_b0_q_w vectors/full_b0_fc2_w
./accel_test faults vectors/b0_q_w            # añadir --bad-addr para la prueba de dirección fuera de rango
```

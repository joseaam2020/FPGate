#!/bin/sh
# Prepares the board's kernel tree for out-of-tree modules (once) and builds fpgate_mem.ko.
# Runs inside the fpgate-kmod container:
#   docker run --rm -v fpgate-ksrc:/ksrc -v "${PWD}:/repo" -w /repo/accel/tools/kmod fpgate-kmod sh build_module.sh
# The tree comes from fetch_kernel.sh; the .config is the board's /proc/config.gz (kconfig_4.5.txt).
# The board's vermagic is "4.5.0-00200-g1ec9f95 SMP mod_unload ARMv7 p2v8": .scmversion supplies the
# "-00200-g1ec9f95" that LOCALVERSION_AUTO would take from git.
set -e
K=/ksrc/linux
export ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-
# gcc >= 10 defaults to -fno-common, which breaks the 4.5 host tools (yylloc duplicates); the kernel
# Makefile overrides HOSTCFLAGS from the environment, so it goes on the command line with the defaults
HOST="HOSTCFLAGS=-Wall -Wmissing-prototypes -Wstrict-prototypes -O2 -fomit-frame-pointer -std=gnu89 -fcommon"
if [ ! -f "$K/.prepared" ]; then
    cp kconfig_4.5.txt "$K/.config"
    echo "-00200-g1ec9f95" > "$K/.scmversion"
    make -C "$K" "$HOST" olddefconfig
    make -C "$K" "$HOST" modules_prepare
    touch "$K/.prepared"
fi
make -C "$K" "$HOST" M="$(pwd)" modules
"${CROSS_COMPILE}strip" --strip-debug fpgate_mem.ko
mkdir -p out
mv fpgate_mem.ko out/
rm -f *.o *.mod.c *.mod .*.cmd modules.order Module.symvers; rm -rf .tmp_versions
echo "--- vermagic (must be: 4.5.0-00200-g1ec9f95 SMP mod_unload ARMv7 p2v8)"
modinfo out/fpgate_mem.ko | grep -E "vermagic|depends"

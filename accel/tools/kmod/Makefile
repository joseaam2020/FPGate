# Kbuild file for the fpgate_mem module. Built by build_module.sh inside the fpgate-kmod container:
#   make -C /ksrc/linux M=$(pwd) ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- modules
obj-m := fpgate_mem.o
# the 4.5 Makefile's cc-option probe falls back to -march=armv5t with current gcc, and then the assembler
# rejects the isb/dsb of the cache maintenance; the last -march wins
ccflags-y := -march=armv7-a -Wa,-march=armv7-a -Wno-attributes -Wno-missing-attributes
# Debian's gcc defaults to PIE, which 4.5 does not disable (4.9 does): without this the module references
# _GLOBAL_OFFSET_TABLE_ and insmod reports "Unknown symbol"
ccflags-y += -fno-pie -fno-PIE -fno-pic -fno-PIC -fno-stack-protector

#!/bin/sh
# Downloads the board's exact kernel source (linux-socfpga, Terasic DE10-Standard commit) into /ksrc/linux.
# Runs inside the fpgate-kmod container with the fpgate-ksrc volume mounted at /ksrc.
set -e
SHA=1ec9f95fa0f2e4e0bb972d55d683294e51e70848
cd /ksrc
rm -rf linux k.tgz
curl -sSL -o k.tgz "https://github.com/altera-opensource/linux-socfpga/archive/$SHA.tar.gz"
mkdir linux
tar xzf k.tgz -C linux --strip-components=1
rm k.tgz
head -3 linux/Makefile
ls linux/arch/arm/configs | grep -i socfpga
du -sh linux

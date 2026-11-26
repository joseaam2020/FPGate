# Cross toolchain to build kernel modules for the DE10-Standard board (linux-socfpga 4.5, ARMv7 hard-float).
# Build:  docker build -t fpgate-kmod accel/tools/kmod
# Use:    docker run --rm -v "${PWD}:/repo" -v "<kernel source dir>:/ksrc" -w /repo fpgate-kmod ...
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-arm-linux-gnueabihf binutils-arm-linux-gnueabihf libc6-dev-armhf-cross \
        build-essential bc bison flex libssl-dev libelf-dev kmod cpio xz-utils u-boot-tools python3 \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
ENV ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf-

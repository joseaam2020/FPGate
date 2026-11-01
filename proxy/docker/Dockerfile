FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential make pkg-config ca-certificates \
        qemu-user \
        libssl-dev openssl valgrind \
        curl python3 socat file \
    && rm -rf /var/lib/apt/lists/*

# ARM cross toolchain based on musl (static binaries that do not depend on the
# board's glibc 2.20; glibc static getaddrinfo needs NSS libs at runtime).
ARG MUSL_ARM_URL=https://musl.cc/arm-linux-musleabihf-cross.tgz
RUN curl -fsSL "$MUSL_ARM_URL" -o /tmp/musl-arm.tgz \
    && echo "11e155a090789c854afb5d91e57c81f441ccdb92f7f277789c3a6d5884a68e58  /tmp/musl-arm.tgz" | sha256sum -c - \
    && tar -xzf /tmp/musl-arm.tgz -C /opt \
    && rm /tmp/musl-arm.tgz
ENV PATH="/opt/arm-linux-musleabihf-cross/bin:${PATH}"

WORKDIR /work/proxy

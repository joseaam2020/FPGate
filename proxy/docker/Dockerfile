FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential make pkg-config ca-certificates bzip2 \
        qemu-user \
        openssl valgrind \
        curl python3 python3-jsonschema python3-jinja2 socat file \
    && rm -rf /var/lib/apt/lists/*

# ARM cross toolchain based on musl (static binaries that do not depend on the
# board's glibc 2.20; glibc static getaddrinfo needs NSS libs at runtime).
ARG MUSL_ARM_URL=https://musl.cc/arm-linux-musleabihf-cross.tgz
RUN curl -fsSL "$MUSL_ARM_URL" -o /tmp/musl-arm.tgz \
    && echo "11e155a090789c854afb5d91e57c81f441ccdb92f7f277789c3a6d5884a68e58  /tmp/musl-arm.tgz" | sha256sum -c - \
    && tar -xzf /tmp/musl-arm.tgz -C /opt \
    && rm /tmp/musl-arm.tgz
ENV PATH="/opt/arm-linux-musleabihf-cross/bin:${PATH}"

# mbedTLS, built twice: /opt/mbedtls/native (x86-64) and /opt/mbedtls/arm (musl).
ARG MBEDTLS_VERSION=3.6.4
ARG MBEDTLS_SHA256=ec35b18a6c593cf98c3e30db8b98ff93e8940a8c4e690e66b41dfc011d678110
RUN curl -fsSL "https://github.com/Mbed-TLS/mbedtls/releases/download/mbedtls-${MBEDTLS_VERSION}/mbedtls-${MBEDTLS_VERSION}.tar.bz2" -o /tmp/mbedtls.tar.bz2 \
    && echo "${MBEDTLS_SHA256}  /tmp/mbedtls.tar.bz2" | sha256sum -c - \
    && mkdir /tmp/mbedtls-src \
    && tar -xjf /tmp/mbedtls.tar.bz2 -C /tmp/mbedtls-src --strip-components=1 \
    && cd /tmp/mbedtls-src \
    && for t in native arm; do \
         if [ "$t" = arm ]; then CCX=arm-linux-musleabihf-gcc; ARX=arm-linux-musleabihf-ar; \
         else CCX=gcc; ARX=ar; fi; \
         make -C library clean >/dev/null 2>&1; \
         make -j"$(nproc)" -C library CC="$CCX" AR="$ARX" CFLAGS="-O2 -fPIC" static \
         && mkdir -p /opt/mbedtls/$t/lib /opt/mbedtls/$t/include \
         && cp library/*.a /opt/mbedtls/$t/lib/ \
         && cp -r include/mbedtls include/psa /opt/mbedtls/$t/include/ \
         || exit 1; \
       done \
    && rm -rf /tmp/mbedtls-src /tmp/mbedtls.tar.bz2

WORKDIR /work/proxy

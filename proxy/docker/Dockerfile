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

# Image decoding stack. Pillow is only the reference implementation the tests
# compare our preprocessing against (the model was trained with PIL).
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake nasm xz-utils python3-pil \
    && rm -rf /var/lib/apt/lists/*

# zlib + libpng + libjpeg-turbo + libwebp, static, built for /opt/deps/native and
# /opt/deps/arm (musl, with NEON enabled: the HPS is a Cortex-A9).
ARG ZLIB_URL=https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
ARG ZLIB_SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
ARG LIBPNG_URL=https://download.sourceforge.net/libpng/libpng-1.6.47.tar.xz
ARG LIBPNG_SHA256=b213cb381fbb1175327bd708a77aab708a05adde7b471bc267bd15ac99893631
ARG LIBJPEG_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.1/libjpeg-turbo-3.1.1.tar.gz
ARG LIBJPEG_SHA256=aadc97ea91f6ef078b0ae3a62bba69e008d9a7db19b34e4ac973b19b71b4217c
ARG LIBWEBP_URL=https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.5.0.tar.gz
ARG LIBWEBP_SHA256=7d6fab70cf844bf6769077bd5d7a74893f8ffd4dfb42861745750c63c2a5c92c
COPY build-deps.sh /tmp/build-deps.sh
RUN sh /tmp/build-deps.sh && rm /tmp/build-deps.sh

# SQLite amalgamation (block log), static, for /opt/sqlite/native and /opt/sqlite/arm.
ARG SQLITE_URL=https://www.sqlite.org/2025/sqlite-amalgamation-3500400.zip
ARG SQLITE_SHA256=1d3049dd0f830a025a53105fc79fd2ab9431aea99e137809d064d8ee8356b032
RUN curl -fsSL "$SQLITE_URL" -o /tmp/sqlite.zip \
    && echo "${SQLITE_SHA256}  /tmp/sqlite.zip" | sha256sum -c - \
    && python3 -c "import zipfile; zipfile.ZipFile('/tmp/sqlite.zip').extractall('/tmp/sqlite')" \
    && cd /tmp/sqlite/sqlite-amalgamation-* \
    && for t in native arm; do \
         if [ "$t" = arm ]; then CCX=arm-linux-musleabihf-gcc; ARX=arm-linux-musleabihf-ar; \
         else CCX=gcc; ARX=ar; fi; \
         mkdir -p /opt/sqlite/$t/lib /opt/sqlite/$t/include \
         && $CCX -O2 -fPIC -DSQLITE_THREADSAFE=1 -DSQLITE_OMIT_LOAD_EXTENSION -DSQLITE_DEFAULT_MEMSTATUS=0 -c sqlite3.c -o sqlite3-$t.o \
         && $ARX rcs /opt/sqlite/$t/lib/libsqlite3.a sqlite3-$t.o \
         && cp sqlite3.h /opt/sqlite/$t/include/ \
         || exit 1; \
       done \
    && rm -rf /tmp/sqlite /tmp/sqlite.zip

WORKDIR /work/proxy

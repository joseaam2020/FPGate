# fpgate-proxy build. Targets: native, arm, asan, ubsan, test, clean
# Run inside the dev container: docker compose run --rm dev make <target>

PHR := third_party/picohttpparser
SRC := $(wildcard src/*.c) $(PHR)/picohttpparser.c
HDR := $(wildcard src/*.h) $(PHR)/picohttpparser.h Makefile

MBED_NATIVE := /opt/mbedtls/native
MBED_ARM    := /opt/mbedtls/arm
MBED_LIBS   := -lmbedtls -lmbedx509 -lmbedcrypto

BUILD_EPOCH := $(shell date +%s)
WARN     := -Wall -Wextra -Werror
HARDEN   := -O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
CSTD     := -std=gnu11
COMMON    = $(CSTD) $(WARN) -Isrc -I$(PHR) -I$(MBED)/include -DBUILD_EPOCH=$(BUILD_EPOCH)
LINK      = -L$(MBED)/lib $(MBED_LIBS)

.PHONY: native arm asan ubsan test clean

native: build/native/fpgate-proxy
arm: build/arm/fpgate-proxy
asan: build/asan/fpgate-proxy
ubsan: build/ubsan/fpgate-proxy

build/native/fpgate-proxy: MBED := $(MBED_NATIVE)
build/asan/fpgate-proxy:   MBED := $(MBED_NATIVE)
build/ubsan/fpgate-proxy:  MBED := $(MBED_NATIVE)
build/arm/fpgate-proxy:    MBED := $(MBED_ARM)

build/native/fpgate-proxy: $(SRC) $(HDR)
	@mkdir -p $(@D)
	$(CC) $(COMMON) $(HARDEN) -pie $(filter %.c,$^) $(LINK) -o $@

# Static against musl: no dependency on the board's glibc 2.20.
build/arm/fpgate-proxy: $(SRC) $(HDR)
	@mkdir -p $(@D)
	arm-linux-musleabihf-gcc $(COMMON) $(HARDEN) -static $(filter %.c,$^) $(LINK) -o $@

build/asan/fpgate-proxy: $(SRC) $(HDR)
	@mkdir -p $(@D)
	$(CC) $(COMMON) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined $(filter %.c,$^) $(LINK) -o $@

build/ubsan/fpgate-proxy: $(SRC) $(HDR)
	@mkdir -p $(@D)
	$(CC) $(COMMON) -O1 -g -fsanitize=undefined -fno-sanitize-recover=undefined $(filter %.c,$^) $(LINK) -o $@

test: asan
	bash tests/integration/test_http.sh build/asan/fpgate-proxy
	bash tests/integration/test_tls.sh build/asan/fpgate-proxy

clean:
	rm -rf build

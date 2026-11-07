# fpgate-proxy build. Targets: native, arm, asan, ubsan, test, clean
# Run inside the dev container: docker compose run --rm dev make <target>

PHR := third_party/picohttpparser
SRC := $(wildcard src/*.c) $(PHR)/picohttpparser.c
HDR := $(wildcard src/*.h) $(PHR)/picohttpparser.h Makefile

# Third-party static libraries, built into the dev image (docker/Dockerfile).
MBED_NATIVE := /opt/mbedtls/native
MBED_ARM    := /opt/mbedtls/arm
DEPS_NATIVE := /opt/deps/native
DEPS_ARM    := /opt/deps/arm

BUILD_EPOCH := $(shell date +%s)
WARN     := -Wall -Wextra -Werror
HARDEN   := -O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
CSTD     := -std=gnu11 -ffp-contract=off   # coefficients must not depend on FMA availability
COMMON    = $(CSTD) $(WARN) -Isrc -I$(PHR) -I$(MBED)/include -I$(DEPS)/include -DBUILD_EPOCH=$(BUILD_EPOCH)
LINK      = -L$(MBED)/lib -L$(DEPS)/lib -lmbedtls -lmbedx509 -lmbedcrypto -lpng -ljpeg -lwebpdecoder -lz -lm -lpthread

.PHONY: native arm asan ubsan test clean

native: build/native/fpgate-proxy
arm: build/arm/fpgate-proxy
asan: build/asan/fpgate-proxy
ubsan: build/ubsan/fpgate-proxy

build/native/fpgate-proxy: MBED := $(MBED_NATIVE)
build/asan/fpgate-proxy:   MBED := $(MBED_NATIVE)
build/ubsan/fpgate-proxy:  MBED := $(MBED_NATIVE)
build/arm/fpgate-proxy:    MBED := $(MBED_ARM)
build/native/fpgate-proxy: DEPS := $(DEPS_NATIVE)
build/asan/fpgate-proxy:   DEPS := $(DEPS_NATIVE)
build/ubsan/fpgate-proxy:  DEPS := $(DEPS_NATIVE)
build/arm/fpgate-proxy:    DEPS := $(DEPS_ARM)

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

# ---- unit test helpers (native, sanitized) ----------------------------------

build/proto_dump: tests/unit/proto_dump.c src/accel_proto.h
	@mkdir -p $(@D)
	$(CC) $(CSTD) $(WARN) -Isrc $< -o $@

build/test_accept: tests/unit/test_accept.c src/image_detect.c src/image_detect.h
	@mkdir -p $(@D)
	$(CC) $(CSTD) $(WARN) -fsanitize=address,undefined -Isrc tests/unit/test_accept.c src/image_detect.c -o $@

PREP_SRC := src/image_prep.c src/image_detect.c src/resample.c
build/prep_tool: DEPS := $(DEPS_NATIVE)
build/prep_tool: tests/unit/prep_tool.c $(PREP_SRC) $(wildcard src/*.h)
	@mkdir -p $(@D)
	$(CC) $(CSTD) $(WARN) -O1 -g -fsanitize=address,undefined -Isrc -I$(DEPS)/include \
	    tests/unit/prep_tool.c $(PREP_SRC) -L$(DEPS)/lib -lpng -ljpeg -lwebpdecoder -lz -lm -lpthread -o $@

test: asan build/proto_dump build/prep_tool build/test_accept
	python3 tests/mock_accel/test_mock_accel.py build/proto_dump
	build/test_accept
	python3 tests/unit/test_prep.py build/prep_tool
	bash tests/integration/test_http.sh build/asan/fpgate-proxy
	bash tests/integration/test_images.sh build/asan/fpgate-proxy
	bash tests/integration/test_workers.sh build/asan/fpgate-proxy
	bash tests/integration/test_tls.sh build/asan/fpgate-proxy

clean:
	rm -rf build

# Cross-built copy of prep_tool to time decoding on the board.
build/arm/prep_tool: DEPS := $(DEPS_ARM)
build/arm/prep_tool: tests/unit/prep_tool.c $(PREP_SRC) $(wildcard src/*.h)
	@mkdir -p $(@D)
	arm-linux-musleabihf-gcc $(CSTD) $(WARN) -O2 -static -Isrc -I$(DEPS)/include \
	    tests/unit/prep_tool.c $(PREP_SRC) -L$(DEPS)/lib -lpng -ljpeg -lwebpdecoder -lz -lm -lpthread -o $@

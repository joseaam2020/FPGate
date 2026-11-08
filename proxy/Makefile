# fpgate-proxy build. Targets: native, arm, asan, ubsan, test, clean
# Run inside the dev container: docker compose run --rm dev make <target>

PHR := third_party/picohttpparser
SRC := $(wildcard src/*.c) $(PHR)/picohttpparser.c build/gen/web_assets.c
HDR := $(wildcard src/*.h) $(PHR)/picohttpparser.h Makefile

# Third-party static libraries, built into the dev image (docker/Dockerfile).
MBED_NATIVE := /opt/mbedtls/native
MBED_ARM    := /opt/mbedtls/arm
DEPS_NATIVE := /opt/deps/native
DEPS_ARM    := /opt/deps/arm
SQL_NATIVE  := /opt/sqlite/native
SQL_ARM     := /opt/sqlite/arm

BUILD_EPOCH := $(shell date +%s)
WARN     := -Wall -Wextra -Werror
HARDEN   := -O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
CSTD     := -std=gnu11 -ffp-contract=off   # coefficients must not depend on FMA availability
COMMON    = $(CSTD) $(WARN) -Isrc -I$(PHR) -I$(MBED)/include -I$(DEPS)/include -I$(SQL)/include -DBUILD_EPOCH=$(BUILD_EPOCH)
LINK      = -L$(MBED)/lib -L$(DEPS)/lib -L$(SQL)/lib -lmbedtls -lmbedx509 -lmbedcrypto -lpng -ljpeg -lwebpdecoder -lsqlite3 -lz -lm -lpthread

.PHONY: native arm asan ubsan test clean

native: build/native/fpgate-proxy
arm: build/arm/fpgate-proxy
asan: build/asan/fpgate-proxy
ubsan: build/ubsan/fpgate-proxy

build/native/fpgate-proxy: MBED := $(MBED_NATIVE)
build/asan/fpgate-proxy:   MBED := $(MBED_NATIVE)
build/ubsan/fpgate-proxy:  MBED := $(MBED_NATIVE)
build/arm/fpgate-proxy:    MBED := $(MBED_ARM)
build/native/fpgate-proxy: SQL := $(SQL_NATIVE)
build/asan/fpgate-proxy:   SQL := $(SQL_NATIVE)
build/ubsan/fpgate-proxy:  SQL := $(SQL_NATIVE)
build/arm/fpgate-proxy:    SQL := $(SQL_ARM)
build/native/fpgate-proxy: DEPS := $(DEPS_NATIVE)
build/asan/fpgate-proxy:   DEPS := $(DEPS_NATIVE)
build/ubsan/fpgate-proxy:  DEPS := $(DEPS_NATIVE)
build/arm/fpgate-proxy:    DEPS := $(DEPS_ARM)

# The admin pages are compiled into the binary.
build/gen/web_assets.c: web/login.html web/dashboard.html web/certificate.html tools/embed_web.py
	@mkdir -p $(@D)
	python3 tools/embed_web.py web $@

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

test: asan build/proto_dump build/prep_tool build/test_accept build/mock_accel
	python3 tests/mock_accel/test_mock_accel.py build/proto_dump
	build/test_accept
	python3 tests/unit/test_prep.py build/prep_tool
	bash tests/integration/test_http.sh build/asan/fpgate-proxy
	bash tests/integration/test_images.sh build/asan/fpgate-proxy
	bash tests/integration/test_workers.sh build/asan/fpgate-proxy
	bash tests/integration/test_accel.sh build/asan/fpgate-proxy build/prep_tool build/mock_accel
	bash tests/integration/test_blocklog.sh build/asan/fpgate-proxy build/mock_accel
	bash tests/integration/test_admin.sh build/asan/fpgate-proxy build/mock_accel
	bash tests/integration/test_ca_expiry.sh build/asan/fpgate-proxy
	bash tests/integration/test_tls.sh build/asan/fpgate-proxy

clean:
	rm -rf build

# Cross-built copy of prep_tool to time decoding on the board.
build/arm/prep_tool: DEPS := $(DEPS_ARM)
build/arm/prep_tool: tests/unit/prep_tool.c $(PREP_SRC) $(wildcard src/*.h)
	@mkdir -p $(@D)
	arm-linux-musleabihf-gcc $(CSTD) $(WARN) -O2 -static -Isrc -I$(DEPS)/include \
	    tests/unit/prep_tool.c $(PREP_SRC) -L$(DEPS)/lib -lpng -ljpeg -lwebpdecoder -lz -lm -lpthread -o $@

# A tiny accelerator stand-in that runs on the board (the Python mock does not).
build/mock_accel: tests/mock_accel/mock_accel.c src/accel_proto.h
	@mkdir -p $(@D)
	$(CC) $(CSTD) $(WARN) -O2 -Isrc $< -o $@

build/arm/mock_accel: tests/mock_accel/mock_accel.c src/accel_proto.h
	@mkdir -p $(@D)
	arm-linux-musleabihf-gcc $(CSTD) $(WARN) -O2 -static -Isrc $< -o $@

# ---- fuzzing (libFuzzer, needs clang) ------------------------------------------------------
# make fuzz               builds every target into build/fuzz/
# make fuzz-run SECS=300  builds them and runs each for SECS seconds from generated seeds

FUZZCC    := clang
FUZZ_TARGETS := http_stream detect image toml creds admin
FUZZ_FLAGS = -std=gnu11 -O1 -g -fno-omit-frame-pointer -ffp-contract=off \
    -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=undefined \
    -Isrc -I$(PHR) -I$(MBED_NATIVE)/include -I$(DEPS_NATIVE)/include -I$(SQL_NATIVE)/include
FUZZ_LIBS_IMG = -L$(DEPS_NATIVE)/lib -lpng -ljpeg -lwebpdecoder -lz -lm
ALL_BUT_MAIN  = $(filter-out src/main.c src/admin.c,$(wildcard src/*.c)) $(PHR)/picohttpparser.c build/gen/web_assets.c

.PHONY: fuzz fuzz-run
fuzz: $(addprefix build/fuzz/,$(FUZZ_TARGETS))

build/fuzz/http_stream: tests/fuzz/fuzz_http_stream.c src/http_stream.c src/http_stream.h $(PHR)/picohttpparser.c
	@mkdir -p $(@D)
	$(FUZZCC) $(FUZZ_FLAGS) tests/fuzz/fuzz_http_stream.c src/http_stream.c $(PHR)/picohttpparser.c -o $@

build/fuzz/detect: tests/fuzz/fuzz_detect.c src/image_detect.c src/image_detect.h
	@mkdir -p $(@D)
	$(FUZZCC) $(FUZZ_FLAGS) tests/fuzz/fuzz_detect.c src/image_detect.c -o $@

build/fuzz/image: tests/fuzz/fuzz_image.c src/image_prep.c src/image_detect.c src/resample.c $(wildcard src/*.h)
	@mkdir -p $(@D)
	$(FUZZCC) $(FUZZ_FLAGS) tests/fuzz/fuzz_image.c src/image_prep.c src/image_detect.c src/resample.c $(FUZZ_LIBS_IMG) -o $@

build/fuzz/toml: tests/fuzz/fuzz_toml.c src/settings.c src/settings.h
	@mkdir -p $(@D)
	$(FUZZCC) $(FUZZ_FLAGS) tests/fuzz/fuzz_toml.c src/settings.c -lm -o $@

build/fuzz/creds: tests/fuzz/fuzz_creds.c src/auth.c src/auth.h
	@mkdir -p $(@D)
	$(FUZZCC) $(FUZZ_FLAGS) tests/fuzz/fuzz_creds.c src/auth.c -L$(MBED_NATIVE)/lib -lmbedcrypto -o $@

# The admin target compiles src/admin.c as part of the harness (it reaches static functions).
build/fuzz/admin: tests/fuzz/fuzz_admin.c src/admin.c $(wildcard src/*.h) $(ALL_BUT_MAIN)
	@mkdir -p $(@D)
	$(FUZZCC) $(FUZZ_FLAGS) tests/fuzz/fuzz_admin.c $(ALL_BUT_MAIN) \
	    -L$(MBED_NATIVE)/lib -L$(DEPS_NATIVE)/lib -L$(SQL_NATIVE)/lib \
	    -lmbedtls -lmbedx509 -lmbedcrypto -lpng -ljpeg -lwebpdecoder -lsqlite3 -lz -lm -lpthread -o $@

fuzz-run: fuzz
	bash tests/fuzz/run.sh $(or $(SECS),60)

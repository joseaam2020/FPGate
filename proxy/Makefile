# fpgate-proxy build. Targets: native, arm, asan, ubsan, test, clean
# Run inside the dev container: docker compose run --rm dev make <target>

PHR := third_party/picohttpparser
SRC := $(wildcard src/*.c) $(PHR)/picohttpparser.c
BIN := build/fpgate-proxy

WARN     := -Wall -Wextra -Werror
HARDEN   := -O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2 -fPIE
CSTD     := -std=gnu11
COMMON   := $(CSTD) $(WARN) -Isrc -I$(PHR)

.PHONY: native arm asan ubsan test clean

native: build/native/fpgate-proxy

arm: build/arm/fpgate-proxy

asan: build/asan/fpgate-proxy

ubsan: build/ubsan/fpgate-proxy

build/native/fpgate-proxy: $(SRC)
	@mkdir -p $(@D)
	$(CC) $(COMMON) $(HARDEN) -pie $^ -o $@

# Static against musl: no dependency on the board's glibc 2.20.
build/arm/fpgate-proxy: $(SRC)
	@mkdir -p $(@D)
	arm-linux-musleabihf-gcc $(COMMON) $(HARDEN) -static $^ -o $@

build/asan/fpgate-proxy: $(SRC)
	@mkdir -p $(@D)
	$(CC) $(COMMON) -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined $^ -o $@

build/ubsan/fpgate-proxy: $(SRC)
	@mkdir -p $(@D)
	$(CC) $(COMMON) -O1 -g -fsanitize=undefined -fno-sanitize-recover=undefined $^ -o $@

test: asan
	bash tests/integration/test_http.sh build/asan/fpgate-proxy

clean:
	rm -rf build

# fpgate-accel (HPS side of the accelerator). Build inside the proxy's dev image:
#   docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make native
#   docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make arm
# The wire format lives in proxy/src/accel_proto.h (source of truth); it is not copied here.

SRC      := src/main.c src/model.c src/ops.c src/pkg.c src/sha256.c src/fpga_pack.c src/accel_drv.c
PROTO    := ../../proxy/src
CSTD     := -std=gnu11 -Wall -Wextra -Wno-unused-parameter
# no fused multiply-add and no fast-math: the float parts must round like float32 operations
FP       := -ffp-contract=off

native: build/native/fpgate-accel
arm: build/arm/fpgate-accel

build/native/fpgate-accel: $(SRC) $(wildcard src/*.h)
	mkdir -p build/native
	gcc $(CSTD) -O3 -march=native $(FP) -fopenmp -I$(PROTO) $(SRC) -lm -o $@

build/arm/fpgate-accel: $(SRC) $(wildcard src/*.h)
	mkdir -p build/arm
	arm-linux-musleabihf-gcc $(CSTD) -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard $(FP) -static -I$(PROTO) $(SRC) -lm -o $@

build/native/test_pack: tests/test_pack.c src/fpga_pack.c src/pkg.c src/sha256.c $(wildcard src/*.h)
	mkdir -p build/native
	gcc $(CSTD) -O2 tests/test_pack.c src/fpga_pack.c src/pkg.c src/sha256.c -o $@

clean:
	rm -rf build

.PHONY: native arm clean

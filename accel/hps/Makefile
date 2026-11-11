# fpgate-accel (HPS side of the accelerator). Build inside the proxy's dev image:
#   docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make native
#   docker run --rm -v "${PWD}:/repo" -w /repo/accel/hps fpgate-proxy-dev make arm
# The wire format lives in proxy/src/accel_proto.h (source of truth); it is not copied here.

SRC      := src/main.c src/model.c src/ops.c src/pkg.c src/sha256.c src/fpga_pack.c src/accel_drv.c src/accel_sim.c
PROTO    := ../../proxy/src
CSTD     := -std=gnu11 -Wall -Wextra -Wno-unused-parameter
# no fused multiply-add and no fast-math: the float parts must round like float32 operations
FP       := -ffp-contract=off

native: build/native/fpgate-accel
arm: build/arm/fpgate-accel

build/native/fpgate-accel: $(SRC) $(wildcard src/*.h)
	mkdir -p build/native
	gcc $(CSTD) -O3 -march=native $(FP) -fopenmp -pthread -I$(PROTO) $(SRC) -lm -o $@

build/arm/fpgate-accel: $(SRC) $(wildcard src/*.h)
	mkdir -p build/arm
	arm-linux-musleabihf-gcc $(CSTD) -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard $(FP) -fopenmp -static -I$(PROTO) $(SRC) -lm -o $@

build/native/test_pack: tests/test_pack.c src/fpga_pack.c src/pkg.c src/sha256.c $(wildcard src/*.h)
	mkdir -p build/native
	gcc $(CSTD) -O2 tests/test_pack.c src/fpga_pack.c src/pkg.c src/sha256.c -o $@

# Kernel tests: each one compares an optimised kernel with a plain reference and must print 0 mismatches.
# They run natively and, for the ARM/NEON build, under qemu-arm (available in the dev image).
KTESTS   := test_quant test_gemm test_ln test_pack_rand
ARMCC    := arm-linux-musleabihf-gcc $(CSTD) -O3 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard $(FP) -fopenmp -static -Isrc -I$(PROTO)
NATCC    := gcc $(CSTD) -O3 -march=native $(FP) -fopenmp -Isrc -I$(PROTO)

test-kernels: $(wildcard src/*.c) $(wildcard tests/*.c)
	mkdir -p build/native build/arm
	set -e; for t in $(KTESTS); do \
	  case $$t in test_pack_rand) S="src/fpga_pack.c src/pkg.c src/sha256.c";; *) S="src/ops.c src/pkg.c src/sha256.c";; esac; \
	  $(NATCC) tests/$$t.c $$S -lm -o build/native/$$t; echo "== native $$t"; build/native/$$t; \
	  $(ARMCC) tests/$$t.c $$S -lm -o build/arm/$$t; echo "== arm $$t"; qemu-arm build/arm/$$t; \
	done

# Overlap tests: the software engine (--sim) runs the linear layers on a worker thread; every layer is
# compared with the CPU result (--check) for several chunk sizes. Must print "identical to the CPU" only.
test-sim: build/native/fpgate-accel
	set -e; for o in "--no-overlap" "" "--chunk 64" "--chunk 1000"; do \
	  echo "== sim $$o"; \
	  build/native/fpgate-accel --selftest 1 --pkg $(PKG) --index pkg/e1_r7.idx --sim --check $$o 2>&1 \
	    | grep -E 'FPGA:|\*\*\*|decisions'; \
	done
PKG      ?= /repo/model/quant/release/e1_r7

clean:
	rm -rf build

.PHONY: native arm test-kernels test-sim clean

# diffusion.c (hku, 16x16 sprite DDPM) on Hwacha: clang -> hwacha-cc -> esp-tools gcc -> Spike / RTL
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CLANG   := $(HOME)/miniforge3/bin/clang
HWCC    := ../../build/hwacha-cc
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
CLFLAGS := -x cl -cl-std=CL1.2 -Xclang -finclude-default-header -Xclang -cl-ext=+cl_khr_fp64 --target=riscv64-unknown-elf \
           -march=rv64gc -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S
EXTRA   ?=
CFLAGS  := $(EXTRA) -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha
UPSTREAM := $(ROOT)/diffusion.c
STEPS   ?= 200
CHECK   ?= 2

all: diffusion.riscv diffusion_x86

# the weights ship with the upstream repository (weights/ckpt.bin, 5.9 MB)
ckpt.bin:
	test -d $(UPSTREAM) || git clone --depth 1 https://github.com/hku/diffusion.c $(UPSTREAM)
	cp $(UPSTREAM)/weights/ckpt.bin $@

diffusion.ll: diffusion.cl
	$(CLANG) $(CLFLAGS) $< -o $@
diffusion.s: diffusion.ll $(HWCC)
	$(HWCC) $< -o $@
diffusion.riscv: diffusion_main.c diffusion.s ckpt.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DSTEPS=$(STEPS) -DCHECK_STEPS=$(CHECK) -o $@ diffusion_main.c diffusion.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
# RTL variant: one step, no scalar reference
diffusion_rtl.riscv: diffusion_main.c diffusion.s ckpt.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DSTEPS=1 -DCHECK_STEPS=0 -o $@ diffusion_main.c diffusion.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
# x86 reference (scalar diffusion.c forward only, same noise): the expected image
diffusion_x86: diffusion_main.c
	gcc -O2 -DX86 -DSTEPS=$(STEPS) -o $@ $< -lm

spike: diffusion.riscv
	$(SPIKE) $<
clean:
	rm -f *.ll *.s *.riscv diffusion_x86
.PRECIOUS: diffusion.ll diffusion.s
# one conv3x3 layer, scalar vs Hwacha (the scalar full step is too slow for the RTL)
diffusion_bench.riscv: diffusion_main.c diffusion.s ckpt.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DLAYER_BENCH -o $@ diffusion_main.c diffusion.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)

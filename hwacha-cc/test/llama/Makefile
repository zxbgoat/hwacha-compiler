# llama2.c (stories260K) on Hwacha: clang -> hwacha-cc -> esp-tools gcc -> Spike / RTL
ROOT    := ../../..
CLFLAGS_EXTRA := -I.
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CLANG   := $(HOME)/miniforge3/bin/clang
HWCC    := ../../build/hwacha-cc
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
CLFLAGS := -x cl -cl-std=CL1.2 -Xclang -finclude-default-header -Xclang -cl-ext=+cl_khr_fp64 --target=riscv64-unknown-elf \
           -march=rv64gc -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha
SIM     := $(ROOT)/chipyard/sims/verilator/simulator-chipyard.harness-HwachaRocketConfig

APPS := expk llama
all: $(addsuffix .riscv,$(APPS))
spike: $(addsuffix .spike,$(APPS))

%.ll: %.cl
	$(CLANG) $(CLFLAGS) -I. $< -o $@
%.s: %.ll $(HWCC)
	$(HWCC) $< -o $@
%.riscv: %_main.c %.s $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
%.spike: %.riscv
	$(SPIKE) $<
%.rtl: %.riscv
	$(SIM) +permissive +max-cycles=2000000000 +permissive-off $< 2>&1 | grep -v "^\[UART\]"
clean:
	rm -f *.ll *.s *.riscv
.PRECIOUS: %.ll %.s
# RTL variant: 4 tokens, no teacher-forced check (Verilator runs at ~2.7k cycles/s)
llama_rtl.riscv: llama_main.c llama.s $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DSTEPS=4 -DNO_CHECK -o $@ $^ $(LDFLAGS)

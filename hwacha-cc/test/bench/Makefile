# RTL/Spike benchmark: hwacha-cc output vs hand-written asm vs scalar C, plus ablations
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CLANG   := $(HOME)/miniforge3/bin/clang
HWCC    := ../../build/hwacha-cc
# esp-tools GCC 9.2 from the Chipyard conda env understands -march=rv64gcxhwacha directly
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
CLFLAGS := -x cl -cl-std=CL1.2 -Xclang -finclude-default-header --target=riscv64-unknown-elf -march=rv64gc -O2 \
           -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S
CFLAGS  := -DPREALLOCATE=1 -DHAVE_HAND -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -march=rv64gcxhwacha -mabi=lp64d -I$(COMMON) -I$(ROOT)/esp-tests/env -I$(COMMON)/../vec-daxpy
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha
N       ?= 4096

VARIANTS := bench bench-nov32 bench-noskip bench-nocoalesce
all: $(addsuffix -n$(N).riscv,$(VARIANTS))

bench.ll: bench.cl
	$(CLANG) $(CLFLAGS) $< -o $@
bench.s: bench.ll $(HWCC)
	$(HWCC) $< -o $@
bench-nov32.s: bench.ll $(HWCC)
	$(HWCC) --no-v32 $< -o $@
bench-noskip.s: bench.ll $(HWCC)
	$(HWCC) --no-skip $< -o $@
bench-nocoalesce.s: bench.ll $(HWCC)
	$(HWCC) --no-coalesce $< -o $@
# binaries are named <variant>-n<N>.riscv so different sizes never collide
%-n$(N).riscv: %.s bench_main.c hand.S $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DN=$(N) -o $@ bench_main.c $< hand.S $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
%.spike: %.riscv
	$(SPIKE) $<
clean:
	rm -f *.ll *.s *.riscv
.PRECIOUS: %.s

# End-to-end: OpenCL -> clang -> hwacha-cc -> esp binutils -> Spike
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
CLANG   := $(HOME)/miniforge3/bin/clang
HWCC    := ../../build/hwacha-cc
CC      := riscv64-unknown-elf-gcc-xhwacha
CLFLAGS := -x cl -cl-std=CL1.2 -Xclang -finclude-default-header --target=riscv64-unknown-elf -march=rv64gc -O2 \
           -Xclang -cl-ext=+cl_khr_fp64 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -march=rv64gcxhwacha -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha

TESTS := spec gather cf lc local misc flag pfmin red

all: $(addsuffix .riscv,$(TESTS))
run: $(addsuffix .run,$(TESTS))

%.ll: %.cl
	$(CLANG) $(CLFLAGS) $< -o $@
%.s: %.ll $(HWCC)
	$(HWCC) $< -o $@
%.riscv: %_main.c %.s $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
%.run: %.riscv
	$(SPIKE) $<

clean:
	rm -f *.ll *.s *.riscv
.PRECIOUS: %.ll %.s
# scalar-FP probe: uniform FP ops kept in vs registers (the RTL's shared-FPU path)
sfp.s: sfp.ll $(HWCC)
	$(HWCC) $< -o $@ --scalar-fp

# MLIR gpu-dialect entry: .mlir -> hwacha-mlir (kernels .ll + host .ll) -> hwacha-cc -> Spike
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
HWMLIR  := ../../build/hwacha-mlir
HWCC    := ../../build/hwacha-cc
CC      := riscv64-unknown-elf-gcc-xhwacha
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -march=rv64gcxhwacha -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha

KERNELS := saxpy_gpu saxpy_linalg gs wg ex mm mmt

all: mlir.riscv
run: mlir.riscv
	$(SPIKE) $<

# per-file settings
saxpy_linalg.ll: MEMREF = --memref desc
mmt.s: HWCCFLAGS = --assume-noalias --fp-contract

%.ll %.host.ll: %.mlir $(HWMLIR)
	$(HWMLIR) $< -o $*.ll --host $*.host.ll $(MEMREF)
%.s: %.ll %.host.ll $(HWCC)
	$(HWCC) $< --host $*.host.ll $(HWCCFLAGS) -o $@
mlir.riscv: mlir_main.c $(addsuffix .s,$(KERNELS)) $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.ll *.s *.riscv
.PRECIOUS: %.ll %.host.ll %.s

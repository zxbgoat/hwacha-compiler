# MLIR gpu-dialect entry: .mlir -> (mlir-opt/mlir-translate via tools/mlir-to-ll.sh) -> hwacha-cc -> Spike
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
TOOLS   := ../../tools
HWCC    := ../../build/hwacha-cc
CC      := riscv64-unknown-elf-gcc-xhwacha
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -march=rv64gcxhwacha -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha

KERNELS := saxpy_gpu saxpy_linalg wg ex mm

all: mlir.riscv
run: mlir.riscv
	$(SPIKE) $<

# per-kernel front-end settings (see tools/mlir-to-ll.sh)
saxpy_linalg.ll: export MEMREF_CONV = desc
mm.ll: export COLLAPSE = 0,1

%.ll: %.mlir $(TOOLS)/mlir-to-ll.sh $(TOOLS)/mlir-extract-kernels.py
	$(TOOLS)/mlir-to-ll.sh $< $@
%.s: %.ll $(HWCC)
	$(HWCC) $< -o $@
mlir.riscv: mlir_main.c $(addsuffix .s,$(KERNELS)) $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f *.ll *.ll.*.mlir *.s *.riscv
.PRECIOUS: %.ll %.s

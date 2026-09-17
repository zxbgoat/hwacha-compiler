# PyTorch -> torch-mlir (linalg on tensors) -> bufferize -> hwacha-mlir -> hwacha-cc -> Spike / RTL
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
MLIROPT := $(HOME)/miniforge3/bin/mlir-opt
PYTHON  := /tmp/tmenv/bin/python    # venv with the torch-mlir wheel on top of the system torch
HWMLIR  := ../../build/hwacha-mlir
HWCC    := ../../build/hwacha-cc
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha
HWMLIRFLAGS ?=
BUFFERIZE := builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},func.func(buffer-deallocation-simplification),convert-bufferization-to-memref,canonicalize,cse)

all: aestuans.riscv

aestuans.mlir aestuans_check.bin: export_aestuans.py
	$(PYTHON) export_aestuans.py /tmp/mnist-diffusion aestuans.mlir aestuans_check.bin
%_memref.mlir: %.mlir
	$(MLIROPT) $< --pass-pipeline='$(BUFFERIZE)' -o $@
hwlib.ll: hwlib.cl
	$(HOME)/miniforge3/bin/clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header --target=riscv64-unknown-elf -march=rv64gc -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S $< -o $@
hwlib.s: hwlib.ll $(HWCC)
	$(HWCC) $< --assume-noalias -o $@
%.ll %.host.ll: %_memref.mlir $(HWMLIR)
	$(HWMLIR) $< -o $*.ll --host $*.host.ll $(HWMLIRFLAGS)
%.s: %.ll %.host.ll $(HWCC)
	$(HWCC) $< --host $*.host.ll --assume-noalias -o $@
%.riscv: torch_main.c %.s hwlib.s %_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DCHECK_FILE='"$*_check.bin"' -o $@ torch_main.c $*.s hwlib.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
%.spike: %.riscv
	$(SPIKE) $<
clean:
	rm -f *.ll *.s *.riscv *_memref.mlir
.PRECIOUS: %.mlir %_memref.mlir %.ll %.host.ll %.s

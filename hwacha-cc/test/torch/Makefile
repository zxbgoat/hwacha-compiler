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
BUFFERIZE := builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},canonicalize,cse,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref)

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
# ONNX models: <name>.onnx -> export_onnx.py (torch-mlir ONNX importer + onnxruntime reference) -> same pipeline
%_onnx.mlir %_onnx_check.bin: %.onnx export_onnx.py
	PATH=/tmp/tmenv/bin:$$PATH $(PYTHON) export_onnx.py $< $*_onnx.mlir $*_onnx_check.bin
%_onnx.riscv: onnx_main.c %_onnx.s hwlib.s %_onnx_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DCHECK_FILE='"$*_onnx_check.bin"' -DENTRY=$(ENTRY) -DNARGS=$(NARGS) -o $@ onnx_main.c $*_onnx.s hwlib.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
ENTRY ?= main_graph
NARGS ?= 1
aestuans.onnx: aestuans_onnx.py
	$(PYTHON) aestuans_onnx.py /tmp/mnist-diffusion $@
%.spike: %.riscv
	$(SPIKE) $<
clean:
	rm -f *.ll *.s *.riscv *_memref.mlir *_onnx.mlir *.torch.mlir
.PRECIOUS: %.mlir %_memref.mlir %.ll %.host.ll %.s

# torchvision classification models: export_tv.py + torch-mlir + hwacha-mlir + hwacha-cc
TVHW ?= 32
%_tv.mlir %_tv_check.bin: export_tv.py
	$(PYTHON) export_tv.py $* $*_tv.mlir $*_tv_check.bin $(TVHW)
%_tv_memref.mlir: %_tv.mlir
	$(MLIROPT) $< --pass-pipeline='$(BUFFERIZE)' -o $@
%_tv.ll %_tv.host.ll: %_tv_memref.mlir $(HWMLIR)
	$(HWMLIR) $< -o $*_tv.ll --host $*_tv.host.ll $(HWMLIRFLAGS)
%_tv.s: %_tv.ll %_tv.host.ll $(HWCC)
	$(HWCC) $< --host $*_tv.host.ll --assume-noalias -o $@
%_tv.riscv: tv_main.c %_tv.s hwlib.s %_tv_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DCHECK_FILE='"$*_tv_check.bin"' -DMODEL='"$*"' -o $@ tv_main.c $*_tv.s hwlib.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)

# minimal DiT (Diffusion Transformer): patchify + self-attention + adaLN + MLP, one forward vs PyTorch.
# DITCFG = "dim depth heads C H P" (default 64 2 4 4 8 2). Intermediate steps reuse the %_memref/%.ll/%.s
# rules; dit.riscv links its own host (dit_main.c) which does the trivial unpatchify.
DITCFG ?=
dit.mlir dit_check.bin: export_dit.py
	$(PYTHON) export_dit.py dit.mlir dit_check.bin $(DITCFG)
dit.riscv: dit_main.c dit.s hwlib.s dit_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DCHECK_FILE='"dit_check.bin"' -o $@ dit_main.c dit.s hwlib.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)

# DiT-S/2 at the real config (latent 4x32x32, patch 2 -> 256 tokens, depth 12, hidden 384, 6 heads).
# DITS2CFG = "depth hidden heads" (default 12 384 6). Sinusoidal timestep embedding precomputed on host.
DITS2CFG ?=
dit_s2.mlir dit_s2_check.bin: export_dit_s2.py export_dit.py
	$(PYTHON) export_dit_s2.py dit_s2.mlir dit_s2_check.bin $(DITS2CFG)
# strip the ~33M-2GB weights to a binary blob + .incbin .S so the .ll/.s stay small (memory-bounded for
# DiT-L/XL); dit_s2_weights.bin.S is emitted next to the .ll and linked in, reading dit_s2_weights.bin.
dit_s2.ll dit_s2.host.ll dit_s2_weights.bin.S: dit_s2_memref.mlir $(HWMLIR)
	$(HWMLIR) $< -o dit_s2.ll --host dit_s2.host.ll --weights-bin dit_s2_weights.bin $(HWMLIRFLAGS)
dit_s2.riscv: dit_s2_main.c dit_s2.s hwlib.s dit_s2_weights.bin.S dit_s2_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DCHECK_FILE='"dit_s2_check.bin"' -o $@ dit_s2_main.c dit_s2.s hwlib.s dit_s2_weights.bin.S $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)

# Three MNIST diffusion models on Hwacha: clang -> hwacha-cc -> esp-tools gcc -> Spike / RTL
#   bot66     bot66/MNISTDiffusion            (depthwise-conv UNet, 1000 steps)   weights trained here (export.py)
#   teapearce TeaPearce/Conditional_Diffusion_MNIST (ContextUnet 128, 400 steps)  pretrained_model.zip from the repo
#   aestuans  aestuans/mnist-diffusion        (ResNet UNet, 500 steps)            weights trained here (export.py)
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
MODELS  := bot66 teapearce aestuans
STEPS   ?= 100000
CHECK   ?= 1

all: $(addsuffix .riscv,$(MODELS))
x86: $(addprefix x86_,$(MODELS))

mnist.ll: mnist.cl
	$(CLANG) $(CLFLAGS) $< -o $@
mnist.s: mnist.ll $(HWCC)
	$(HWCC) $< --assume-noalias -o $@
define MODEL_RULES
$(1).riscv: mnist_main.c mnist.s $(1).bin $(1)_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DMODEL_$(shell echo $(1) | tr a-z A-Z) -DSTEPS=$(STEPS) -DCHECK_STEPS=$(CHECK) -o $$@ mnist_main.c mnist.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
$(1)_rtl.riscv: mnist_main.c mnist.s $(1).bin $(1)_check.bin $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -DMODEL_$(shell echo $(1) | tr a-z A-Z) -DSTEPS=1 -DCHECK_STEPS=0 -o $$@ mnist_main.c mnist.s $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
x86_$(1): mnist_main.c
	gcc -O2 -DX86 -DMODEL_$(shell echo $(1) | tr a-z A-Z) -DSTEPS=$(STEPS) -o $$@ $$< -lm
$(1).spike: $(1).riscv
	$(SPIKE) $$<
endef
$(foreach m,$(MODELS),$(eval $(call MODEL_RULES,$(m))))

# weights: export.py needs the three upstream repos (cloned under /tmp by default) and, for bot66 and
# aestuans, their training runs (see NOTES.md); teapearce uses the pretrained_model.zip of its repo
%.bin %_check.bin:
	python3 export.py /tmp

clean:
	rm -f *.ll *.s *.riscv x86_*
.PRECIOUS: mnist.ll mnist.s

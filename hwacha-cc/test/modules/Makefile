# Basic nn layers, each PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc -> Spike, checked vs PyTorch.
ROOT    := ../../..
COMMON  := $(ROOT)/esp-tests/benchmarks/common
MLIROPT := $(HOME)/miniforge3/bin/mlir-opt
PYTHON  := /tmp/tmenv/bin/python
HWMLIR  := ../../build/hwacha-mlir
HWCC    := ../../build/hwacha-cc
CC      := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
CFLAGS  := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
           -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE   := spike --isa=rv64gc --extension=hwacha
# collapse every parallel dim into the lane dimension: for these small single-layer tests it vectorizes
# the pad/copy/reduction loops that the default lane-innermost mapping would unroll and overflow vs on.
HWMLIRFLAGS ?= --collapse-all
LAYERS := linear conv3x3 conv1x1 conv_s2 conv5x5 convtranspose2 dwconv3x3 maxpool2 avgpool2 \
          adaptiveavgpool batchnorm2d layernorm groupnorm relu leakyrelu gelu silu sigmoid tanh hardswish \
          hardsigmoid elu softmax flatten upsample upsample_bilinear pixelshuffle attention mlp \
          softplus mish hardtanh relu6 selu celu softsign tanhshrink softshrink hardshrink logsigmoid \
          logsoftmax prelu glu threshold normalize instancenorm lrn pad sdpa unfold interpolate \
          conv3d conv1d convtranspose1d convtranspose3d maxpool1d maxpool3d avgpool1d avgpool3d \
          adaptiveavgpool1d adaptiveavgpool3d adaptivemaxpool1d adaptivemaxpool2d lppool1d lppool2d lppool3d \
          reflectionpad2d zeropad2d constantpad2d replicationpad2d circularpad2d zeropad1d batchnorm1d batchnorm3d \
          instancenorm1d instancenorm3d rmsnorm lazybatchnorm2d rrelu softmin softmax2d identity dropout dropout2d alphadropout \
          unflatten pixelunshuffle channelshuffle upsamplingnearest2d upsamplingbilinear2d \
          rnn lstm gru rnncell transformerencoderlayer transformerencoder transformerdecoderlayer \
          transformerdecoder transformer groupconv groupconv_s2
BUFFERIZE := builtin.module(one-shot-bufferize{bufferize-function-boundaries=1 function-boundary-type-conversion=identity-layout-map},canonicalize,cse,ownership-based-buffer-deallocation,canonicalize,buffer-deallocation-simplification,bufferization-lower-deallocations,cse,canonicalize,convert-bufferization-to-memref)

# the conv/pool/norm library kernels live with the torchvision tests; reuse that prebuilt assembly
HWLIB := ../torch/hwlib.s
$(HWLIB):
	$(MAKE) -C ../torch hwlib.s

# one recipe per layer: export -> bufferize -> hwacha-mlir -> hwacha-cc -> link
%.riscv: export_module.py mod_main.c $(HWLIB) $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(PYTHON) export_module.py $* $*.mlir $*_check.bin
	$(MLIROPT) $*.mlir --pass-pipeline='$(BUFFERIZE)' -o $*_memref.mlir
	$(HWMLIR) $*_memref.mlir -o $*.ll --host $*.host.ll $(HWMLIRFLAGS)
	$(HWCC) $*.ll --host $*.host.ll --assume-noalias -o $*.s
	$(CC) $(CFLAGS) -DCHECK_FILE='"$*_check.bin"' -DMODEL='"$*"' -o $@ mod_main.c $*.s $(HWLIB) $(COMMON)/crt.S $(COMMON)/syscalls.c $(LDFLAGS)
%.spike: %.riscv
	$(SPIKE) $*.riscv
# build and run every layer, one PASS/FAIL line each
run:
	@pass=0; fail=0; for L in $(LAYERS); do \
	  if $(MAKE) -s $$L.riscv >/tmp/mod_$$L.log 2>&1; then \
	    r=$$($(SPIKE) $$L.riscv 2>&1 | grep -oE "$$L (PASS|FAIL)" | tail -1); \
	    echo "$${r:-$$L NO-OUTPUT}"; case "$$r" in *PASS) pass=$$((pass+1));; *) fail=$$((fail+1));; esac; \
	  else echo "$$L BUILD-FAIL ($$(grep -ioE 'out of Hwacha[^\"]*|unsupported call[^\"]*' /tmp/mod_$$L.log|head -1))"; fail=$$((fail+1)); fi; \
	done; echo "---- $$pass passed, $$fail failed of $$(echo $(LAYERS)|wc -w)"
clean:
	rm -f *.ll *.s *.riscv *_memref.mlir *.mlir *_check.bin
.PRECIOUS: %.mlir %_check.bin %_memref.mlir %.ll %.host.ll %.s %.riscv

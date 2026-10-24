# CUDA entry: clang's CUDA front end (no nvcc, no NVIDIA headers) -> hwacha-cc -> esp-tools gcc -> Spike / RTL
#   device: clang -x cuda --cuda-device-only  -> NVVM-flavoured LLVM IR, the same shape hwacha-cc's GPU-dialect entry takes
#   host:   clang -x cuda --cuda-host-only -O0 for riscv64, launch stubs rewritten by hwacha-cc --host
ROOT     := ../../..
COMMON   := $(ROOT)/esp-tests/benchmarks/common
CLANG    := $(HOME)/miniforge3/bin/clang
CLANGINC := $(dir $(CLANG))../lib/clang/23/include
HWCC     := ../../build/hwacha-cc
CC       := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc
SYSROOT  := $(shell $(CC) -print-sysroot)
CXXINC   := $(SYSROOT)/include/c++/$(shell $(CC) -dumpversion)
CUDAFLAGS := -x cuda -nocudainc -nocudalib -Xclang -target-sdk-version=12.0 --cuda-gpu-arch=sm_50 -Iinclude -I$(CLANGINC) -std=c++14 -fno-exceptions -fno-rtti
DEVFLAGS := $(CUDAFLAGS) --cuda-device-only -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S
HOSTFLAGS := $(CUDAFLAGS) --cuda-host-only -O0 --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d \
            -nostdinc -isystem $(CXXINC) -isystem $(CXXINC)/riscv64-unknown-elf \
            -isystem $(CLANGINC) -isystem $(SYSROOT)/include -D__riscv_xlen=64 -emit-llvm -S
CFLAGS   := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
            -fno-tree-loop-distribute-patterns -fno-math-errno -march=rv64gcxhwacha -mabi=lp64d -I$(COMMON) -I$(ROOT)/esp-tests/env
LDFLAGS  := -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE    := spike --isa=rv64gc --extension=hwacha
SIM      := $(ROOT)/chipyard/sims/verilator/simulator-chipyard.harness-HwachaRocketConfig

APPS := saxpy shared nn pathfinder hotspot bfs nw kmeans atomics
all: $(addsuffix .riscv,$(APPS))
run: $(addsuffix .spike,$(APPS))

%.dev.ll: %.cu include/cuda_runtime.h
	$(CLANG) $(DEVFLAGS) $< -o $@
%.host.ll: %.cu include/cuda_runtime.h
	$(CLANG) $(HOSTFLAGS) $< -o $@
%.s: %.dev.ll %.host.ll $(HWCC)
	$(HWCC) $< --host $*.host.ll $(HWCCFLAGS) -o $@
%.riscv: %.s hwcuda_rt.c $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
%.spike: %.riscv
	$(SPIKE) $<
# +loadmem: hwcuda_rt.c's arena puts the "device" heap in .bss (HWCUDA_ARENA, 4MB); loading that over
# TSI takes hours, the SimDRAM backdoor takes a second
%.rtl: %.riscv
	$(SIM) +permissive +max-cycles=2000000000 +loadmem=$(abspath $<) +permissive-off $< 2>&1 | grep --line-buffered -v "^\[UART\] UART0"
clean:
	rm -f *.ll *.s *.riscv
.PRECIOUS: %.dev.ll %.host.ll %.s

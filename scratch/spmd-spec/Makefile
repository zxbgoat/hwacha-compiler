COMMON := ../../esp-tests/benchmarks/common
CC     := riscv64-unknown-elf-gcc-xhwacha
CFLAGS := -DPREALLOCATE=1 -mcmodel=medany -static -std=gnu99 -O2 -fno-common -fno-builtin-printf \
          -fno-tree-loop-distribute-patterns -march=rv64gcxhwacha -I$(COMMON) -I../../esp-tests/env
LDFLAGS:= -static -nostdlib -nostartfiles -lm -lgcc -T $(COMMON)/test.ld
SPIKE  := spike --isa=rv64gc --extension=hwacha

spmd.riscv: main.c kernels_hwacha.S $(COMMON)/crt.S $(COMMON)/syscalls.c
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

run: spmd.riscv
	$(SPIKE) $<

dump: spmd.riscv
	riscv64-unknown-elf-objdump -d $< > spmd.dump

clean:
	rm -f spmd.riscv spmd.dump

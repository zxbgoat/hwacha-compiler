# hwacha-compiler

Research on the Hwacha vector-fetch architecture (UC Berkeley) through a compiler.
`hwacha-cc` compiles OpenCL C kernels to Hwacha vector-fetch code: the divergent part of a kernel
becomes vector-fetch (`vf`) blocks, the uniform control flow runs on the scalar control thread, and the
result is verified on the Spike ISA simulator and measured on the Chipyard Verilator RTL of
`HwachaRocketConfig`.

The full engineering log (design decisions, bugs found in the RTL and in the compiler, every
measurement, dead ends) is `NOTES.md` (Chinese). This file is the practical guide.

## Contents

| path | what |
|---|---|
| `hwacha-cc/src` | the compiler: `Analysis.{h,cpp}` (kernel/work-item id detection, id-index widening, uniformity via LLVM `UniformityInfo`, SCEV address classification into stream / strided / gather / uniform), `CodeGen.{h,cpp}` (worker-thread vf blocks with predicated linearization, control-thread IR with inline asm, control-thread regions), `main.cpp` (driver) |
| `hwacha-cc/test/run` | 36 regression kernels in 8 binaries |
| `hwacha-cc/test/bench` | saxpy / clamp / divloop / stencil / gather against hand-written Hwacha assembly (`hand.S`), plus ablation variants |
| `hwacha-cc/test/apps` | Rodinia OpenCL kernels (nn, kmeans, bfs, streamcluster `pgain`, pathfinder): kernels unmodified, bare-metal hosts rewritten |
| `hwacha-cc/test/llama` | llama2.c stories260K inference: `llama.cl`, `llama_main.c`, `hwacha_math.h` (vectorized `expf`), model + tokenizer embedded |
| `hwacha-cc/test/gpt2` | GPT-2 forward pass from llm.c on a tiny random model, scalar reference verbatim from `train_gpt2.c` |
| `hwacha-cc/test/gemm` | 256x256x256 sgemm, compared with the Berkeley hand-written `vec-sgemm-naive` / `vec-sgemm-opt` |
| `hwacha-cc/test/berkeley` | RTL results of the Berkeley hand-written benchmarks |
| `hwacha-cc/test/rtl-run.sh` | runs binaries one after another on the RTL simulator, logs to a file |
| `patches/esp-isa-sim.patch` | Spike: `insn_t::bits()` undefined shift for 8-byte instructions (breaks every worker-thread instruction with GCC 13), missing `<cstdint>` |
| `patches/chipyard-hwacha-rtl-fixes.patch` | Chipyard 1.11.0 `generators/hwacha`: four bugs (icache row width, frontend row reuse, SMU TLB `prv`, predicate ALL reduction) without which no loop runs on the RTL |
| `scratch/spmd-spec` | the hand-written OpenCL-to-Hwacha mapping (kernels + hand assembly) that preceded the compiler |
| `docs/` | Hwacha ISA, microarchitecture and evaluation manuals (UCB EECS-2015-262/263/264), also as text |
| `env.sh` | puts the locally built Spike and binutils on `PATH` |

## Requirements

Everything was done on Linux x86-64 (WSL2, 32 cores, 15 GB RAM) without root. Two separate toolchains are
involved because no single one covers both the modern OpenCL front end and the Hwacha assembler:

| component | version used | role |
|---|---|---|
| LLVM + clang | 23.1.1 from conda-forge (`llvmdev`, `clangdev`) | OpenCL C -> LLVM IR front end; `hwacha-cc` links against this LLVM; `llc` compiles the control threads |
| CMake | 3.27 (3.x, not 4) | building `hwacha-cc` |
| Spike with the `hwacha` extension | `ucb-bar/esp-isa-sim` (2022) + `patches/esp-isa-sim.patch` | functional simulation and correctness checks |
| Hwacha-aware binutils | esp fork of binutils 2.29 (in `esp-tools`), or the `esp-tools` conda package | assembling `vf` blocks (`-march=rv64gcxhwacha`) |
| RISC-V GCC + newlib | GCC 9.2 from the `esp-tools` conda package (Chipyard's `.conda-env/esp-tools`), accepts `-march=rv64gcxhwacha` directly | bare-metal hosts; links the generated `.s` |
| Chipyard | 1.11.0 (the last release that still carries Hwacha) + `patches/chipyard-hwacha-rtl-fixes.patch` | RTL, Verilator simulator `simulator-chipyard.harness-HwachaRocketConfig` |
| `esp-tests` | `ucb-bar/esp-tests` | `benchmarks/common` (crt, syscalls, linker script) used by every host program; `isa/rv64uv` tests; hand-written `vec-*` benchmarks |

Only the last row and `esp-isa-sim` are strictly required for the Spike path; the RTL path needs Chipyard.

## Setting up

All upstream trees are expected as siblings of this README (they are git-ignored):
`esp-isa-sim/`, `esp-tests/`, `esp-tools/`, `chipyard/`; toolchains go to `install/`.

### 1. LLVM/clang and CMake

```bash
conda create -p ~/miniforge3 ...           # any conda; the paths below assume ~/miniforge3
conda install -c conda-forge llvmdev=23 clangdev=23 cmake=3.27
```

### 2. Spike with Hwacha

```bash
git clone https://github.com/ucb-bar/esp-isa-sim
git -C esp-isa-sim apply ../patches/esp-isa-sim.patch
mkdir -p build/spike && cd build/spike
../../esp-isa-sim/configure --prefix=$PWD/../../install && make -j1 && make install
# optional: a second copy that prints an "H:" commit log of every Hwacha command (debugging vf blocks)
mkdir -p ../spike-hlog && cd ../spike-hlog
../../esp-isa-sim/configure --prefix=$PWD/../../install-hlog --enable-hcommitlog && make -j1 && make install
```

Use `spike --isa=rv64gc --extension=hwacha prog.riscv`. Spike's `mcycle`/`rdcycle` counts scalar
instructions only; use it for correctness, never for performance.

### 3. RISC-V GCC with the Hwacha assembler

Easiest: the `esp-tools` conda package that Chipyard's setup installs into `chipyard/.conda-env/esp-tools`
(GCC 9.2 + binutils with `xhwacha`). All Makefiles under `hwacha-cc/test` point there:
`CC := $(ROOT)/chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc`.

Without Chipyard: build binutils from `esp-tools/riscv-gnu-toolchain/riscv-binutils-gdb` and a plain
riscv-gnu-toolchain GCC (7.2 was used, needs `CXXFLAGS="-O2 -fpermissive -Wno-error"` with a modern host GCC),
then use the wrapper `install/bin/riscv64-unknown-elf-gcc-xhwacha`, which strips `xhwacha` from `-march` for
`cc1` and passes it to the assembler. Also clone `esp-tests` next to this directory (only `benchmarks/common`
and `env/` are needed by the hosts).

### 4. Chipyard RTL (optional, for cycle counts)

```bash
git clone -b 1.11.0 https://github.com/ucb-bar/chipyard && cd chipyard
echo y | ./build-setup.sh esp-tools --use-lean-conda --skip-toolchain --skip-ctags --skip-firesim --skip-marshal
# if the script dies after creating .conda-env (no conda on PATH in a lean environment), activate it and re-run
# with --skip-conda; see NOTES.md "安装"
git apply ../patches/chipyard-hwacha-rtl-fixes.patch
source ~/miniforge3/etc/profile.d/conda.sh && conda activate $PWD/.conda-env
source env.sh && export RISCV=$PWD/.conda-env/esp-tools
cd sims/verilator && setsid nohup make CONFIG=HwachaRocketConfig SIM_OPT_CXXFLAGS=-O1 -j2 &
```

Notes that cost days: the fesvr headers/library in `$RISCV` must come from an upstream `riscv-isa-sim`
(the esp-tools one is too old for Chipyard 1.11's `SimDRAM.cc`); with 15 GB of RAM use `-j2` and `-O1`
and detach the build (`setsid nohup`), otherwise the memory watchdog kills it. The simulator runs at
about 2.7k cycles/s. Run it as

```bash
./simulator-chipyard.harness-HwachaRocketConfig +permissive +max-cycles=4000000000 +loadmem=prog.riscv +permissive-off prog.riscv
```

`+loadmem=` loads the ELF through the DRAM model's backdoor; without it a 1 MB image (llama) takes hours
to load over TSI. `hwacha-cc/test/rtl-run.sh` wraps this.

### 5. The compiler

```bash
source env.sh
cmake -S hwacha-cc -B hwacha-cc/build -DLLVM_DIR=$HOME/miniforge3/lib/cmake/llvm
cmake --build hwacha-cc/build -j
hwacha-cc/build/hwacha-cc --help
```

Pipeline for one `.cl` file (what every test Makefile does):

```bash
clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header --target=riscv64-unknown-elf -march=rv64gc \
      -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S k.cl -o k.ll
hwacha-cc k.ll -o k.s                 # vf blocks (<kernel>_wt...) + control threads (<kernel>_ct, via llc)
riscv64-unknown-elf-gcc -march=rv64gcxhwacha -mabi=lp64d -mcmodel=medany -static -O2 \
      -I esp-tests/benchmarks/common -I esp-tests/env main.c k.s esp-tests/benchmarks/common/crt.S \
      esp-tests/benchmarks/common/syscalls.c -nostdlib -nostartfiles -lm -lgcc -T esp-tests/benchmarks/common/test.ld -o prog.riscv
```

Host side: each kernel `k(args...)` becomes a C-callable `void k_ct(long n, args...)` that runs the whole
1-D NDRange of `n` work-items (stripmined by the control thread) and returns after a `fence`. Uniform
arguments are passed by value, buffers as pointers; `__local` buffers and `barrier` are supported for
one work-group per stripmine. The globals `hwacha_group_size` (caps the vector length so a group is
exactly that size) and `hwacha_vl_short` (set if the hardware could not provide it) are in
`test/apps/common.h`.

Options: `--analyze` (print per-kernel uniformity and address classification and exit), `--kstats`
(registers, instructions, masks, jumps per kernel), `--keep` (keep the control-thread `.ll`/`.s`),
`--verbose`, and the ablations `--no-ct-loops` (keep loops inside the vf block), `--no-skip`,
`--no-coalesce`, `--no-v32`. `--scalar-fp` and `--no-subword-rmw` produce code that runs on Spike but
hangs the RTL (shared FPU; masked sub-word stores), see NOTES.md.

## Running the tests

All commands assume `source env.sh` and a built compiler. Spike runs take seconds to a minute; RTL runs are
listed with their approximate wall time.

| suite | Spike | RTL |
|---|---|---|
| regression (36 kernels) | `cd hwacha-cc/test/run && make run` -> 8x `ALL KERNELS PASSED` | `make <name>.rtl` for one binary |
| microbenchmarks | `cd ../bench && make N=1024 && spike --isa=rv64gc --extension=hwacha bench-n1024.riscv` -> `ALL VERIFIED` | `./run-all.sh 1024` (5 variants in parallel, ~15 min), `make N=4096 bench-n4096.riscv` (~40 min); logs in `results/` |
| Rodinia | `cd ../apps && make spike` -> 6x `PASS` | `./run-rtl-seq.sh results/x.out nn.riscv kmeans.riscv pgain.riscv pathfinder.riscv bfs.riscv` (~1 h total) |
| llama2.c | `cd ../llama && make llama.riscv && spike ... llama.riscv` -> 40 tokens, text identical to x86 `run`, `llama PASS` | `make llama_rtl.riscv && ../rtl-run.sh results/x.out llama_rtl.riscv` (4 tokens: ~70 min scalar + ~25 min vector) |
| GPT-2 forward | `cd ../gpt2 && make gpt2.riscv && spike ... gpt2.riscv` -> `gpt2 PASS`; `gcc -O2 -DX86 gpt2_main.c -lm` prints the x86 reference lines | `../rtl-run.sh results/x.out gpt2.riscv` (~3 h, scalar reference dominates) |
| sgemm 256^3 | `cd ../gemm && make gemm.riscv && spike ... gemm.riscv` | `make gemm_rtl.riscv` (no scalar reference) then `../rtl-run.sh ...` (~2.5 h) |
| Berkeley hand-written | `esp-tests/benchmarks`: `make RISCV_PREFIX=... vec-sgemm-opt.riscv`, see `test/berkeley/results` | same binaries with `rtl-run.sh` |

The x86 reference for llama is `test/llama/run_x86` (`gcc -O2 run.c -lm`, run with
`stories260K.bin -z tok512.bin -t 0 -n 40`). Expected output of every RTL run of this snapshot is in the
respective `results/` directory.

## Results snapshot (Chipyard HwachaRocketConfig RTL, cycles)

| program | scalar Rocket | hwacha-cc | speedup |
|---|---|---|---|
| saxpy N=4096 | 55851 | 3776 (hand-written 3938) | 14.8x |
| clamp N=4096 | 74176 | 3102 (hand-written 3300) | 23.9x |
| kmeans 1024 points x 8 dims x 5 clusters | 1269172 | 96719 | 13.1x |
| streamcluster pgain 1024 x 8 | 331274 | 28776 | 11.5x |
| nn 2048 | 100377 | 15072 | 6.7x |
| pathfinder 8 x 1024 | 248183 | 144914 | 1.7x |
| bfs 2048 nodes (GPU-style level-synchronous, work-inefficient by design) | 191905 | 621840 | 0.3x |
| llama2.c stories260K, per token (transposed weights) | 2.88M | 235k | 12.3x |
| sgemm 256^3 | (hand-written naive 13918519, opt 4262080) | 5211891 | 2.7x naive, 0.82x opt |

What the numbers say: unit-stride streams and control-thread-driven loops get the compiler within
10-25% of hand-written code on streaming kernels; the remaining gap on gemm is register blocking (one
vf per loop iteration is issue-bound when the vector length is small). Details and the ablations are in
NOTES.md.

## Known RTL constraints the compiler works around

- Scalar-destination floating-point ops inside a vf block go to the shared Rocket FPU and never complete
  on the RTL: all FP work is kept in vector registers (`needsFPU`).
- A masked sub-word unit-stride store with a single active lane deadlocks the VMU: masked byte/half
  stores are emitted as load / select / unmasked store.
- Predicate-logic ops are never masked (Spike and RTL agree): conditional predicate moves are 3-input
  `vpop` muxes.
- A consensual jump (`vcjal`) costs ~50 cycles: blocks are skipped only when they hold at least two
  instructions, never at loop headers, and uniform loops leave the vf block entirely.

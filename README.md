# hwacha-compiler

Research on the Hwacha vector-fetch architecture (UC Berkeley) through a compiler.
`hwacha-cc` compiles OpenCL C kernels to Hwacha vector-fetch code: the divergent part of a kernel
becomes vector-fetch (`vf`) blocks, the uniform control flow runs on the scalar control thread, and
the result is verified on the Spike ISA simulator and measured on the Chipyard Verilator RTL of
`HwachaRocketConfig`.

The full engineering log — design decisions, every RTL and compiler bug found, all measurements, dead
ends — is `NOTES.md` (Chinese). This file is the practical guide to rebuilding and reproducing from a
fresh clone.

## What is in this repository

Only first-party work is tracked. Large upstream trees and build products are **not** in the repo
(see `.gitignore`) and must be fetched/built as described below.

| path | what |
|---|---|
| `hwacha-cc/src` | the compiler: `Analysis.{h,cpp}` (kernel/id detection, id-index widening, uniformity via LLVM `UniformityInfo`, SCEV address classification into stream/strided/gather/uniform), `CodeGen.{h,cpp}` (worker-thread vf blocks, control-thread IR, control-thread regions, cross-lane reduction trees), `main.cpp` |
| `hwacha-cc/test/run` | regression: `spec gather cf lc local misc flag pfmin red` (8 binaries, ~40 kernels) |
| `hwacha-cc/test/bench` | saxpy/clamp/divloop/stencil/gather vs hand-written Hwacha asm (`hand.S`) + ablations |
| `hwacha-cc/test/apps` | Rodinia kernels (nn, kmeans, bfs, streamcluster `pgain`, pathfinder), unmodified kernels + bare-metal hosts |
| `hwacha-cc/test/llama` | llama2.c stories260K inference (model + tokenizer embedded, vectorized `expf`) |
| `hwacha-cc/test/gpt2` | GPT-2 forward from llm.c on a tiny random model, incl. `work_group_reduce` variants |
| `hwacha-cc/test/gemm` | 256³ sgemm vs Berkeley `vec-sgemm-naive`/`opt`; `hwacha-cc/test/berkeley` holds their RTL logs |
| `hwacha-cc/test/rtl-run.sh` | run one or more binaries on the RTL sim (adds `+loadmem`), log to a file |
| `scripts/build-sim.sh` | build the Chipyard Verilator sim of `HwachaRocketConfig` (threads/opt configurable) |
| `patches/` | local fixes to the upstream trees — see the table under "Patches" below |
| `scratch/spmd-spec` | the hand-written OpenCL→Hwacha mapping that preceded the compiler |
| `docs/` | Hwacha ISA / microarch / eval manuals (UCB EECS-2015-262/263/264), also as text |
| `env.sh` | puts the locally built Spike and binutils on `PATH` |
| `NOTES.md` | the full engineering log |

### Patches

| file | applies to | why |
|---|---|---|
| `esp-isa-sim.patch` | `esp-isa-sim` (Spike) | `insn_t::bits()` undefined shift for 8-byte insns (every worker insn reads 0 under GCC 13); missing `<cstdint>` |
| `esp-isa-sim-hwacha-trace.patch` | `esp-isa-sim` | optional `H:` commit-log of Hwacha commands, for debugging vf blocks |
| `chipyard-hwacha-rtl-fixes.patch` | `chipyard/generators/hwacha` | four integration bugs (icache row width, frontend row reuse, SMU TLB `prv`, predicate ALL reduction), the FPU type-tag fix, and plusarg-gated VMU/TileLink trace |
| `chipyard-rocketchip-fpu-fix.patch` | `chipyard/generators/rocket-chip` | RoCC FPU port was tied to `DontCare` after the arbiter connection, hanging Hwacha scalar FP |

## Prerequisites

Done on Linux x86-64 (WSL2, 32 cores, 15 GB RAM), no root. Everything installs into a user-local conda
and a build tree next to this README. Install once:

- **conda** (miniforge). The paths below assume `~/miniforge3`.
- **LLVM 23 + clang** for the OpenCL front end and `llc`:
  `conda install -c conda-forge llvmdev=23 clangdev=23 cmake=3.27`  (CMake must be 3.x, not 4)
- **git**, a host C++ toolchain, and `dtc`/`makeinfo` (`conda install -c conda-forge dtc texinfo`).

## Layout after cloning

Clone the repo, then fetch the upstream trees as siblings inside it (they are git-ignored):

```
hwacha-compiler/            <- this repo
├── esp-isa-sim/            <- git clone ucb-bar/esp-isa-sim   (Spike + hwacha)
├── esp-tests/              <- git clone ucb-bar/esp-tests     (benchmarks/common, isa tests)
├── chipyard/               <- git clone -b 1.11.0 ucb-bar/chipyard  (RTL; brings esp-tools GCC)
├── install/  install-hlog/ <- built Spike(s) (created below)
└── build/                  <- Spike build dirs (created below)
```

The RTL path needs `chipyard`; the Spike-only path needs just `esp-isa-sim` and `esp-tests`.

## Build, step by step

### 1. Spike with the Hwacha extension

```bash
cd ~/hwacha-compiler
git clone https://github.com/ucb-bar/esp-isa-sim
git -C esp-isa-sim apply patches/esp-isa-sim.patch
mkdir -p build/spike && (cd build/spike && ../../esp-isa-sim/configure --prefix=$PWD/../../install && make -j$(nproc) && make install)
# optional commit-log build for debugging vf blocks:
git -C esp-isa-sim apply patches/esp-isa-sim-hwacha-trace.patch
mkdir -p build/spike-hlog && (cd build/spike-hlog && ../../esp-isa-sim/configure --prefix=$PWD/../../install-hlog --enable-hcommitlog && make -j$(nproc) && make install)
```

`source env.sh` then puts `spike` on `PATH`. Spike is a functional model: use it for correctness,
never for cycles (`rdcycle` counts scalar instructions only).

### 2. RISC-V GCC with the Hwacha assembler

Simplest is the `esp-tools` conda package Chipyard installs into `chipyard/.conda-env/esp-tools`
(GCC 9.2 + binutils that accept `-march=rv64gcxhwacha`). Every Makefile under `hwacha-cc/test`
already points at `chipyard/.conda-env/esp-tools/bin/riscv64-unknown-elf-gcc`, so doing the Chipyard
step below also provides the compiler for the Spike path. If you only want Spike and not the RTL, build
binutils from `esp-tools/riscv-gnu-toolchain/riscv-binutils-gdb` and use the
`install/bin/riscv64-unknown-elf-gcc-xhwacha` wrapper (it strips `xhwacha` from `-march` for `cc1` and
passes it to the assembler), and clone `esp-tests` for `benchmarks/common` and `env/`.

### 3. Chipyard RTL (only for cycle counts)

```bash
cd ~/hwacha-compiler
git clone -b 1.11.0 https://github.com/ucb-bar/chipyard
cd chipyard
echo y | ./build-setup.sh esp-tools --use-lean-conda --skip-toolchain --skip-ctags --skip-firesim --skip-marshal
# if it dies after creating .conda-env (no conda on PATH), activate it and re-run with --skip-conda (see NOTES.md "安装")
git apply ../patches/chipyard-hwacha-rtl-fixes.patch
git -C generators/rocket-chip apply ../../patches/chipyard-rocketchip-fpu-fix.patch
cd ~/hwacha-compiler
./scripts/build-sim.sh          # detached, ~40-60 min; edit THREADS/OPT at the top
```

`scripts/build-sim.sh` runs `make CONFIG=HwachaRocketConfig VERILATOR_THREADS=8 SIM_OPT_CXXFLAGS=-O2`.
The multithreaded `-O2` sim runs at **~14k cycles/s** (5.2× the single-thread `-O1` default). Notes
that cost days the first time: the fesvr in `$RISCV` must come from an upstream `riscv-isa-sim`
(esp-tools' is too old for Chipyard 1.11's `SimDRAM.cc`); with 15 GB RAM keep `-j2` and detach the
build (`setsid nohup`) or the memory watchdog kills it.

Run a binary on the sim (note `+loadmem` must sit inside the `+permissive` window):

```bash
SIM=chipyard/sims/verilator/simulator-chipyard.harness-HwachaRocketConfig
$SIM +permissive +max-cycles=4000000000 +loadmem=$PWD/prog.riscv +permissive-off $PWD/prog.riscv
```

`+loadmem=` back-doors the ELF into DRAM; without it a 1 MB image (llama) takes hours over TSI.
`hwacha-cc/test/rtl-run.sh <out> <bin>...` wraps this correctly.

### 4. The compiler

```bash
cd ~/hwacha-compiler
source env.sh
cmake -S hwacha-cc -B hwacha-cc/build -DLLVM_DIR=$HOME/miniforge3/lib/cmake/llvm
cmake --build hwacha-cc/build -j
hwacha-cc/build/hwacha-cc --help
```

One `.cl` file end to end (what every test Makefile does):

```bash
clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header --target=riscv64-unknown-elf -march=rv64gc \
      -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S k.cl -o k.ll
hwacha-cc/build/hwacha-cc k.ll -o k.s            # <kernel>_wt vf blocks + <kernel>_ct control thread (via llc)
riscv64-unknown-elf-gcc -march=rv64gcxhwacha -mabi=lp64d -mcmodel=medany -static -O2 \
      -I esp-tests/benchmarks/common -I esp-tests/env main.c k.s \
      esp-tests/benchmarks/common/crt.S esp-tests/benchmarks/common/syscalls.c \
      -nostdlib -nostartfiles -lm -lgcc -T esp-tests/benchmarks/common/test.ld -o prog.riscv
```

Each kernel `k(args...)` becomes a C-callable `void k_ct(long n, args...)` that runs the whole 1-D
NDRange of `n` work-items and returns after a `fence`. `__local`/`barrier` are supported for one
work-group per stripmine; `hwacha_group_size` caps the vector length to the work-group size.
Useful flags: `--analyze`, `--kstats`, `--keep`, `--verbose`; ablations `--no-ct-loops`, `--no-skip`,
`--no-coalesce`, `--no-v32`; and `--scalar-fp` / `--no-subword-rmw` (see "Known RTL constraints").

## Reproduce the tests

`source env.sh` and build the compiler first. Fastest path to confidence is the **Spike column** — it
checks correctness in seconds to a minute per suite. The **RTL column** produces cycle counts and takes
minutes to hours.

| suite | Spike (correctness) | RTL (cycles) |
|---|---|---|
| regression | `cd hwacha-cc/test/run && make run` → 9× `ALL KERNELS PASSED` | `make <name>.rtl` for one binary |
| microbench | `cd ../bench && make N=1024 && spike --isa=rv64gc --extension=hwacha bench-n1024.riscv` → `ALL VERIFIED` | `./run-all.sh 1024` (~15 min); `make N=4096 bench-n4096.riscv` (~40 min) |
| Rodinia | `cd ../apps && make spike` → 6× `PASS` | `./run-rtl-seq.sh results/x.out nn.riscv kmeans.riscv pgain.riscv pathfinder.riscv bfs.riscv` (~1 h) |
| llama2.c | `cd ../llama && make llama.riscv && spike --isa=rv64gc --extension=hwacha llama.riscv` → 40 tokens match x86, `llama PASS` | `make llama_rtl.riscv && ../rtl-run.sh results/x.out llama_rtl.riscv` (4 tokens, ~1.5 h) |
| GPT-2 fwd | `cd ../gpt2 && make gpt2.riscv && spike --isa=rv64gc --extension=hwacha gpt2.riscv` → `gpt2 PASS` (both lane-per-row and reduction variants) | `../rtl-run.sh results/x.out gpt2.riscv` (~3 h, scalar ref dominates) |
| sgemm 256³ | `cd ../gemm && make gemm.riscv && spike --isa=rv64gc --extension=hwacha gemm.riscv` | `make gemm_rtl.riscv && ../rtl-run.sh results/x.out gemm_rtl.riscv` (~2.5 h) |
| Berkeley asm | — | `cd esp-tests/benchmarks && make RISCV_PREFIX=<esp-tools>/bin/riscv64-unknown-elf- vec-sgemm-opt.riscv`, run with `rtl-run.sh` |

The x86 reference for llama is `hwacha-cc/test/llama/run_x86` (`gcc -O2 run.c -lm`, run with
`stories260K.bin -z tok512.bin -t 0 -n 40`). Each `test/*/results/` directory holds the expected output
of this snapshot's RTL runs.

Tips: RTL runs are long — launch them detached (`setsid nohup … &`) and cap cycles (`+max-cycles`);
the scalar reference inside llama/gpt2/gemm dominates wall time, so once its cycle count is known you
can build `*_rtl` variants that skip it.

## Results snapshot (Chipyard HwachaRocketConfig RTL, cycles)

| program | scalar Rocket | hwacha-cc | speedup |
|---|---|---|---|
| saxpy N=4096 | 55851 | 3776 (hand 3938) | 14.8× |
| clamp N=4096 | 74176 | 3102 (hand 3300) | 23.9× |
| kmeans 1024×8×5 | 1269172 | 96719 | 13.1× |
| streamcluster pgain 1024×8 | 331274 | 28776 | 11.5× |
| nn 2048 | 100377 | 15072 | 6.7× |
| pathfinder 8×1024 | 248183 | 144914 | 1.7× |
| bfs 2048 (work-inefficient by design) | 191905 | 621840 | 0.3× |
| llama2.c stories260K, per token | 2.88M | 235k | 12.3× |
| GPT-2 forward, tiny, 16 tokens | 31244913 | 1627888 | 19.2× |
| sgemm 256³ | (hand naive 13918519, opt 4262080) | 5211891 | 2.7× naive, 0.82× opt |

Unit-stride streams and control-thread loops get within 10–25% of hand-written code on streaming
kernels; the gemm gap is register blocking. NOTES.md has the ablations and per-kernel breakdowns.

## Known RTL constraints and fixes

The four Chipyard-integration bugs and the two scalar-FP bugs are **fixed** by the patches above
(`sfp` test passes; loops run). The compiler still works around two hardware issues:

- **Masked sub-word unit-stride store** hangs the VMU on sparse patterns — a store-credit leak on the
  byte-store response path (localized in NOTES.md, not yet fixed in RTL). The compiler lowers masked
  byte/half stores to load/select/store; `--no-subword-rmw` disables that workaround.
- **Predicate-logic ops are never masked** (Spike and RTL agree): conditional predicate moves are
  3-input `vpop` muxes.
- A **consensual jump (`vcjal`) costs ~50 cycles**, so blocks are skipped only when they hold ≥2
  instructions and never at loop headers; uniform loops leave the vf block entirely (control-thread
  regions).
- Scalar (vs-destination) FP is now functional on RTL after the FPU patches; the compiler still keeps
  FP in vector registers by default (`needsFPU`), `--scalar-fp` opts into vs-destination FP.

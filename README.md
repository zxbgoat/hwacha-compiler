# hwacha-compiler

Research on the Hwacha vector-fetch architecture (UC Berkeley) through a compiler:
`hwacha-cc` compiles OpenCL C kernels to Hwacha vector-fetch code and runs them on the
Spike ISA simulator and on the Chipyard Verilator RTL of `HwachaRocketConfig`.

The full engineering log (design, bugs, measurements, dead ends) is `NOTES.md` (Chinese).

## Layout

| path | what |
|---|---|
| `hwacha-cc/src` | the compiler: `Analysis.{h,cpp}` (kernel/id detection, uniformity, SCEV address classification), `CodeGen.{h,cpp}` (worker-thread vf blocks, control-thread IR, control-thread regions), `main.cpp` (driver) |
| `hwacha-cc/test/run` | 36 regression kernels in 8 binaries (`make run` on Spike) |
| `hwacha-cc/test/bench` | saxpy/clamp/divloop/stencil/gather vs hand-written Hwacha assembly (`hand.S`), RTL runner |
| `hwacha-cc/test/apps` | Rodinia OpenCL kernels (nn, kmeans, bfs, streamcluster pgain, pathfinder), unmodified kernels + bare-metal hosts |
| `hwacha-cc/test/llama` | llama2.c stories260K inference (`llama.cl`, `llama_main.c`, `hwacha_math.h` with a vectorized expf) |
| `hwacha-cc/test/gpt2` | GPT-2 forward pass from llm.c on a tiny random model |
| `hwacha-cc/test/gemm` | 256x256 sgemm vs the Berkeley hand-written `vec-sgemm-*` benchmarks |
| `hwacha-cc/test/berkeley` | RTL results of the Berkeley hand-written benchmarks |
| `patches/` | local fixes to upstream trees: Spike (`esp-isa-sim.patch`), Chipyard 1.11.0 Hwacha RTL (`chipyard-hwacha-rtl-fixes.patch`, four bugs) |
| `scratch/spmd-spec` | the hand-written OpenCL->Hwacha mapping spec that preceded the compiler |
| `docs/` | Hwacha ISA / microarchitecture / evaluation manuals (UCB tech reports) |
| `env.sh` | puts the locally built Spike/binutils on PATH |

Not in the repository (clone/build separately, see NOTES.md "Phase 0" and the RTL sections):
`esp-isa-sim`, `esp-opcodes`, `esp-tests`, `esp-tools`, `hwacha`, `esp-llvm`, `chipyard` (1.11.0, the last release with Hwacha),
and the `install*/` toolchains. `hwacha-cc` needs a modern LLVM (23 used) with clang for the OpenCL front end.

## Build and run

```bash
source env.sh
cmake -S hwacha-cc -B hwacha-cc/build -DLLVM_DIR=$HOME/miniforge3/lib/cmake/llvm
cmake --build hwacha-cc/build
cd hwacha-cc/test/run  && make run            # regression on Spike
cd ../apps             && make spike          # Rodinia on Spike
cd ../llama            && make llama.riscv && spike --isa=rv64gc --extension=hwacha llama.riscv
cd ../gpt2             && make gpt2.riscv  && spike --isa=rv64gc --extension=hwacha gpt2.riscv
```

RTL: build the Chipyard simulator with the patch applied (see NOTES.md), then use
`test/rtl-run.sh <out> <binaries...>` (adds `+loadmem=` so large images load through the DRAM backdoor).

Compiler options worth knowing: `--analyze` (print the analysis), `--kstats` (per-kernel register/instruction stats),
`--no-ct-loops` (keep loops inside the vf block: ablation for control-thread regions), `--verbose`.

## Results snapshot (Chipyard HwachaRocketConfig RTL, cycles)

| program | scalar Rocket | hwacha-cc | speedup |
|---|---|---|---|
| saxpy N=4096 | 55851 | 3776 (hand-written 3938) | 14.8x |
| kmeans 1024x8x5 | 1269172 | 96719 | 13x |
| streamcluster pgain | 331274 | 28776 | 11.5x |
| llama2.c stories260K, per token | 2.88M | 235k | 12.3x |
| sgemm 256^3 | (hand-written opt 4262080) | 5211891 | 0.82x of hand-written |

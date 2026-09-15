# Hwacha 架构研究：工具链现状调研（2026-09-08）

## 仓库清单（已 clone 到本目录）

| 目录 | 来源 | 最后提交 | 内容 |
|---|---|---|---|
| esp-llvm | ucb-bar/esp-llvm | 默认分支 2017-12（LLVM 3.9） | RISC-V 后端。**Hwacha 代码只在分支 `hwachav4_llvm39`（2019-02）、`hwachav4`（2017-08）、`hwacha`（2015-11）上**，当前已 checkout `origin/hwachav4_llvm39` |
| esp-opcodes | ucb-bar/esp-opcodes | 2019-03 | Hwacha 指令编码：`opcodes-hwacha`（控制线程 23 条）、`opcodes-hwacha-ut`（工作线程 ~230 条），含 LaTeX 表格生成 |
| esp-isa-sim | ucb-bar/esp-isa-sim | 2022-08 | Spike，含 `hwacha/` 扩展（`--extension=hwacha`），是当前最容易跑起来的 Hwacha 功能模型 |
| esp-tests | ucb-bar/esp-tests | 2021-06 | `isa/rv64uv/` 103 个向量 ISA 测试；`benchmarks/vec-*` 手写汇编 benchmark（daxpy/sgemm/spmv 等） |
| hwacha | ucb-bar/hwacha | 2023-08 | Chisel RTL |
| esp-tools | ucb-bar/esp-tools | — | 元仓库，子模块含 esp-gnu-toolchain（binutils 有 `xhwacha` 汇编支持，`-march=rv64gxhwacha`） |

## 编程模型（Hwacha v4，来自 esp-tests/benchmarks/vec-daxpy）

标量控制线程（Rocket 上跑）：
```
li t0, VCFG(2,0,0,1); vsetcfg t0      # 配置向量寄存器数量/精度
vmcs vs1, a3                           # 标量 → 向量共享标量寄存器
stripmine:
  vsetvl t0, a0                        # 协商向量长度
  vmca va0, a1; vmca va1, a2           # 传地址寄存器
  vf 0(t5)                             # 向量 fetch：把一个指令块丢给向量单元
  ... 更新指针 / 计数，循环
fence
```
工作线程（向量单元执行的 vf 块）：
```
daxpy_v:
  vpset vp0
  vld vv0, va0; vld vv1, va1
  vfmadd.d vv1, vs1, vv0, vv1
  vsd vv1, va1
  vstop
```
关键点：两条指令流、四类寄存器（vv 向量、vs 共享标量、va 地址、vp 谓词）、向量块内可有分支（vcjal）和谓词化控制流。

## esp-llvm 的 Hwacha 后端结构（hwachav4_llvm39 分支，相对 riscv-trunk 约 +5100 行）

编程模型是 **OpenCL SPMD**：kernel 函数通过 `opencl.kernels` 元数据识别，
`llvm.hwacha.veidx` intrinsic 表示元素索引（即 get_global_id）。

Pass 流水线（`RISCVTargetMachine.cpp`）：
1. `RISCVVectorFetchIROpt`（IR 级 ModulePass）：对 kernel 内 load/store 的地址用 SCEV 分析，
   把 `base + veidx*stride` 形式拆出来，base 提升为 kernel 参数（后续变成 `vmca` 地址寄存器），
   把 kernel 重写为带地址参数的新函数。
2. 通用 DCE / DeadArgElimination（后者打了补丁）。
3. ISel → 普通 RISC-V MachineInstr。
4. `MachineProgramDependenceGraph` + `MachineScalarization`（新增分析）：判定哪些值是
   per-element 变化的（需要 vv 寄存器）、哪些是 uniform 的（可以放 vs 寄存器）。
5. `RISCVVectorFetchMachOpt`：**核心**，逐条把标量 MachineInstr 改写成对应的 Hwacha 工作线程指令
   （按操作数是 vector/scalar 选 VVV/VVS/VSV/VSS/SSS 变体），把控制流转成谓词（`convertToPredicates`）。
6. `RISCVVectorFetchRegFix`、`RISCVVectorFetchPreEmitOpt`：寄存器类修正和 emit 前处理。

相关文件：
- `lib/Target/RISCV/RISCVVectorFetchOpimizer.cpp`（1713 行，上面 4 个 pass）
- `lib/Target/RISCV/RISCVInstrInfoXhwacha.td`（326）、`RISCVRegisterInfoXhwacha.td`（818）、`RISCVCallingConvXhwacha.td`
- `lib/Analysis/{ProgramDependenceGraph,Scalarization}.cpp`、`lib/CodeGen/Machine{ProgramDependenceGraph,Scalarization}.cpp`
- `include/llvm/IR/IntrinsicsRISCV.td`（只有 veidx 一个 intrinsic）

状态：作者自述 "Code dump. Old changes to llvm3.9 bump"，是 WIP 代码；`test/CodeGen/RISCV/` 没有任何 Hwacha 测试；
只做汇编生成，汇编/链接依赖 esp-gnu-toolchain。**OpenCL 前端和运行时不在此仓库中**（需要 clang 的 OpenCL 支持产出 `opencl.kernels` 元数据，并自己写 host 侧启动代码）。

## 硬件侧

- Chipyard **1.9.x–1.11.0** 含 `generators/hwacha` 子模块和 `HwachaRocketConfig` / `HwachaLargeBoomConfig`；
  **1.12.0 起移除**。要跑 RTL 用 Chipyard 1.11.0。
- esp-tools 构建：Chipyard 内 `./scripts/build-toolchains.sh esp-tools`。

## 下一步建议

阶段 0（能跑）：
1. 构建 esp-isa-sim（Spike + hwacha）和 esp-gnu-toolchain，跑通 `esp-tests/isa/rv64uv` 和 `benchmarks/vec-daxpy`。
   这给出一个不依赖 RTL 的功能验证环境。
2. 用 Chipyard 1.11.0 出 `HwachaRocketConfig` 的 Verilator 仿真，做性能研究用。

阶段 1（编译器）：
3. 决策：在 LLVM 3.9 fork 上修，还是把 Hwacha 部分迁到现代 LLVM。
   - 老 fork 优点：pass 流水线已经成形，可以先看它能编出什么；缺点：LLVM 3.9 用现代 GCC 编译要打补丁，RISC-V 后端是伯克利自研的、和主线后端完全不同。
   - 迁移方案：以主线 RISC-V 后端为基础，新增 XHwacha 子目标 + 寄存器/指令 .td，把 Scalarization / PDG / VectorFetchMachOpt 三块移植过来。这几块逻辑相对独立于 LLVM 版本，主要工作是 API 适配。
4. 前端：不一定要 OpenCL。可以先用一个更简单的约定（比如 C 函数 + `__attribute__((annotate("hwacha_kernel")))` + `__builtin_hwacha_veidx()`），产出与 `opencl.kernels` 等价的元数据，绕开 OpenCL 运行时。

---

# 阶段 0 完成记录（2026-09-09）：Spike + 工具链可用

## 已构建（全部安装到 `install/`，用 `source env.sh` 进入环境）

| 组件 | 源 | 构建目录 | 备注 |
|---|---|---|---|
| Spike（`spike`，`--extension=hwacha`） | esp-isa-sim | build/spike | 两处补丁，见下 |
| Spike 带 Hwacha 提交日志 | 同上，`--enable-hcommitlog` | build/spike-hlog → `install-hlog/` | stderr 打印 `H: VSETCFG/VSETVL/write_arf/EXCPT…`，调试 vf 块必备 |
| binutils 2.29（esp fork，`-march=rv64gxhwacha`） | esp-tools/riscv-gnu-toolchain/riscv-binutils-gdb | build/binutils | 汇编/反汇编 Hwacha 控制线程和工作线程指令均正常 |
| GCC 7.2.0 + newlib | riscv-gnu-toolchain（子模块 URL 需从 git:// 改 https） | build/gnu | 宿主 GCC 13 编译时加 `CXXFLAGS="-O2 -fpermissive -Wno-error"` 即可 |
| `riscv64-unknown-elf-gcc-xhwacha` | install/bin 里的包装脚本 | — | GCC 7.2 不认 `-march=…xhwacha`；脚本把它改成 `-march=rv64gc -Wa,-march=rv64gcxhwacha` |

依赖：`dtc`、`makeinfo` 通过 conda 安装到用户目录（无 sudo）。

## 补丁

1. `esp-isa-sim/fesvr/device.h`：加 `#include <cstdint>`（GCC 13 编译失败）。
2. **`esp-isa-sim/riscv/decode.h:81` `insn_t::bits()`**：原来是 `b & ~(UINT64_MAX << (length()*8))`，
   对 8 字节指令移位量为 64，是未定义行为。GCC 13 生成的代码让所有 64 位工作线程指令读成 0，
   于是 vf 块第一条指令就报 `VF_ILLEGAL_INSTRUCTION`（表现为 rv64uv 里 80 个测试全部 tohost=1337）。
   改为 `length()>=8 ? b : …`。**这是老 Spike 在新编译器下不可用的根本原因，和 Hwacha 本身无关。**

## 验证结果

- `esp-tests/isa/rv64uv`：95 个 `-p`（裸机）测试 **95/95 通过**（build/tests-isa）。
  `-v`（虚拟内存）变体没编：`env/v/entry.S` 用了 binutils 2.29 不认识的 CSR 名（`stval` 等）。
- `esp-tests/benchmarks/vec-*`：14 个手写汇编 benchmark（daxpy/saxpy/hsaxpy/sdaxpy/vvadd/stream/
  sgemm-naive/sgemm-opt/dgemm-opt/dgemm-opt-multi/hgemm-opt/sdgemm-opt/hsgemm-opt/saxpy-streamx）
  **全部校验通过**（build/bmarks）。Spike 是功能模拟器，`mcycle` 只是控制线程指令数，
  但这本身就是 vector-fetch 模型的一个观测量：例如 vec-daxpy 整个数组只用了 146 条标量指令。
- 探针 `scratch/spike-probe/probe.S`：`vsetcfg 5,1` → maxvl=408（Spike 模型：8*(256/nxpr)），
  `vsetvl` 返回 min(请求, 408)，拷贝无误。

## 复现命令

```bash
source ~/hwacha-compiler/env.sh
# ISA 测试
cd build/tests-isa && make -f ../../esp-tests/isa/Makefile src_dir=../../esp-tests/isa \
    RISCV_GCC=riscv64-unknown-elf-gcc-xhwacha -j32 rv64uv
spike --isa=rv64gc --extension=hwacha rv64uv-p-vvadd_d      # 退出码 0 = 通过
# benchmark
cd build/bmarks && make -f ../../esp-tests/benchmarks/Makefile src_dir=../../esp-tests/benchmarks \
    RISCV_GCC=riscv64-unknown-elf-gcc-xhwacha vec-daxpy.riscv
spike --isa=rv64gc --extension=hwacha vec-daxpy.riscv
# 看 vf 块内部发生了什么
~/hwacha-compiler/install-hlog/bin/spike --isa=rv64gc --extension=hwacha vec-daxpy.riscv 2>&1 | grep '^H:'
```

## 文档（docs/，已转成 txt）

- EECS-2015-262 Hwacha Vector-Fetch Architecture Manual v3.8.1（ISA）
- EECS-2015-263 Hwacha Microarchitecture Manual v3.8.1
- EECS-2015-264 Hwacha Preliminary Evaluation Results v3.8.1
  （官方 www2.eecs 链接已 404，从 digicoll.lib.berkeley.edu 的 record 134760/136977/135141 下载）

## 关键架构事实（来自 ISA 手册，供后续编译器设计参考）

- 寄存器：vv0–255（向量）、vp0–15（谓词）、vs0–63（共享标量，vf 块内可读写，vs0=0）、va0–31（地址，vf 块内只读）。
- `vsetcfg #v64,#pred[,#v32,#v16]` 决定最大硬件向量长度：寄存器用得越少，vlen 越长；至少保证 8。
- 控制线程指令 32 位（RoCC custom-0/1）；工作线程指令 **64 位**，vf 块须 8 字节对齐。
- vf 块内控制流只有 `vcjal/vcjalr`（consensual：谓词 all/any 才跳），其余靠谓词化；`vstop` 结束块，`vfence` 排序访存。
- 工作线程只会产生非对齐访存和非法指令两类异常。

---

# 实验：老 OpenCL/LLVM 编译器还能不能用（2026-09-09）

构建目录 build/llvm39（`ninja llc`，只编 RISCV target），测试输入在 scratch/opencl-llvm39/。

## 让它编过、跑起来需要的补丁

1. CMake 必须是 3.x（CMake 4 删了 CMP0051 OLD）：`conda install cmake=3.27`。
2. `lib/Target/RISCV/RISCV.td`：feature 名 `"Xhwacha"` → `"xhwacha"`。LLVM 会把 `-mattr` 小写化再查表，原名永远匹配不上，
   意味着这条分支从来没法通过命令行打开 Hwacha 特性（作者当年大概是靠别的方式或者本地改动）。
3. `RISCVTargetMachine.cpp` `addPreISel()` 缺 `return`，UB；GCC 13 -O2 直接掉进下一个函数，llc 秒崩。
4. 输入 IR 必须预先带 `!hwacha.vfcfg = !{!{i64 0,i64 0,i64 0,i64 0}}` 占位，否则 RegFix 空指针。

## 输入约定（通过读代码反推，没有任何文档/测试）

- kernel 靠 `!opencl.kernels` 元数据识别；`get_global_id(0)` 要写成 `call i64 @llvm.hwacha.veidx()`。
- 模块里必须有一个**调用 kernel 的标量函数**：IR 级 pass 是在调用点用 SCEV 展开 hoist 出来的地址，
  调用点被改写成 `vsetcfg; vsetvl; vmca…; vf`。
- 命令：`llc -mtriple=riscv64-unknown-elf -mcpu=Rocket -mattr=+xhwacha -O2 x.ll`

## 结果

**vvadd（`c[i]=a[i]+b[i]`，i64）**：llc 跑完，输出结构对但代码错：
```
vvadd:            host:
  vpset vp0         addi x10,x10,8   ; ← 地址 hoist 多加了一个元素
  veidx vv0         addi x11,x11,8
  ld x5,0(x11)      addi x12,x12,8
  ld x6,0(x10)      lui/addi x5, vvadd
  vadd vs0,x6,x5    vsetcfg x6,1,0,0,1
  sd vs0,0(x12)     li x6,4 ; vsetvl x6,x6   ; ← 向量长度是常数 4
  vstop             vf 0(x5)
```
- 访存没变成 `vld/vsd`，寄存器类全错（ld 到标量 x 寄存器、vadd 目的是 vs0 即常数 0）。
  原因：IR pass 把 `base+8*id` 拆走后，kernel 里的 load 地址变成了新增的 byval 参数，
  Scalarization 因此判它"uniform"；而参数又没按 `CC_RISCVXhwacha` 进 va 寄存器
  （byval 只加在了调用点，没加到新函数的形参上），MachOpt 的"从 va 寄存器 COPY"路径根本没触发。
- 控制线程侧没有 stripmine 循环，`vsetvl` 是硬编码常量。
- esp 汇编器拒绝这段输出。

**saxpy（float）**：MachOpt 里 `Unable to handle Opcode` VEIDX/VPSET 后断言失败，直接崩。
f32 路径比 i64 更不完整。

## 结论

老代码能证明"OpenCL SPMD → vf 块"这条流水线的**骨架**是通的：kernel 识别、地址 hoist、
uniformity 分析、控制线程发 vsetcfg/vf，每一步都有对应代码。但没有一个环节是完成的，
把它修到能出正确代码，等于把 MachOpt 的寄存器类逻辑、参数传递、向量长度管理都重写一遍，
还得先解决 LLVM 3.9 的 RISC-V 后端和主线完全不同这个包袱。
**结论：当参考读，不当基础改。** 值得移植的是设计（三层 pass 划分、Scalarization 用 PDG 做
控制依赖传播、`isVariant` 作为 lane-变化的种子），不是代码。

---

# SPMD → Hwacha 映射规格（2026-09-09）：scratch/spmd-spec/

三个 OpenCL kernel（kernels.cl）按下面五条规则手工翻译成 Hwacha 代码（kernels_hwacha.S），
C 宿主（main.c）用标量参考校验，N=5000，Spike 上 **3/3 通过**，stripmine 为 4×1024+904（i64 kernel 2×2048+904）。
`make run` 复现。

| kernel | 覆盖的映射点 | vf 块 |
|---|---|---|
| saxpy `y[i]=a*x[i]+y[i]` | 单位步长流、uniform 标量广播 | vlw,vlw,vfmadd.s,vsw |
| clamp_scale `if(x[i]>0) y=a*x else y=0` | 分歧控制流 → 谓词化 | vcmpflt.s vp1 → `@vp1 vfmul.s` / `@!vp1 vadd` |
| iota `c[i]=base+2*i` | get_global_id 当数据 | veidx + vs 偏移 |

规则（未来编译器的输出规格）：
- R1 以 `get_global_id(0)` 为索引、步长为常数的 `__global` 指针 → va 寄存器，每次 stripmine 前 `vmca`，之后指针前进 vl×步长。
- R2 uniform 的 kernel 参数 → vs 寄存器，循环外 `vmcs` 一次。
- R3 `get_global_id(0)` 用作数据时 = vs_offset + `veidx`，vs_offset 每次 stripmine 用 `vmcs` 重发。
- R4 逐元素条件的 if/else → `vcmp*` 写 vp，两个分支分别用 `@vp` / `@!vp` 谓词化；只有分支很长才值得用 `vcjal` 做 consensual 跳过。
- R5 `vsetcfg` 按实际用到的寄存器数填 (#v64, #v32, #v16, #vp)；用得越少 vlen 越长。

踩到的 ISA 细节：工作线程的立即数形式（`vaddi`、`vslli`）**目的只能是 vs 寄存器**，向量寄存器要用寄存器-寄存器形式
（2*i 写成 `vadd vv0,vv0,vv0`）。`vs0` 恒为 0，作为 float 就是 +0.0f，可直接用于比较。

---

# hwacha-cc 阶段 1：分析（2026-09-10）

目录 `hwacha-cc/`，基于 conda-forge 的 LLVM 23.1.1（llvmdev/clangdev，无需自己编 LLVM）。
`build/hwacha-cc kernels.ll` 打印分析报告。测试输入 `test/kernels.cl`（三个规格 kernel）、`test/kernels2.cl`（难例）。

## 前端

clang 23 直接编 OpenCL C 到 riscv64 IR：
`clang -x cl -cl-std=CL1.2 -Xclang -finclude-default-header --target=riscv64-unknown-elf -O2 -fno-vectorize -fno-slp-vectorize -fno-unroll-loops -emit-llvm -S`
- kernel 识别：函数带 `!kernel_arg_addr_space` 元数据（`opencl.kernels` 早已废弃）；跳过 clang 生成的 `__clang_ocl_kern_imp_*` 副本。
- `get_global_id(0)` 是外部调用 `_Z13get_global_idj`，分析阶段直接认它，不改 clang。
- 简单 if/else 在 -O2 已被折成 `select`；带 store 的分支保留为 CFG。

## 一致性分析（src/Analysis.cpp）

用 LLVM 自带的 `UniformityInfo`。LLVM 23 的接口：TTI 子类重写 `getValueUniformity()`，对 id 调用返回 `NeverUniform`，
`hasBranchDivergence()` 返回 true；构造 `UniformityInfo(DT, CI, &TTI)` 后必须调 `compute()`；查询用 `isUniformAtDef` /
`hasDivergentTerminator`。需要 `#include "llvm/ADT/GenericUniformityImpl.h"`。（`isSourceOfDivergence` 已从 TTI 移除。）

结果和手写规格一致：参数 uniform（→vs），id 派生值 divergent（→vv）。
值得注意的一条：分歧 trip count 的循环里，归纳变量和累加器在循环体内仍判为 **uniform**（所有活跃 lane 共享同一个 k），
只有循环出口的 phi 是 divergent（时间分歧）。对 Hwacha 意味着循环体可以用 vs 寄存器，出口处用谓词化的 move 把
各 lane 退出时的值收进 vv。

## 地址分类

给 id 调用挂 `!range [0, 2^30)` 元数据，SCEV 就能把 `int i = get_global_id(0)` 产生的 trunc/sext 折掉。
然后对每个访存指针的 SCEV S 做"求导"：Stride = S[id:=id+1] − S[id]，Base = S[id:=0]，两者都不含 id 才是线性的。
比结构匹配通用，能处理 AddRec（均匀循环里的滑动窗口 `x[i+k]` → base 是 `{x,+,4}`，stride 4）。

| 例子 | 分类 |
|---|---|
| `x[i]` | STREAM stride=4 base=x |
| `c[i]`（i64） | STREAM stride=8 |
| `x[i+k]`（k 均匀循环） | STREAM stride=4 base={x,+,4}：基址在 vf 块内随循环变，**不能靠 va 寄存器更新**（vf 块内 va 只读），代码生成要用索引访存或 vla 类指令 |
| `tab[idx[i]]` | GATHER |
| `x[k]`（k 均匀） | UNIFORM（标量 load 进 vs） |
| `a[i*ld]` | 目前 GATHER。32 位 `i*ld` 有 nsw 但 SCEV 不肯拆 sext(mul)，识别不出运行时步长；已知限制，Hwacha 有 strided load 可用 |

## 下一步（阶段 2：代码生成）

输入：分析结果 + IR。输出：vf 块汇编 + 控制线程 C/汇编。先做直线代码（saxpy、iota），再做 select/谓词，再做 CFG 结构化。

---

# hwacha-cc 阶段 2a：直线代码生成跑通（2026-09-10）

`hwacha-cc kernels.ll -o kernels.s` 输出一个 .s，含每个 kernel 的 `<name>_wt`（vf 块）和 `<name>_ct(long n, args...)`
（方案 A：C 可调用的控制线程函数，内部 stripmine）。端到端测试在 `hwacha-cc/test/run/`，`make run`。

## 结构（src/CodeGen.cpp）

- **工作线程**：自己的发射器直接产生汇编文本。地址先在 IR 级物化（stream → va 编号；gather → 均匀基址 + 用 SCEVExpander
  展开的分歧字节偏移），然后逐条指令选择：按 divergent/uniform 分 vv/vs，i1 分 vp；后缀由汇编器自动选变体；
  i32 用 w 后缀；select → `vcmp` + `@vp`/`@!vp` 两条谓词化 move；比较的取反用 `vpop` 真值表 0x55；
  常量和均匀参数统一由控制线程 `vmcs` 送进 vs（"vs 里的常量池"）。寄存器分配是按最后使用位置释放的线性扫描。
- **控制线程**：生成为普通 LLVM IR 函数（stripmine 循环、基址 phi、按 vl×stride 前进），Hwacha 控制指令是
  `asm sideeffect`，交给 llc（`-no-integrated-as --code-model=medium`）编成 RISC-V 汇编，去掉 `.attribute/.option`、
  `fmv.x.w→fmv.x.s` 后与 vf 块拼在一起，esp binutils 2.29 能直接汇编。主线 RISC-V 后端一行没改。

## 结果

| kernel | 测试 | 生成的 vf 块要点 |
|---|---|---|
| saxpy / clamp_scale / iota | spec.cl，N=5000，3/3 PASS | 与手写规格逐条一致 |
| gather `tab[idx[i]]` | gather.cl，3/3 PASS | `vlxw vv, vs_base, vv_idx` |
| scatter `out[idx[i]]=…` | 同上 | `vsxw` |
| mixed（kernel 内均匀标量 load + select） | 同上 | `vlsw vs, vs_addr`，其余同 clamp_scale |

## 踩的坑

- llc 的集成汇编器会校验内联汇编，不认 Hwacha 指令 → `-no-integrated-as`。
- 默认 medlow 代码模型用 `lui/addi` 绝对地址，够不到 0x80000000 → `--code-model=medium`。
- 无操作转换（sext/trunc/bitcast）和源共用寄存器时，释放转换值不能释放寄存器（`Alias` 集合）。

## 未支持（明确报错）

带分支/循环的 kernel（"kernels with control flow are not supported yet"）、非常数步长、循环携带的流基址、
i1→整数转换。下一步：阶段 2b，CFG 结构化 + 谓词化，先过 cond_store，再过 divloop2 / uniform_loop。

---

# hwacha-cc 阶段 2b：控制流（2026-09-10）

Karrenberg 式谓词化线性化，实现在 `src/CodeGen.cpp` 的 `linearize / computePositions / emitBlock / beginLoop / endLoop / handleEdge`。
测试 `test/run/cf.cl`（cond_store、nested_if、divloop、uloop、divloop_vv），N=2500，**5/5 通过**；spec/gather 回归通过。

## 方法

- 前置：`LoopSimplify + LCSSA`（每个循环有 preheader 和唯一 latch；循环内定义、循环外使用的值都经过出口块的 LCSSA phi）。
- **线性化**：拓扑序摊平无环区域，循环作为整体（`LoopBegin … LoopEnd`）嵌在外层顺序里，递归处理嵌套循环。
- **块谓词**：入口块 vp0；其他块 = 各入边掩码的 OR。条件分支的两条边：`vpop e, P, c, c, 0x80`（P&c）和 `0x02`（P&!c），
  真值表按 (a,b,c=b) 的索引算，c==b 时只有 0/1/6/7 四种索引会出现。
- **phi** → 各入边掩码下的谓词化 move（vv 用 `@m vadd d, v, vs0`，vp 用 `@m vpop d, v, v, v, 0xAA`）。
- **循环**：活跃掩码 A（进入时复制 preheader 边掩码，之后**原地**更新）；每条出口边 m：LCSSA phi 在此处捕获
  `@m vadd cap, v, vs0`，出口累加器 `T |= m`（循环前 `vpclear`），`A &= !m`；latch 后头 phi 用回边掩码 move，
  `@back vcjal 1, vs_link, .Lhead`（c=1 即 any：还有活跃 lane 就回跳）。出口块谓词 = 对应累加器。
- 分歧循环里的均匀值（k、s）留在 vs：所有活跃 lane 共享；退出的 lane 在出口边被捕获进 vv。这就是"时间分歧"的实现。
- 谓词只加在向量目的和访存指令上；vs 目的的标量指令不加（Spike 里标量写不看谓词）。

## 踩的坑

- 分支条件寄存器在算边掩码前就按 LastUse 释放了，导致目的和源同名 → 终结指令的释放推迟到边掩码算完。
- 循环头 phi 的 LastUse 必须延到 LoopEnd（latch 处还要写它）。
- 常量/参数的 vs 寄存器曾是懒分配的，在循环里会拿到前面临时值释放的寄存器，而那条写指令每次迭代都执行 → 改为开头预分配。
- 两个操作数都是标量的比较，Spike 只执行一次、只写 lane 0 的谓词（`v_is_scalar`：目的标志为 0 就当标量）→ 先把一个操作数广播成 vv。
- 出口累加器若在首次遇到出口边时才分配并"复制"，那条复制每轮都执行，会覆盖之前退出的 lane → 循环前分配清零，循环内 OR。
- `vfneg.s d, s` 汇编器不收，用 `vfsgnjn.s d, s, s`；fabs 用 `vfsgnjx`。

## 仍未支持

循环携带的流基址（均匀循环里的 `x[i+k]`；va 在 vf 块内只读，需索引访存 + 递增偏移）、非常数步长、i1→整数、
块内无活跃 lane 时的均匀 store 仍会执行、均匀分支尚未用 `vcjal 0` 跳过（现在一律摊平）。

---

# hwacha-cc 阶段 2c：循环携带的流基址、运行时步长（2026-09-10）

va 寄存器在 vf 块内只读且不带步长，所以基址在块内推进的流（`x[i+k]`）和运行时步长（`a[i*ld]`）不能用 va。
统一降成**索引访存**：`vlx*/vsx* vv, vs_base, vv_offset`。地址 SCEV 用 `splitUniform` 递归拆成均匀部分 U 和分歧部分 D：
加法逐项拆；AddRec 把起点和步长分别拆，U 成为块内的标量归纳变量（`{A,+,4}` → vs 里每轮加 4），D 是分歧偏移。
两者都用 SCEVExpander 在访存点展开，展开后重新跑一次一致性分析（新指令否则会被当成分歧值）。
gather 也走同一条路，之前 `a[i*ld]` 退化成 gather 的限制随之消失。

测试 `test/run/lc.cl`：window（滑动窗口）、strided、stencil（常数偏移仍是普通流）、matvec（行步长 n 是运行时值，`v[j]` 是均匀 load），
**4/4 通过**；全部 17 个 kernel 回归通过（`make run`）。

坑：索引访存的基址/偏移不是 load 的 IR 操作数，活跃区间计算必须把它们当隐式使用，否则偏移寄存器在循环前就被 phi 复用。

---

# hwacha-cc 阶段 2d：空块跳过、`__local` 内存与 barrier（2026-09-11）

## 空块跳过
任何块只要主体 ≥2 条指令且谓词不是 vp0，就在主体前发 `@!P vcjal 0, vs_link, .Lskip`（c=0 即 all：所有 lane 都不活跃才跳），
跳到边掩码计算之前。均匀分支自然受益（未走的分支整段跳过），分歧分支在某次 stripmine 恰好没人走时也跳。

## OpenCL 工作组 → Hwacha stripmine
- **工作组 = 一次 stripmine 的 vl 个元素**。`get_local_id(0)` = `veidx`；`get_local_size(0)` = vl，`get_group_id(0)` = 迭代序号，
  两者由控制线程每轮 `vmcs` 送入；`get_global_id(0)` 仍是偏移 + veidx。
- `reqd_work_group_size(X,1,1)` → 控制线程每轮请求 `vsetvl min(剩余, X)`；要求 n 是 X 的倍数（OpenCL 本来就要求）。
  没有该属性时组大小就是硬件给的 vl（例如 Spike 上 3 个 vv 寄存器 → 680），kernel 必须只依赖 get_local_size。
- **`__local` 数组**在 clang 输出里就是内部全局变量。控制线程模块里建一个同形状的静态缓冲区（组是顺序执行的，一个就够），
  地址 `vmcs` 进 vs。以 local id 为索引的访存是"局部流"（`LSTREAM`）：仍用 va，但控制线程**不随迭代推进基址**。
  分析里 `decompose` 现在分别对全局 id 和局部 id 求导，两者都出现的地址当 gather。
- **`barrier()` → `vfence`**。vf 块内指令本来就是按序对所有 lane 执行的，barrier 只需保证访存顺序。
- 标量 store `vss{w,d}` 的操作数顺序是 **地址在前、值在后**（`#S,#T`），和向量 store 相反；之前写反了会写到值当地址的位置。

测试 `test/run/local.cl`：group_sum（256 元素工作组的树形归约，barrier 在均匀循环里、`if (lid < s)` 分歧）、
shift_left（动态组大小 + `% get_local_size`），**2/2 通过**；全部 19 个 kernel 回归通过。

## 编译器现状小结
覆盖：流 / gather / scatter / 广播标量 / select / 分歧 if / 分歧与均匀循环 / 循环携带的流基址 / 运行时步长 /
局部内存 / barrier / 工作组查询 / 空块跳过。剩余空白主要是性能层面：寄存器分配没有考虑用量对 vlen 的影响、
均匀比较的广播开销、以及所有这些在真 RTL 上的周期数。

---

# hwacha-cc 阶段 3a：寄存器用量与统计（2026-09-11）

`hwacha-cc x.ll --kstats` 打印每个 kernel 的 vv64/vv32/vs/vp 用量、按 Spike 模型（8×256/寄存器数，vp 为 8×1024/数）算的 maxvl、
静态指令数、掩码/move/跳转数。用来对照手写版本。

三项改动：
1. **目的寄存器复用**：单条指令先读源后写目的，所以目的可以直接用在本指令最后一次使用的同类操作数寄存器。
   saxpy/stencil 的 vv 从 3 降到 2，与手写一致。坑：被接管的操作数不能再按 LastUse 释放（`Transferred` 集合）。
2. **循环头 phi 合并**：latch 传入值若在循环体内定义、且 phi 在其定义点之后不再被读（`PhiRealLast`，在延长到 LoopEnd 之前记录），
   就直接写进 phi 的寄存器，latch 处的 move 消失。写操作本来就在活跃掩码下，退出的 lane 不受影响。
3. **32 位寄存器类 `VW`**：≤32 位的分歧值单独一个池，编号排在 64 位寄存器之后，vsetcfg 的 #v32 字段填其数量，move 用 `vaddw`。
   Spike 的 vsetcfg 把各宽度寄存器数直接相加，看不到 vlen 收益；RTL（util-confprec）一个 64 位表项放两个单精度值，
   f32 kernel 的 vlen 应翻倍。saxpy 现在是 vv64=0 vv32=2。

全部 19 个 kernel 回归通过。含均匀访存的块现在只要有一条指令也加跳过，避免无活跃 lane 时执行标量 store。

---

# hwacha-cc 阶段 3b：窄位宽、双精度、原子（2026-09-11）

- **零扩展 load**：窄整数 load 的所有使用者都是 `zext` 时发 `vl{b,h,w}u`，zext 变成别名；`sext` 本来就是默认的符号扩展 load。
- **双精度**：`vld/vsd` + `.d` 浮点变体，kernel 里要 `#pragma OPENCL EXTENSION cl_khr_fp64 : enable`，clang 加 `-cl-ext=+cl_khr_fp64`。
- **原子操作**：riscv 目标上 clang 把 OpenCL 1.2 的 `atomic_add` 等编成库调用 `_Z10atomic_addPU8CLglobalVii(ptr, i32)`，
  按名字映射到 `vamo{add,min,max,and,or,xor,swap}.{w,d}`（sub/dec 用取负后的 add；unsigned 由 mangling 里的 `Vjj/Vmm` 判断）。
  语法 `vamoadd.w vd, 0(vv_addr), src`，地址是逐 lane 的向量寄存器，正好是 GEP 降低出来的分歧指针。
  原子调用即使结果不用也是 `Needed` 的根。
- 字节 store：`vsb` 取低 8 位，i8 运算直接用 64 位 `vadd`。

测试 `test/run/misc.cl`：hist（uchar load + `atomic_add` 直方图）、widen（sext/zext 的 short load 乘成 long）、dbl、bytes，**4/4 通过**；
全部 23 个 kernel 回归通过。

---

# RTL 环境：Chipyard 1.11.0 + Hwacha（2026-09-11）

目录 `chipyard/`（`git clone --depth 1 -b 1.11.0`，1.12 起 Hwacha 被移除）。无 sudo，全部依赖来自 Chipyard 自带的 conda 环境。

## 安装（两段，因为脚本的第 1 步在精简环境下有个坑）
1. `echo y | ./build-setup.sh esp-tools --use-lean-conda --skip-toolchain --skip-ctags --skip-firesim --skip-marshal`
   第 1 步用 conda-lock 建出 `.conda-env`（gcc 11、verilator 5.020、sbt、openjdk 20、dtc，**以及 ucb-bar 频道的 `esp-tools` 包：
   带 xhwacha 的 GCC 9.2、Spike，装在 `.conda-env/esp-tools/`**），但精简环境里没有 conda 自身，脚本 `source .conda-env/etc/profile.d/conda.sh` 失败退出。
2. 手动激活后重跑剩余步骤：`conda activate $PWD/.conda-env; export RISCV=$PWD/.conda-env/esp-tools;
   echo y | ./build-setup.sh esp-tools --use-lean-conda --skip-conda --skip-toolchain --skip-ctags --skip-firesim --skip-marshal`
   → 子模块、sbt 预编译、CIRCT（firtool 装到 $RISCV/bin）。约 2 GB conda 环境 + 子模块。

## 使用
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ~/hwacha-compiler/chipyard/.conda-env
source ~/hwacha-compiler/chipyard/env.sh; export RISCV=~/hwacha-compiler/chipyard/.conda-env/esp-tools
cd chipyard/sims/verilator && make CONFIG=HwachaRocketConfig -j8      # 首次几十分钟
./simulator-chipyard-HwachaRocketConfig +max-cycles=200000000 prog.riscv
```
esp-tools 的 GCC 9.2 直接接受 `-march=rv64gcxhwacha`，benchmark（`hwacha-cc/test/bench`）改用它编译；
`riscv64-unknown-elf-gcc-xhwacha` 包装脚本只在用自建 GCC 7.2 时需要。
esp-tests 的精简 printf 不支持 `%-22s`、`%f` 这类格式，可变参数会错位导致 `%s` 读到垃圾指针。

## 仿真器构建的坑（2026-09-11）
- 跳过第 3 步后没有 fesvr（主机侧 HTIF 库）。`toolchains/esp-tools/riscv-isa-sim`（2021 版）的 fesvr 接口太旧
  （`memif_endianness_t`），Chipyard 1.11 的 SimDRAM.cc 需要 `endianness_t`；用 `toolchains/riscv-tools/riscv-isa-sim`（上游）
  编出来的 fesvr 头文件和 `libfesvr.a` 装进 `$RISCV`（它的 spike 覆盖了 esp-tools 的 spike，无所谓，功能验证用自建的带 hwacha 的 Spike）。
- 15 GB 内存下，Verilator 生成的 C++ 用 `-j8` 编会被杀；Spike 的几个二进制并行链接也会。最终：Spike `-j1`，
  仿真器 `SIM_OPT_CXXFLAGS=-O1 -j2`，并用 `setsid nohup` 脱离会话跑。
- 运行：`./simulator-chipyard.harness-HwachaRocketConfig +permissive +max-cycles=N +permissive-off prog.riscv`
  （二进制名里有 `.harness`；不加 `+permissive` 时 `+max-cycles` 会被当成文件名）。
- **速度：约 2.7k 周期/秒**（-O1 模型）。hello 约 55 s（含 TSI 加载和启动），12 万周期的循环约 45 s。
  benchmark 里的标量参考循环是大头，N=4096 一轮约 25 分钟；消融用 N=1024。

## RTL 上 Hwacha 取指失败的排查（2026-09-11）
现象：任何含 vf 的程序（hwacha-cc 生成的、手写的 spmd-spec、官方 vec-saxpy、rv64uv 单元测试）在 Verilator 上都触发
`scalar-unit.scala:296 assert(... id_ctrl.ival, "illegal instruction exception!")`；Chipyard 自己的 `make run-binary` 同样。
排除项：指令编码（与 instructions.scala 的 BitPat 逐条匹配）、vsetcfg 值（RTL 是 `imm12 | rs1`，寄存器低位有效）、
vf 地址、Verilator 寄存器初始化策略（`+verilator+rand+reset+0/1` 结果相同）、内存模型（`+dramsim` 同样失败）。
`+verbose` 日志：vf 发出后向量单元前端第一次响应就是异常，Rocket 不再执行任何指令。
怀疑：Hwacha 前端（`frontend.scala`）创建 rocket-chip 的 TLB 后没有给 `tlb.io.req.bits.prv/cmd/v` 赋值——这些是新版
rocket-chip 的 TLBReq 字段（TLB 用 `io.req.bits.prv` 而不是 status 判断特权级）。未连接 → prv=0（U 态）→ PMP/PMA 检查
按用户态执行。修改：`prv := RegEnable(req.bits.status.prv, req.valid)`、`cmd := M_XRD`、`v := false`，并加了
`H: ITLB …` / `H: IFETCH …` 两条 printf（仅 `+verbose` 时打印）。重新生成仿真器验证中。

### 根因（2026-09-11）
TLB 修改后 `+verbose` 日志显示翻译正常，但 **取回的数据偏了 16 字节**（请求 0x308 拿到 0x318 的指令）。
rocket-chip ICache 的数据阵列按 `refillCycles = cacheDataBeats = blockBytes*8/rowBits` 索引行，Hwacha 的
`HwachaIcacheKey.rowBits = 64` → 8 行/块；而 TileLink 回填实际是 128 位一拍（Chipyard 1.11 把 sbus 宽度和加速器解耦，
CHANGELOG "Decoupled sbus width from boom|hwacha|gemmini"），写入按拍号 0..3、读取按 8 字节行号 0..7，错位。
修复：`rowBits = 128`（与 sbus 一致），Hwacha 前端本就有 rowBytes > fetchBytes 时按 PC 选字的逻辑。
TLB 的 prv/cmd/v 赋值保留（正确性修复，之前未连接时 prv=0）。

### 最终修复（2026-09-11，`generators/hwacha` 本地补丁，`git diff` 可见）
1. `configs.scala`：`HwachaIcacheKey.rowBits = 128`（与 sbus 一致，ICache 回填/读取行索引才一致）。
2. `frontend.scala`：新版 ICache 只返回请求的 8 字节，不是整行 → `MiniFrontend` 去掉"同一行复用"（`s0_same_block := false`）
   和按 PC 的移位（`fetch_data := deq.bits`）；否则奇数字被移成 0。
3. `frontend.scala`：给 TLB 请求补上 `prv/cmd/v`（新版 TLBReq 字段），并加 `H: ITLB/IFETCH` 跟踪 printf（仅 `+verbose`）。
验证：rv64uv-p-vvadd_d、vcjal 在 Verilator 上通过（tohost=1，$finish）。这说明 Chipyard 1.11.0 自带的 Hwacha 在 128 位 sbus 下
本来就是坏的（CI 的 hwacha 组显然没有真正跑过或没被当回事），1.12 移除它大概也与此有关。

### 第二个 RTL 问题：vf 块内的标量访存（2026-09-11）
修好取指后，直线 kernel 在 RTL 上跑通并有了数字，但 divloop 挂死：取指日志显示前端反复重取 `vlsw` 之后那条指令，
`H: write_srf` 显示 `vlsw` 的地址算出来了、数据却永远没回来 → 记分板不放行。原因同 TLB 那类：`smu.scala` 的 TLB 请求
没设 `prv/v`（`vmu-addr.scala` 已设 `prv := status.dprv`，所以向量访存正常），benchmark 的 crt.S 又不配置 PMP，
用户态访问被拒。修复：SMU 加 `prv := req.status.dprv; v := req.status.dv`。
顺带发现：以 vs 为目的的浮点运算（`vfadd.s.ss`）走共享 FPU，在这个配置下也会挂（先被误判为根因）。编译器已改为
所有需要 FPU 的运算一律进向量寄存器（`needsFPU`，`--scalar-fp` 可关），这是 Spike 看不到的 RTL 约束。

### 首批 RTL 数据（修 SMU 之前，只有直线 kernel）
N=4096：saxpy 标量 55952 周期 / hwacha-cc 3735 / 手写 4016（13.66 → 0.91 周期/元素，15×）；
clamp 标量 74405 / hwacha-cc 4060 / 手写 3314。N=1024：saxpy 1516 vs 手写 1332，clamp 1267 vs 手写 990。
clamp 差距来自 select 的两条 move；已加优化：只有一个使用者且使用者是同块 select 的指令，直接在 select 条件下
写进 select 的寄存器（现在 clamp_scale 与手写逐条一致）。消融变体在这些 kernel 上无差异（没有循环、vlen 不是瓶颈）。

### 第三个 RTL 问题：consensual 跳转的谓词归约（2026-09-11）
SMU 修好后 divloop 的循环能跑，但停不下来：退休日志显示稳态每轮只执行"循环头跳过跳转（被采纳，即活跃掩码已空）→
k 更新 → 回跳（被采纳）"三条标量指令，即 `any(vp6)` 在 vp6 应为空时仍判真。RTL 的 `vpop` 真值表索引与 Spike 一致，
排除编码问题。归约在 `vfu-rpred.scala` 的 `RPredLane`：`when(io.op.valid){cond := init}` 之后紧跟
`when(io.req.fire){cond := cond & / | ...}`，同一拍两者都成立时后者赢，累加从上一次跳转的结果开始，
上一次为真则永远为真。已改为"新 op 先复位再累加"的写法，重建仿真器验证中。
复位竞争修了仍然死循环。再读 `RPredLane`：ALL 归约写的是 `cond & (pred | ~active).orR`——`.orR` 让"所有 lane"退化成
"任一 lane"，于是循环头的 `@!vp1 vcjal 0`（all）在**第一个 lane 退出**时就成立，跳过整个循环体（含出口条件的比较），
其余仍活跃的 lane 永远得不到更新，回跳的 `any(vp6)` 永远为真。这与日志里"循环体只执行十几次、尾部执行几千次"完全吻合。
修复：`.orR` → `.andR`（master 侧跨 lane 的组合本来就是 AND）。Spike 的 `cond_all` 是正确的 all，所以 Spike 上从未暴露。
这也是编译器用 `@!P vcjal 0` 做块跳过在 RTL 上必踩的坑；手写 benchmark 不用 all 归约所以没发现。

### 验证与补丁汇总（2026-09-11）
ALL 归约修复后 divloop 小程序在 RTL 上结果全对。**全部 RTL 补丁导出在 `hwacha-rtl-fixes.patch`**（对 `generators/hwacha`，
Chipyard 1.11.0 pinned commit bf799dc）：
1. `configs.scala` icache `rowBits = 128`（sbus 宽度解耦后取指数据错位 16 字节）；
2. `frontend.scala` 不复用整行、不移位（新版 ICache 只返回请求的 8 字节）；TLB 请求补 `prv/cmd/v`；`+verbose` 取指跟踪；
3. `smu.scala` TLB 请求补 `prv/v`（标量访存挂死）；
4. `vfu-rpred.scala` ALL 归约 `.orR`→`.andR`（all 退化成 any）；op 复位与首拍累加的同拍竞争。
编译器侧对应的 RTL 约束：需要 FPU 的运算不能以 vs 为目的（`needsFPU`）。

---

# RTL 性能结果（Chipyard 1.11.0 HwachaRocketConfig + 本地补丁，Verilator，2026-09-11）

`hwacha-cc/test/bench`，`make N=1024 all` / `./run-all.sh 1024`，结果在 results/。周期数来自 `rdcycle`，包住整个 `_ct` 调用。

## N=1024（单次 stripmine 即可装下）

| kernel | Rocket 标量 C | hwacha-cc | 手写汇编 | 加速 |
|---|---|---|---|---|
| saxpy | 12945 (12.6/元素) | 1515 (1.47) | 1363 (1.33) | 8.5× |
| clamp_scale（select） | 17618 (17.2) | 1004 (0.98) | 1026 (1.00) | 17.5× |
| divloop（分歧循环，trip 0..15） | 68924 (67.3) | 19582 (19.1) | — | 3.5× |
| stencil（3 点） | 20594 (20.1) | 1787 (1.74) | — | 11.5× |
| gather（随机置换） | 12963 (12.7) | 2217 (2.16) | — | 5.8× |

- clamp_scale 经 select 优化后与手写持平（之前 1267）。saxpy 比手写慢 11%，vf 块逐字相同，差在控制线程（待查）。
- **消融**：`--no-v32`、`--no-coalesce` 在这些 kernel 上无差别（N=1024 一次装下，vlen 不是瓶颈；divloop 的累加器进了 vv 后
  合并不起作用）。**`--no-skip` 让 divloop 从 31158 降到 19158**：consensual 跳转要等谓词归约（约 50 周期，标量单元停等），
  循环头的跳过几乎从不成立，是纯开销。已改为循环头不发跳过（divloop → 19582），其它块保留。
- 仿真速度：约 2.7k 周期/秒（Verilator -O1），N=1024 一轮约 20 分钟，5 个并行。

## N=4096（4 次 stripmine + 尾块；divloop 为循环头仍有跳过的旧版本）

| kernel | Rocket 标量 C | hwacha-cc | 手写汇编 | 加速 |
|---|---|---|---|---|
| saxpy | 55928 (13.7/元素) | 3741 (0.91) | 3935 (0.96) | 15× |
| clamp_scale | 74557 (18.2) | 3104 (0.75) | 3287 (0.80) | 24× |
| divloop | 300205 (73.3) | 123134 (30.1，旧版) | — | 2.4×（新版按 N=1024 推算约 3.8×） |
| stencil | 85752 (20.9) | 6085 (1.48) | — | 14× |
| gather | 65970 (16.1) | 7534 (1.83) | — | 8.8× |

多次 stripmine 时 hwacha-cc 的控制线程比手写略快（saxpy 3741 vs 3935）：llc 生成的循环把 vf 地址和步长计算提到了循环外。
每元素周期数随 N 增大而下降，说明单次 vf 的启动/排空开销约 1000 周期量级，是短向量的主要成本。

---

# 真实程序：Rodinia OpenCL kernel（2026-09-12）

`hwacha-cc/test/apps/`：nn、kmeans（两个 kernel）、bfs（两个）、streamcluster pgain、pathfinder，kernel 源码取自 Rodinia 的
OpenCL 版本（`rodinia/` 原文件，**一字未改**），宿主程序自写（数据生成、标量参考、校验、rdcycle）。`make all; make spike`。
五个程序在 Spike 上全部通过。

## 为此补的功能
- OpenCL 数学库调用（`_Z4sqrtf` 等 → `vfsqrt`；fabs/fmin/fmax）；`llvm.smin/smax/umin/umax` → 比较 + 谓词 move。
- **非单位步长的流**（结构体数组 `d_locations[i].lat`，步长 8）→ `vlst*/vsst*`，步长放在额外的 va 寄存器里，控制线程 `vmca` 一次。
- scatter 的源是 uniform 值时先广播。
- `__local` **指针参数**（pgain 的 `coord_s`、pathfinder 的 `prev/result`）：clang 输出为普通指针参数，宿主传一块每组复用的缓冲区。
- 宿主可设的工作组大小 `hwacha_group_size`（依赖 `get_local_size` 的 kernel 必须用），以及 `hwacha_vl_short` 标志：
  kernel 的寄存器用量决定硬件最大向量长度，请求的组大小超过它时静默出错——pathfinder 用 9 个向量寄存器，maxvl 只有 224，
  BLOCK=256 就不行（改 128）。这和 GPU 的 occupancy 限制同源。
- i1 → 整数（`zext/sext i1`）。

## 三个真正的编译器 bug（都是 Rodinia 才暴露的）
1. **发射期间才合成的常量**（zext 的移位量 32、GEP 比例）懒分配到 vs 时可能拿到块里更早的临时值用过的寄存器，控制线程在 vf
   之前就把常量送进去，块里先被临时值覆盖。修复：输入类（参数、常量、特殊调用）从 vs 池**高端向下**分配，临时值从低端向上。
2. **均匀值的 select / 汇合块 phi / 整数 min-max 用谓词化 move 合成**，但目的是 vs 时谓词被忽略，两条 move 都执行。
   修复：这类值一律进向量寄存器；并且**向量性沿数据流传播**（`computeClasses` 不动点）：任何操作数是向量的运算结果也是向量，
   地址变成向量的均匀 load/store 改成基址 vs0 的索引访存。
3. **谓词逻辑指令不受谓词控制**：Spike 的 `vpop` 用 `WRITE_PPR_NO_PRED`，RTL 的 `vipred` 发射不带 vp 操作数。
   我用"带谓词的 vpop 复制"做 i1 的 select、phi move、LCSSA 捕获，后一条永远覆盖前一条。修复：三输入真值表一次算出
   `D = M ? S : D`（`vpop D, M, S, D, 0xD8`），完全不用谓词。之前的测试全靠"先 false 后 true"的巧合通过。
另有一个隐患：谓词 select 的目的寄存器不能复用条件寄存器。

## 已知限制
去掉 barrier 的 pathfinder 变体让 clang 生成的 CFG 需要 >15 个 vp 寄存器（掩码没有溢出机制）。

## Rodinia 在 RTL 上的周期数（Verilator，本地补丁版 Hwacha）

| 程序 | 规模 | Rocket 标量 C | hwacha-cc | 加速 |
|---|---|---|---|---|
| nn（距离 + sqrt） | 2048 | 99464 (48.6/元素) | 15077 (7.4) | 6.6× |
| kmeans（swap + assign） | 1024 点 × 8 维 × 5 类 | 1348765 | 34303 + 130925 = 165228 | 8.2× |
| streamcluster pgain | 1024 × 8 维，组 256 | 330023 | 39254 | 8.4× |
| pathfinder | 8 行 × 1024 列，组 128 | 247489 | 143617 | 1.7× |
| bfs | 2048 节点，度 4，10 层 | 182333 | 645785 | **0.28×（慢 3.5 倍）** |

观察：pathfinder 的加速最低。它每个时间步只有两次迭代，却有三个 barrier（`vfence`）、七八个 consensual 跳转（每个约 50 周期）
和大量掩码运算，而有效计算只有几条 min 和加法；工作组之间还有 HALO 重叠。这类"同步密集、计算稀疏"的 kernel 在 vector-fetch 模型
上收益天然有限。nn 的 7.4 周期/元素主要是 `vfsqrt`（迭代实现）。

### bfs 在 RTL 上挂死的排查（2026-09-12）
逐项探针（都在 RTL 上通过）：字节/半字单位步长访存（全 lane）、字节 scatter/gather、标量 store、64 位乘法、双精度 FMA、
原子加、同一数组的 scatter→load 顺序（含循环）、稀疏谓词的字 load/store。**挂死的是稀疏谓词的字节单位步长 store**
（`@vp vsb`，只有一个 lane 活跃）——bfs 里 `g_graph_mask[tid]=false` 正是这种。VMU 的部分写路径（`vmu-memif.scala` 的
半拍掩码移位 + `PredicateByteMask`）是按 64 位拍设计的，128 位 sbus 下子字掩码可能出错导致请求永不完成。
编译器先做绕过（`--no-subword-rmw` 可关）：谓词化的子字单位步长 store 改为"load → 谓词 select → 无谓词 store"，
每个 lane 重写自己的字节，不活跃的写回刚读的值；对单位步长是安全的（每个地址只属于一个 lane），scatter 不能这样做（重复地址）。
探针细化（RTL）：字节单位步长 store 只有 lane 0 活跃 → 挂死；只有 lane 3 活跃 → 正常；隔一个 lane → 正常；
半字单 lane → 正常；字节 scatter 单 lane → 正常。条件相当特殊（首元素活跃、其后整段不活跃），应是 VMU 子字掩码/计数
的边角 bug。编译器绕过（load/select/store）对所有谓词化子字单位步长 store 生效，Spike 回归 36/36。

bfs 补记（2026-09-12）：绕过稀疏字节 store 后在 RTL 上通过（0 mismatches），但比标量**慢 3.5 倍**。原因不在编译器：
Rodinia 的 OpenCL bfs 是 GPU 式的层同步稠密扫描——每一层两个 kernel 各扫全部 2048 个节点，而只有 frontier 上的节点有活；
标量参考是工作量最优的队列 BFS。10 层 × 2 个 kernel × 13 次 stripmine，每次 vf 又有分歧循环和多个 consensual 跳转（各约 50 周期），
固定开销远超有效工作。这类 kernel 依赖 GPU 的海量线程来摊薄无效扫描，在单向量单元上是反例。

## llama2.c（stories260K）与"控制线程循环"（2026-09-12）

目标：在 Hwacha 上跑一个真正的 LLM 推理。模型 karpathy/tinyllamas stories260K（dim 64，hidden 172，5 层，8 头 / 4 kv 头，
词表 512，26 万参数），权重与 tok512 分词器用 `.incbin` 嵌进裸机二进制（`hwacha-cc/test/llama/`）。
宿主 `llama_main.c`：逐字移植 run.c 的 forward 作标量参考；hwacha 版把 forward 拆成 9 种 kernel（`llama.cl`）：
matmul（行主序权重，每 lane 走自己的一行，跨 lane 步长 n）、matmul_t（宿主转置一次权重，每个 j 各 lane 读连续一行，单位步长）、
rmsnorm（平方和这个 64 元素归约留在宿主）、rope（cos/sin 查宿主预算的表）、att_score（(h,t) 展平成一个 NDRange）、
att_softmax（每头一个 lane，vl=8）、att_value、residual、silu_mul。贪心采样，与 x86 原版 `run -t 0` 逐 token 对比。

### 向量化 expf（`hwacha_math.h`）
纯 OpenCL C 写的 hw_expf：clamp → k=round(x·log2e)（1.5·2^23 魔数，靠 RNE 加法）→ r=x−k·ln2（hi/lo 两段）→ 6 阶多项式（Cephes 系数）
→ as_float(as_int(p)+(k<<23))。编译器不需要新指令，只依赖 fmadd/cvt/移位/位转换（位转换在寄存器里是空操作）。
Spike 与 RTL 上对 libm 的最大相对误差 7e-8；RTL 上 1024 个元素 4871 周期（4.75 周期/元素）。
顺手修了一个潜伏 bug：C 的 float→int 是截断，但生成的 vfcvt.w.s 没写舍入模式（默认 dyn=RNE）；现在显式发 `rtz`。

### 编译器补的两个分析漏洞
1. `int i = get_global_id(0); w[i*n+j]`：clang 用 i32 算下标（trunc id → mul nsw → zext nneg），SCEV 推不过扩展，地址被判成 gather。
   新增前置变换 widenIdArithmetic：sext(nsw 运算) = nsw 运算(sext)，且 id < 2^30，把整条链重建成 i64。matmul 变成 stride=4·n 的流。
2. `h = gid/hs` 这种 SCEV 不建模的除法在 decompose 里被当成 id 无关的未知量，导致 att_value 的 v[t*kv_dim] 被误判成 stride 4 的流；
   现在 SCEVUnknown 若包装的是发散指令就算含 id。

### Spike 结果（40 token）
标量参考与 hwacha 版生成文本与 x86 完全一致："Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, r"。
teacher-forced 逐步比较 logits 最大差 1e-5，argmax 0 处不同。

### 控制线程循环（CT loop）
第一版生成的 matmul 循环体：2 条有效指令配 18 条循环机器（掩码、ExitTotal、vcmpeq、vpop 链）加一次约 50 周期的 consensual jump，
64 次迭代光跳转就 3200 周期。Hwacha 的本意是均匀控制流放在标量核：vf 块之间向量寄存器保留，控制线程可以自己循环、每次迭代发一个 vf。
实现（`--no-ct-loops` 关闭）：
- 条件：深度 1 的单块循环，退出条件均匀，preheader 只经均匀分支可达（reachUniform），循环里所有均匀指令能克隆到控制线程（无调用；
  均匀 load 前面不能有可能先执行的 store）。
- 向量侧：循环体成为独立的 vf 块（`<k>_wt_b<i>`），循环后的部分是 `<k>_wt_a<i>`；体内均匀指令（归纳变量、地址、均匀 load、退出比较）
  不发射，只给被向量指令读的那些分配 vs 寄存器（IterInputs，寄存器钉住到循环结束）；无 Active 掩码、无 vcjal；LCSSA phi 在循环后统一拷贝。
- 控制线程：把循环的均匀切片克隆成标量 LLVM IR（header phi → phi，LCSSA → 直接映射，get_local_size/group_id → vl/group），
  每次迭代 `vmcs` IterInputs、`vmca` 随循环推进的流基址（base + offset·stride，stride 可以是运行期均匀值）、`vf body`。
  循环是否执行由控制线程照着均匀 CFG 重新求 reached(preheader)。
- 随循环推进的流基址在 header 里用 SCEVExpander 展开成均匀指令，由控制线程求值：matmul_t 的循环体变成
  `vlw vv1, va0; vfmadd.s; vaddw`（单位步长）；matmul 是 `vlstw vv1, va0, va2`（运行期步长）；att_softmax/att_value 的
  循环也都成了 CT loop。Rodinia 五个程序的内层循环都是嵌套的（深度 2），暂未受益。
- 踩坑：SCEV 持有 KernelAnalysis 的 DominatorTree 引用，recomputeUniformity 原来 make_unique 重建 DT，之后 SCEVExpander 崩溃/乱选值，
  改成 DT->recalculate；Expander 新建的指令要重算 uniformity 才能判均匀；IterInputs 的 vs 寄存器不能按最后使用释放（所有 vmcs 在 vf 前
  一起发）；均匀 i1（退出比较）只有在没被向量指令读时才留在控制线程。
回归：36 个 kernel、5 个 Rodinia、llama 全 PASS。RTL 对比（loop-in-vf vs CT loop，各 4 token，`+loadmem` 后门装载 1.1MB 映像）运行中。

### 推广：控制线程区域（同日晚些时候）
单块循环的版本推广成"区域"：深度 1 的循环嵌套，只要其中所有分支都是均匀的，就整体交给控制线程：
- 控制线程镜像整个嵌套的 CFG（每个 kernel 块一个标量块，均匀 phi → 标量 phi，均匀指令克隆到自己块的克隆里）；向量侧每个 kernel 块
  是一个 vf 段（`<k>_wt_r<i>_b<n>`）。所有 lane 走同一条边，所以 phi 拷贝放在源块的段尾；若同一块的多个后继的拷贝之间有寄存器
  相互覆盖（不能排序），退而为每条边单独发一个 vf（`_e<n>`），控制线程在对应的边上插入它。
- 区域可以处于发散的 `if (gid < n)` 之下：进入循环时的掩码成为区域内所有向量指令的谓词；控制线程照着均匀条件求"是否可达"，
  发散条件按"可能"处理。
- 控制线程的标量 load 与之前发出的向量 store 的竞争：BasicAA+TBAA 判无别名则放行；区域之外的 store 可能先执行则在进入区域前
  由控制线程发 `fence`（等向量单元排空，每个 stripmine 组一次）；区域之内的 store 则拒绝该区域。
- 随循环推进的流基址在最内层 addrec 循环的 header 展开，控制线程在访问它的块前 `vmca`。
- 踩坑：LLVM 23 的 `BasicBlock::getTerminator()` 对未完成的块不再返回 nullptr 而是返回最后一条指令（release 下断言消失），
  导致"在终结指令前插入"变成"在最后一条指令前插入"，phi 之后的指令全跑到 phi 前面去了。
效果：kmeans 两个 kernel（含二重循环）和 pgain 的第二个循环成为区域；pgain 第一个循环因为循环内有 store 而保留掩码版。
回归 36 kernel、5 个 Rodinia、llama 全 PASS。RTL 对比（kmeans、pgain、llama 各两种）运行中。

RTL 结果（2026-09-12，区域 vs 循环留在 vf 内，同一编译器 `--no-ct-loops` 对照）：
| kernel | 标量 | 循环在 vf 内 | 控制线程区域 | 区域收益 | 对标量 |
|---|---|---|---|---|---|
| kmeans_swap | — | 35159 | 23664 | 1.49× | |
| kmeans_c（1024 点×8 维×5 类） | 1269172 | 130709 | 73055 | 1.79× | 17.4× |
| pgain（1024×8 维，只有第二个循环成区域） | 331274 | 39339 | 28776 | 1.37× | 11.5× |
全部 PASS。kmeans 整体 1348765 → 96719（14×，之前 8.2×）。

### llama RTL 结果（4 token，`+loadmem`，2026-09-12）
控制线程循环版（单块版编译器）：
| | 周期 | 每 token | 对标量 | matmul 占比 |
|---|---|---|---|---|
| 标量 Rocket | 11539928 | 2.88M | 1 | |
| hwacha，行主序权重（vlstw 跨行步长） | 1825757 | 456k | 6.3× | 91% |
| hwacha，转置权重（vlw 单位步长） | 992308 | 248k | 11.6× | 83% |
生成文本 "Once upon a time" 与标量一致，PASS。每 token 约 26 万次乘加，转置版 matmul 每 token 19.8 万周期 ≈ 1.3 MAC/周期：
每次迭代一个只有 3 条指令的 vf（vlw、vfmadd、phi 拷贝），固定的取指/发射开销占大头；下一步是区域内的 phi 合并（去掉拷贝）
和控制线程循环展开（一个 vf 做多次迭代）。同一权重、同一算法，单位步长比跨行步长快 2 倍：Hwacha 的 VMU 对单位步长有专门通路。
区域内加了 phi 合并：latch 值可以直接算进 header phi 的寄存器（读 phi 的那条指令自己写回也安全），循环的 LCSSA phi
直接别名到该寄存器（header phi 的寄存器保持到 LCSSA 的最后使用）。matmul_t 的循环体从 4 条降到 2 条：`vlw; vfmadd`。
区域版编译器（ct2）与 phi 合并版（ct3）RTL 结果（4 token）：
| 版本 | 行主序 | 转置 | 转置 matmul |
|---|---|---|---|
| 单块 CT loop（旧） | 1825757 | 992308 | 793396 |
| 区域（ct2） | 1834396 | 1021721 | 821990 |
| 区域 + phi 合并（ct3） | 1805131 | 939194 | 742719 |
ct3 转置版每 token 23.5 万周期，对标量 12.3×；matmul 每 token 18.6 万周期 ≈ 1.4 MAC/周期。循环体两条指令仍只有约 1.4 MAC/周期，
说明每次迭代一个 vf 的固定开销（取指、发射、vmcs/vmca）主导，下一步是控制线程循环展开。

## 全量回归（2026-09-13，最终编译器）
Spike：36 回归 kernel、5 Rodinia、expk、llama、微基准全部 PASS。RTL（`+loadmem`）：
- 微基准 N=1024：saxpy 1572（手写 1329）、clamp 1021（手写 1024）、divloop 17281（原 19582）、stencil 1815、gather 2243；
  N=4096：saxpy 3776（手写 3938）、clamp 3102（手写 3300）、divloop 67115（原 123134，区域化的收益）、stencil 6176、gather 7715。全部 VERIFIED。
- Rodinia：nn 15072、kmeans 96719、pgain 28776、pathfinder 144914、bfs 621840（标量 191905），全部 PASS。
- Berkeley 手写：vec-saxpy 10000 元素 13167 周期（含数据初始化的 mcycle）；vec-sgemm-opt / naive 256 宽与 hwacha-cc gemm_row 对比运行中。
新增 test/gemm（gemm_row：宿主按行发射，j 循环成区域，体两条指令；gemm_flat：整块一次发射，A/B 都成 gather）和 test/berkeley。
编译器补了 `freeze` 指令（别名到操作数）。

### sgemm 256×256×256 对比（RTL，2026-09-13）
| 版本 | 周期 | MAC/周期 |
|---|---|---|
| Berkeley 手写朴素版（每个 (i,j) 一个 vf，C 行每次读写） | 13918519 | 1.2 |
| hwacha-cc gemm_row（宿主按行发射，j 循环为控制线程区域，累加器留在寄存器） | 5211891 | 3.2 |
| Berkeley 手写优化版（4×4 寄存器分块） | 4262080 | 3.9 |
| hwacha-cc gemm_flat（整块一次发射，i=gid/n，A、B 都成 gather） | 36499744 | 0.46 |
编译器生成的版本比手写朴素版快 2.7 倍、比手写优化版慢 22%。与 llama 的 1.4 MAC/周期对照：这里 vl=256，
每次迭代一个 vf 的固定开销被摊薄；剩下的差距就是寄存器分块（一个 vf 里做多行多 j）。

## GPT-2 前向（llm.c）移植（2026-09-13）
`hwacha-cc/test/gpt2/`：小模型 V=1024、C=64、NH=4、L=2、T=16，参数用固定 LCG 随机初始化（x86 与 RISC-V 一致），
标量参考逐字取自 train_gpt2.c（encoder/layernorm/matmul_forward_naive/attention/gelu/residual/softmax）。
kernel（`gpt2.cl`）：encoder；layernorm 拆成 ln_stats（每行一个 lane，C 内循环成区域，步长流）和 ln_apply；矩阵乘按行发射、
权重宿主转置成 [in][out]（同 llama）；attention 拆成 att_score（(h,t,t2) 展平，t2>t 写 0）、att_softmax（每 (h,t) 一个 lane，
t2 循环走满 T 用 select 做因果掩码以保持均匀）、att_value（t2 走满 T，att 超过 t 为 0）；gelu 用 tanh(y)=1−2/(exp(2y)+1)
配 hw_expf；softmax_row 每行一个 lane。宿主要提供 `__math_oflowf/__math_uflowf`（newlib tanhf 引用，libm 里缺）。
Spike：argmax 与 x86 参考全同，logits 逐元素零差，概率最大差 <1e-9；PASS。10 个区域，matmul 体 `vlw; vfmadd`。
RTL 运行中（标量约 2000 万 Spike 指令，预计 3 小时）。
GPT-2 前向 RTL 结果（16 token）：标量 31244913 周期，hwacha-cc 1627888 周期（19.2×，每 token 约 10 万），argmax 全同、logits 零差、PASS。
分解：matmul 912654（56%）、softmax 307563（18%）、attention 248756（15%）、gelu 77073、layernorm 67127、encoder 8361、residual 5595。
softmax_row 与 att_softmax 每行一个 lane（vl=16 或 64），利用率低，是下一步该按 (行, 元素) 展平并做跨 lane 归约的地方。

## 跨 lane 归约（2026-09-13）
工作线程 ISA 里唯一的跨 lane 原语是 `vfirst`（取第一个活跃 lane 的值到 vs），没有树形归约，逐元素用它归约要 5 条指令乘 vl。
实现走内存 + 控制线程：`work_group_reduce_add/min/max`（OpenCL 2.0 名字，CL1.2 下用 `test/run/hwacha_builtins.h` 的
overloadable 声明；作用域 = 一个 stripmine 组 = work-group）。
- 向量侧：`tmp = identity; @P tmp = x; vsw tmp, va_scratch`（被掩掉的 lane 写单位元），然后 `vstop` 切段（`<k>_wt_x<n>`）。
- 控制线程：`fence` 等向量 store 落地，标量循环按 vl 归约暂存区（顺序与标量参考一致，layernorm 的结果逐位相同），
  `vmcs` 送回 vs 寄存器；控制线程自己也保留该值（区域内依赖它的均匀计算可以直接用）。
- 区域内的归约把块切成多段（BlockTail），控制线程在段之间插入归约循环；区域外走 Segments。
- 分析侧：归约调用标为 AlwaysUniform；class 传播里它是唯一合法的"读向量写标量"；不算 store、可克隆。
- 顺手补：`rsqrt`（vfsqrt + vfdiv），以及一个潜伏 bug：consensual 跳转的链接寄存器 LinkVS 原来在块体发射之后才分配，
  而跳转插在块体之前，块内死掉的值的寄存器可能被选中当链接寄存器并被覆盖；现在整个 kernel 预留一个 vs。
- 区域内随循环推进的 Local 流（按 local id 索引）不能再加 offset·stride，之前多加了，第二个组起全错。
测试 `test/run/red.cl`：rsum/rmax/risum、掩码贡献 + 广播（rnorm）、layernorm 式两次归约（rln）、区域循环内归约（rrows），全 PASS。
限制：归约不能出现在 vf 内的（发散）循环里；`if (lid==0) out[g]=s` 这种写法在区域里是发散分支，改成所有 lane 写同一地址
（均匀 store）即可。
GPT-2：layernorm 合成一个 kernel（组 = 一行 64 lane）、att_softmax 组 = 16、词表 softmax 组 = 128 lane 每 lane 8 个元素。
Spike 上两版都 PASS，概率最大差 2e-9。RTL 运行中。
备注：llama 的"循环留在 vf 内"基线（llama_rtl_noct）跑了 22 小时后随会话结束被杀，没有拿到数字；标量阶段之后向量阶段
已超过 20 小时，说明它至少比区域版慢 40 倍以上。区域的收益已由 kmeans/pgain/divloop 的成对对照给出，不再重跑。
RTL 结果（2026-09-14，GPT-2 16 token，标量 31219278）：
| 阶段 | 每行一个 lane | 归约版 | 组大小 / 每 lane 元素 |
|---|---|---|---|
| layernorm（5 次调用，16 行×64） | 67028 | 187339 | 64 / 1 |
| attention softmax 部分（含 score/value） | 247870 | 311530 | 16 / 1 |
| 词表 softmax（16 行×1024） | 305734 | 228768 | 128 / 8 |
| 总计 | 1634056 | 1733738 | |
归约回归（red，六个 kernel）RTL 全 PASS，GPT-2 两版 PASS。
结论：一次归约的固定代价（fence 排空 + vl 次标量访存累加 + 重新发射 vf）约 500 到 1000 周期，行短（64、16）时
比"每行一个 lane 串行循环"贵得多；只有行长（1024）且每 lane 先串行部分归约再跨 lane 时才划算（快 25%）。
改进方向：(1) 归约树在向量单元内做——把 vl 个值存回后用递减 vl 的 vf 段做 log 步加法，避免标量循环；(2) 让控制线程
在同一个 kernel 里同时推进多个组（组间流水），把 fence 的排空开销摊到多次归约上；(3) 编译器按行长自动选写法。

### 归约树进向量单元（2026-09-14）
三项改进的落地：
1. 归约树：lane 存值后，控制线程发 log2(vl) 个树步 vf（`<k>_wt_t<n>`：`vfence; vlw a, va_lo; vlw b, va_hi; op; vsw a, va_lo`），
   每步 `vsetvl n−m`、`vmca va_hi, scratch+m·esz`，m=⌈n/2⌉（非 2 的幂也对），最后 `vsetvl VL` 恢复；整数 min/max 用 vcmplt + 谓词 move。
2. 无 fence：结果留在 scratch[0]，下一段开头 `vfence; vlsw vsRes, vsScratch` 由向量单元自己取回，控制线程不停顿，
   多个组之间自然流水。控制线程只有在自己需要该值（区域里依赖它的均匀计算）时才懒惰地 `fence` + load。
3. 按行长选写法只能在 kernel 层做（重新映射 NDRange），编译器给出代价模型：树步约 log2(vl) 次固定开销，无排空；
   行短时"每 lane 串行 + 最后一步跨 lane"仍是正确写法，见下表。
Spike：red 六个、36 回归、Rodinia、llama、gemm、GPT-2 两版全 PASS；RTL：red 六个 PASS。GPT-2 对比运行中。
RTL 结果（GPT-2 16 token，2026-09-14）：
| 阶段 | 每行一个 lane | 归约（控制线程标量循环） | 归约树（向量单元内，无 fence） |
|---|---|---|---|
| layernorm | 67062 | 187339 | 135447 |
| attention | 247951 | 311530 | 300516 |
| 词表 softmax | 305312 | 228768 | 190855 |
| 总计 | 1632504 | 1733738 | 1633700 |
全部 PASS。softmax（行长 1024、每 lane 8 元素）归约树比每行一个 lane 快 37%；layernorm（行长 64）和 attention（行长 16）
仍是固定开销主导，树每步一次 vf 发射，6 步或 4 步的代价超过 64 或 16 元素的串行循环。行短时的正确写法仍是
"lane 内串行，最后一步跨 lane"；再往下压只能把小 vl 的树在一个 vf 里展开。

## RTL bug 深挖：标量浮点挂死的真正根因（2026-09-14）
之前记的"vs 目的浮点走共享 FPU、在此配置下没接通"只对了一半。用 +verbose 在 Hwacha 的 RoCC 边界、tile 的 FPU 仲裁器、
Rocket FPU 三处加握手打印后定位到两个独立的 bug：
1. `hwacha/scalar-decode.scala`：FPU 译码表的 typeTagIn/typeTagOut 两列还是旧接口的"single 位"（Y=单精度）,
   而现在 Rocket 的 FPU 用类型索引（S=0, D=1, minFLen=32）。每条单精度标量运算都被当成双精度,反之亦然。
   修复：把这两列单独引出为 fpu_single_in/out,再 `typeTag := Mux(single, 0, 1)`。
2. `rocket-chip/tile/RocketTile.scala`：协处理器连接循环无条件对每个 RoCC 赋 `fpu_req.ready := DontCare`、
   `fpu_resp.valid/bits := DontCare`。这段在模块体里、晚于 HasLazyRoCCModule 把 Hwacha 的 FPU 端口连到共享 FPU
   仲裁器,后连接覆盖前连接,于是 Hwacha 侧永远 `fpu_req.ready=0`、永远收不到响应。**这才是标量浮点挂死的真凶。**
   边界打印铁证:仲裁器 in_req ready=1、FPU 算出结果、仲裁器收到响应,但 Hwacha 侧 fpu_req.ready=0、无响应。
   修复:那四行 DontCare 用 `if (!lm.usesFPU)` 包起来,只给不用 FPU 的 RoCC 做端口收尾。
另外:稀疏字节 store 的"VMU 死锁"复现不出来。TileLink + VMU 跟踪显示 VMU 无卡住请求;当前编译器生成的稀疏字节 store
探针在新旧仿真器上都通过。之前的现象很可能是 consensual 跳转链接寄存器被覆盖的编译器 bug(做归约时已修)所致,
需在 bfs 上进一步确认后即可去掉 `--no-subword-rmw` 绕过。
这两处 FPU 修复独立于编译器:编译器仍默认把 FP 留在向量寄存器(needsFPU),修好后 `--scalar-fp` 生成的代码才可用于 RTL。

## 收尾三件事（2026-09-14）
1. **稀疏字节 store 绕过不能去掉——是真的 RTL bug。** bfs 用当前编译器（已含 LinkVS 修复）以 `--no-subword-rmw` 重编后,
   在修复 FPU 的仿真器上仍挂死(标量参考打印后卡住,3M 周期超时)。MRT 信用跟踪定位:掩码字节 store 的某些 beat(整组 lane 被
   掩掉、跨 8 元素条带边界)预留了存储信用却不归还,`pending.store` 卡在 scount=496/512 永不清零,后续 fence/vf 结束永远等待。
   共泄漏 16 个信用(2 组 × 8)。这是 VMU/sequencer 存储信用记账 bug,不是编译器问题(简单的单 lane-0 探针触发不了,bfs 才触发)。
   修它要改 vmu-pred/vmu-memif 的 sret 逻辑,是另一个多小时的活。`--no-subword-rmw` 绕过保留。
2. **FPU 修复独立验证:** sfp 三个 kernel(单精度乘加、整数转浮点、双精度)在修复后仿真器上全 PASS;spec、red 无回归
   (spec 三个 5000 元素 kernel 需要约 2-3M 周期,之前用 500k/1.5M 上限误判为挂,8M 上限下 ALL PASS)。
   两处 FPU 修复导出到 `patches/chipyard-hwacha-rtl-fixes.patch`(scalar-decode 类型标签)和 `patches/chipyard-rocketchip-fpu-fix.patch`
   (RocketTile FPU 端口 DontCare 守卫)。补丁里还含 plusarg 门控的 TileLink/VMU/MRT 跟踪(默认关闭,用于性能模型和信用泄漏调试)。
3. **多线程仿真器速度:** VERILATOR_THREADS=8 + SIM_OPT_CXXFLAGS=-O2,实测 **约 14000 周期/秒**,对比原单线程 -O1 的 2700,
   快 5.2 倍。GPT-2 那种约 3100 万周期的运行从 6 小时降到约 40 分钟。构建脚本 `scripts/build-sim.sh`。

## 稀疏字节 store 死锁的根因与修复（2026-09-14 晚）
用 `+hwacha_sret_trace=1` 逐 beat 对账后发现:9 个真正写内存的 beat 都发出了 PutPartial、都收到了 AccessAck、VMT 也都
读回了正确的 ecnt(8,1,1,…)并置了 sret_resp——响应路径完全正常。泄漏在最后一行:
`io.sret.cnt := Mux(req_en, req_cnt, 0) + Mux(resp_en, resp_cnt, 0)`。两个操作数都是 4 位(`CInt.decode()` 给 1..8),
Chisel 的 `+` 不扩位,先按 4 位截断再赋给 5 位端口。字节模式下一个全掩掉 beat 的请求时归还(8)与一个 8 元素 store 的
响应归还(8)落在同一拍 → 16 截成 0 → 两份信用一起丢,正好是 MRT 卡在 496/512 的 16。
- 只有字节模式的 beat 才有 8 个元素(字模式最多 4+4=8 不溢出)→ 只有子字 store 触发;
- 需要两种归还事件同拍 → 依赖访存时序,单 lane 探针撞不上,bfs 能撞上;
- 与"只有 lane 0 活跃"无关,之前的表征是巧合。
修复:`+` 改 `+&`(扩位加)。属于 Hwacha 原有 bug(不是 Chipyard 集成腐烂)。
验证(修复后的仿真器,22:40 构建):bfs 去掉绕过 PASS(607864 周期,比带绕过的 621840 还快 2%),pfmin_raw 四个 kernel PASS,
red、spec 无回归;Spike 上 36 回归 + Rodinia + llama + GPT-2 用新默认全 PASS。
编译器默认改为直接发射掩码子字 store;`--no-subword-rmw` 改名为 `--subword-rmw`(为未打补丁的 RTL 保留绕过)。
至此本项目发现的 6 个 RTL bug 全部修复,补丁在 patches/。

## 接入 MLIR：gpu dialect 入口（2026-09-14 深夜）

选的是"路线一"：不改 hwacha-cc 的核心，只在前面加一个 MLIR 前端，把 gpu dialect 落到
hwacha-cc 已经理解的 OpenCL 约定上。

**环境**：`conda install -c conda-forge mlir=23.1.1`（与 llvmdev/clangdev 23.1.1 同版本，装完
libLLVM 没变，hwacha-cc 不需要重编）。带来 `mlir-opt`、`mlir-translate`、`mlir-runner`；没有
Python 绑定。

**流水线**（`hwacha-cc/tools/mlir-to-ll.sh`）：

1. 输入若没有 `gpu.module`（linalg / scf.parallel 级别）：`convert-linalg-to-parallel-loops`
   →（可选 `test-scf-parallel-loop-collapsing`，把多维并行循环压成一维）→ `gpu-map-parallel-loops`
   → `convert-parallel-loops-to-gpu` → `gpu-kernel-outlining`。
2. `lower-affine` → `convert-scf-to-cf` → `convert-gpu-to-nvvm`（默认 bare-ptr memref 约定，需要
   静态 shape；`MEMREF_CONV=desc` 用 descriptor 约定，动态 shape 可用，kernel 参数变成 5 元组）
   → `reconcile-unrealized-casts`。
3. `tools/mlir-extract-kernels.py` 把 `gpu.module` 体抠出来单独成一个 module（丢掉宿主侧的
   `gpu.launch_func`），`mlir-translate --mlir-to-llvmir` 得到 LLVM IR。

用 NVVM 而不是自己写一个 gpu→llvm 的 conversion，是因为 `convert-gpu-to-nvvm` 已经把 gpu/arith/
memref/math/cf 全套降到 LLVM dialect，只留下几个 NVVM 内建；把这几个内建改写掉比重写一套
pattern 便宜得多。

**hwacha-cc 里的适配层**（`src/GPUAdapt.cpp`，parseIRFile 之后自动检测 `ptx_kernel` 调用约定）：

| NVVM 形式 | 改写成 |
|---|---|
| `ptx_kernel` 调用约定 | C 约定 + `!kernel_arg_addr_space`（这是 isKernel 的标记） |
| `llvm.nvvm.read.ptx.sreg.tid.x / ctaid.x / ntid.x` | `trunc(_Z12get_local_idj(0)) / _Z12get_group_idj / _Z14get_local_sizej`，带 `!range [0,2^30)` |
| 同上，但函数有 `"nvvm.maxntid"="1,1,1"`（gpu-map-parallel-loops 对一维循环的默认映射：每 block 一个线程）或 `--gpu-block1` | `ctaid.x → get_global_id(0)`，`tid.x → 0`，`ntid.x → 1` |
| y/z 维 | 0 / 1（启动是一维的；多维循环先 collapse） |
| `llvm.nvvm.barrier*` | `_Z7barrierj(1)` |
| `addrspace(3)` 全局变量（gpu workgroup 内存） | addrspace(0) 的 internal 全局，指针类型沿 GEP/phi/select 传播改写 |
| `__nv_sqrtf/fmaf/fabsf/fminf/fmaxf…`（libdevice） | `llvm.sqrt/fma/fabs/minnum/maxnum` 内建；其余去掉 `__nv_` 前缀成 libm 名 |
| `ctaid*ntid + tid` | 折叠成 `get_global_id(0)`，-O2 前后各做一次（-O2 之后加法会被拆成两级 GEP） |

改写完跑与 clang 一样的 -O2 流水线（关掉向量化和展开；MLIR 出来的 IR 是 -O0 形态，descriptor
的 insertvalue/extractvalue 链要靠它折掉），之后就是原来的 KernelAnalysis / CodeGen。

`ctaid*ntid+tid` 的折叠很关键：不折的话 hwacha-cc 不知道它等于全局 id，地址被归成 gather
（`vlxw`），折了之后是单位步长流（`vlw va0`）。

**测试**（`hwacha-cc/test/mlir`，一个二进制 `mlir.riscv`，Spike 全过）：

| kernel | 写法 | 走的路径 | 生成代码 |
|---|---|---|---|
| saxpy_gpu | 手写 `gpu.func`，CUDA 风格 `bid*bdim+tid` + 边界判断 | bare-ptr | `vlw/vsw` 流 + 谓词 |
| saxpy_linalg | `linalg.generic` 自动外提，block 大小 1 | descriptor | `vlstw/vsstw`（步长是运行时参数） |
| wg | workgroup memref + `gpu.barrier` | bare-ptr | `vsw va` 到本地缓冲 + `vfence` + `vlsw` |
| ex | `math.absf/sqrt/fma` | bare-ptr | `vfsgnjx/vfsqrt/vfmadd` |
| mm | `linalg.matmul` 64×64，二维并行循环 collapse 成一维 | bare-ptr | 外层 `vdiv/vrem` 算行列，k 循环成为控制线程循环区域 |

宿主侧没接 `gpu.launch_func`：还是 C 里直接调 `<kernel>_ct(n, args...)`，n = 总工作项数，
`hwacha_group_size` = block 大小（descriptor 约定时每个 memref 传 5 个参数）。

**没做的 / 限制**：`nctaid`（grid 大小）kernel 里拿不到；`math.exp` 变成 `__nv_expf → expf`，
hwacha-cc 的 kernel 内不支持 libm 调用（llama 那条路是自己写的 `hw_expf`），要么在 MLIR 层
`math-polynomial-approximation`，要么以后给 hwacha-cc 加一个向量 expf 的内建；多维 grid 只能
靠 collapse；`test-scf-parallel-loop-collapsing` 是个 test pass，正式做要自己写 collapse。

## MLIR 入口收尾：hwacha-mlir、launch_func、expf、分块 matmul（2026-09-15）

上一节留下的四件事全做了，顺带把 Python/shell 的胶水换成了一个正规的 C++ 前端。

**hwacha-mlir**（`hwacha-cc/mlir/hwacha-mlir.cpp`，链接 conda 包里的 libMLIR，CMake 找到 MLIR 就编）。
输入 gpu dialect 或 linalg/scf.parallel，输出 kernel 的 LLVM IR（给 hwacha-cc）和宿主的 LLVM IR
（`hwacha-cc --host` 接收，跑同样的 -O2，Linker 链进控制线程 module，一起过 llc）。流程：

1. 内嵌的 transform 脚本（`transform.with_named_sequence` 的嵌套 module）先跑 `transform-interpreter`。
2. 没有 `gpu.module` 时：`convert-linalg-to-parallel-loops` → `fold-memref-alias-ops`（tiling 产生的
   subview 折回下标，不然 bare-ptr 约定不接受带动态 offset 的 layout）→ 嵌套的 scf.parallel 改成
   scf.for（自己写的 `nestedParallelToFor`）→ 最外层 parallel 用 `collapseParallelLoops` 压成一维
   （替掉了之前的 test pass）→ mapping 设成 block_x → `convert-parallel-loops-to-gpu` →
   把 launch 里用到的常量克隆进 launch（`sinkConstants`）→ 常量次数 ≤ 8 的 scf.for 全展开
   （`loopUnrollFull`）→ `gpu-kernel-outlining`。
3. `lower-affine`、`convert-scf-to-cf`；kernel 侧 clone 一份跑 `gpu.module(convert-gpu-to-nvvm)`，
   把 gpu.module 里的东西提到顶层再 `translateModuleToLLVMIR`。
4. 宿主侧：先把 gpu.func 的函数体掏空（只留 gpu.return，让 launch_func 的符号检查通过，又不让
   workgroup memref 之类进宿主转换），`finalize-memref-to-llvm`、`convert-func-to-llvm`、arith/cf/
   index/math → 自己的 `lowerLaunches`：n = grid×block 六个维度之积；`hwacha_group_size` =
   block 大小（block 为 1 时存 0，让硬件自选 vl）；`hwacha_grid_size` = block 数；memref 操作数看穿
   `unrealized_conversion_cast` 拿到 LLVM 描述符，bare 约定取 aligned 指针，desc 约定展开五元组；
   `llvm.call @<kernel>_ct(n, args...)`；删掉 gpu.module，`reconcile-unrealized-casts`，翻译。

踩的坑：`parsePassPipeline` 的字符串不能再写 `builtin.module(...)` 外壳（PassManager 已经锚在
module 上，再套一层就是去找嵌套的 module，什么都不跑）；必须 `registerAllExtensions`，不然
`convert-gpu-to-nvvm` 收集不到 arith/cf/index 的 ConvertToLLVM 接口，只转 gpu 算子；两个 llc 输出
直接拼接会撞 `.Lpcrel_hi0` 之类的局部标号，所以改成 IR 级链接。

**hwacha-cc 这边新增**：`--host`；`nctaid.x` → 读 `hwacha_grid_size`（kernel 里的声明在控制线程
module 里变成 weak 定义，`cloneUniform` 对声明型全局变量改用 WeakAny 而不是 Internal，否则链接
时宿主引用不到）；`expandExpf`：`expf / llvm.exp.f32 / _Z3expf / __nv_expf` 在分析前展开成
hw_expf 同款的直线向量算术（所以 OpenCL 里现在也可以直接写 `exp`）；`--assume-noalias`：所有指针
kernel 参数加 noalias；`--fp-contract`：单用途 fmul + fadd/fsub → fmuladd → `vfmadd`。

**分块 matmul**（`test/mlir/mmt.mlir`）：linalg.matmul + transform 脚本
`generalize → tile_using_for [4,0,0] → interchange [1,2,0]`，得到
`scf.for i0 step 4 { parallel(j) { for k { parallel(i' in 4) {...} } } }`：i0 留在宿主（16 次
launch），j 是 lane，k 是控制线程循环，i' 展开。配合 `--assume-noalias` LICM 把
`c[i0+i', j]` 提升成 4 个循环 phi，生成的 vf 代码正是手写版的形态：

```
matmul_blk_kernel_wt_r0_b0:      # 每个 k 迭代
    vlw vv8, va4                 # b[k, j] 单位步长
    vfmadd.s vv4, vs2, vv8, vv4  # a[i0+i', k] 四个标量经 vmcs
    vfmadd.s vv5, vv8, vs7, vv5
    ...
```

常量必须 sink 进 launch：外提默认把 lb/step 当 kernel 参数传，四个 c 地址的差就成了
`step*64`，BasicAA 证不出不重叠，LICM 不提升。

**测试**：`test/mlir` 七个 kernel，宿主全部由 MLIR 生成（C 只准备数据、校验、计时）：
saxpy(gpu，16×256 launch)、saxpy(linalg，desc 约定)、grid-stride（`gpu.grid_dim`，4×64 个工作项
处理 4096 元素）、workgroup、math+exp、matmul(linalg)、matmul(tiled)。Spike 和 RTL 全过。

RTL 上 64³ matmul 的周期数（`results/mlir_rtl.out`）：

| 版本 | 周期 | MAC/cycle |
|---|---|---|
| linalg.matmul 直接降低（一维 collapse，每个 lane 一个输出，c 每个 k 迭代读写） | 650540 | 0.40 |
| 同一个 linalg.matmul + transform 脚本（4 行分块，寄存器累加，vfmadd） | 71654 | 3.66 |

9.1×。3.66 MAC/cycle 已经和手写 OpenCL 的 gemm（3.2）、Berkeley 手写汇编 `vec-sgemm-opt`（3.9，
256³）在一个量级；剩下的差距是每次 launch 的固定开销（16 次 launch，每次 vsetcfg/fence）和
k 循环里每个 vf 块只有 5 条指令、vl 只有 64。把 i0 也放进 kernel（tile_using_forall 后 mapping 成
sequential）或者 j 方向再分块都可以再收一点，但形态已经对了。

## 分块 matmul 的外层循环进 kernel（2026-09-15 晚）

上一节的 16 次 launch 改成一次。做法全在 transform 脚本里：先 `tile_using_forall [0,1,0]` 把 j
切成 forall（并行语义，之后 `scf-forall-to-parallel` 变成最外层 scf.parallel，即 lane 维），再在
里面 `tile_using_for [4,0,0]` 按 4 行分块，最后 `interchange [2,0,1]` 让 k 在 i'、j' 外面：

```
scf.parallel (j) { scf.for i0 step 4 { scf.for k { scf.parallel (i', j'=1) {...} } } }
```

hwacha-mlir 的流水线前面加了 `scf-forall-to-parallel`；其余（嵌套 parallel → for、常量 sink、小循环
展开、外提）不变。kernel 里 i0 和 k 成了两层控制线程循环区域，n = 64（一个 lane 一列）。

顺手修了一个 codegen 低效：内层循环退出到外层循环体的 LCSSA phi 之前按"普通 phi"处理，每个 k
迭代把 4 个累加器拷贝到另 4 个寄存器（`vaddw vvX, vv0, vs0`），k 块 9 条指令里有 4 条是拷贝。
`emitCTBlock` 现在对"离开区域内某个内层循环"的边也走别名路径（`InnerExit`），累加器寄存器直接
给出口 phi 用。k 块只剩 `vlw` + 4 条 `vfmadd`。回归、gemm、llama 在 Spike 上都仍然通过。

RTL 上 64³ matmul：

| 版本 | 周期 | MAC/cycle |
|---|---|---|
| linalg 直接降低 | 649988 | 0.40 |
| 分块，16 次 launch（上一节） | 71654 | 3.66 |
| 分块，一次 launch，i0 在 kernel 里 | 69450 | 3.77 |

launch 开销本来就不大（16 次共约 2k 周期）。剩下的瓶颈是每个 k 迭代一个 vf 块只有 5 条指令、
vl = 64：1024 个 vf 块各约 68 周期，基本就是控制线程发射 + vf 启动的固定开销。要再往上只能
增大每个 vf 块的工作量：j 方向用更长的向量（矩阵更大）、或者把 k 也分块展开（一个 vf 块做
多个 k）——后者对 hwacha-cc 是控制线程循环的 unroll，是下一步可以做的事。
## diffusion.c：扩散模型采样（2026-09-15 深夜）

hku/diffusion.c 是一个 llama2.c 风格的纯 C 扩散模型推理引擎：16×16 精灵图、5 类条件、
ContextUnet（n_feature = 64，1.48M 参数，5.9 MB fp32，仓库自带 ckpt.bin），DDPM 采样 200 步，
每步一次 U-Net 前向 = 150.7M MAC。算子集：3×3 卷积、1×1 shortcut 卷积、转置卷积（k=s=2 和
k=s=4）、BatchNorm（对单张图按通道算统计量）、GroupNorm(8)、GELU、ReLU、MaxPool、AvgPool、
两层 MLP 的时间/条件嵌入、逐通道仿射。和 llama/GPT-2 一样：算子换成 OpenCL kernel，主程序、权重
加载、采样循环留在 C（`test/diffusion`）。

**布局是全部设计**：所有激活都放在零填充的平面 `[C][Hp*Wp]`（Hp = H+2）里，缓冲区前后各留
2·Wp+8 个 float 的 guard。3×3 卷积于是不用做任何边界判断：lane = 填充后的输出位置，
控制线程循环 ic × 9 个 tap，每个 tap 是一条单位步长的移位流（`xs[(t/3-1)*Wp + t%3-1]`），
一次加载喂 8 条 `vfmadd`（8 个输出通道一组，8 个累加器常驻 vv，8 个权重经 vmcs 进 vs）：

```
conv3x3_wt_r0_b1:            # 每个 (ic, tap) 迭代
    vlw vv16, va0
    vfmadd.s vv8,  vs26, vv16, vv8
    ... ×8
```

边界位置算出来是垃圾，但紧跟在每个卷积后面的 BN/GN + 激活 kernel 只在内部区域算并把边界写 0，
所以下一层看到的填充永远是 0。代价是多算 324/256 = 1.27 倍的元素。转置卷积 k=s=2 反过来：
lane = 输入像素，4 个输出通道 × 4 个 tap 共 16 个累加器，最后 scatter 到两倍大的平面；k=s=4 的
up0 输入是 1×1，就是一个 gemv。归一化的统计量：lane = 通道，控制线程循环内部像素（跨 lane
是步长为 plane 的流），host 把通道和并成组；apply kernel 逐元素，用整数除法算出通道/行/列做
内部掩码。GELU 用 `exp` 写（tanh z = 1 - 2/(1+e^{2z})），靠编译器新的 expf 展开。

**寄存器的坑**：第一版把 9 个 tap 全展开 + 8 个输出通道，每个 ic 迭代要 72 个权重标量，
`out of Hwacha registers of class vs`（vs 只有 64 个）；4 个通道 36 个权重也不够（参数、常量、
per-iteration 的流基址都要 vs）。改成 tap 也做控制线程循环，每次迭代只需 8 个权重。

**裸机的坑**：环境的 printf 不支持 `%g`，把 float 参数当指针解引用，trap 成 `tohost=1337`；
定义一个 weak 的 `handle_trap` 打出 cause/epc 才定位到 vprintfmt。libm 还缺 `__errno`。

**验证**：同一二进制里带原版 diffusion.c 的标量前向（照抄，malloc 换成 bump arena），前 2 步
逐元素比对预测噪声；rand() 换成 LCG，x86 版（`-DX86`，只跑标量）和 Hwacha 版抽同样的噪声，
200 步之后的图可以直接比。RTL 上标量整步要 1.5G 周期（30 小时），所以 RTL 的加速比用
`-DLAYER_BENCH`：同一个 3×3 卷积层（64→16 通道）标量 vs Hwacha。

**Spike 结果**：前 2 步预测噪声与标量参考最大差 4e-6（参考最大值 4.6）；200 步跑完
（约 45 分钟），最终 16×16 图与 x86 参考**逐像素完全一致**（`results/spike200.log` vs
`results/x86_200.log`）。每步前向 20.42M Spike 周期；标量参考同一步是 1.5G 周期量级（单个
64→16 的 3×3 卷积层标量 42.9M vs Hwacha 0.27M，157×，Spike 计数）。RTL 的一步和卷积层
基准结果见下。

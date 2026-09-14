// Code generation for one OpenCL kernel:
//   worker thread  -> Hwacha vector-fetch block, emitted as assembly text
//   control thread -> LLVM IR function "<kernel>_ct(i64 n, <kernel args>...)" in CTModule,
//                     with Hwacha control instructions as inline asm; compiled by llc later.
#pragma once
#include "Analysis.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace hwacha {

struct CodeGenOptions {
  bool Verbose = false;
  bool Stats = false;      // print per-kernel register/instruction statistics
  bool NoV32 = false;      // configure every vector register as 64-bit (ablation)
  bool NoSkip = false;     // do not emit consensual jumps around inactive blocks (ablation)
  bool NoCoalesce = false; // do not coalesce loop-carried values into their phi register (ablation)
  bool ScalarFP = false;   // allow scalar (vs) floating-point ops in the block (hangs on Chipyard RTL)
  bool SubwordRMW = false;   // lower masked sub-word unit-stride stores to load/select/store: workaround for the
                             // unpatched Hwacha RTL (store-credit overflow, see patches/); default: emit them directly
  bool NoCTLoops = false;  // keep every loop inside the vf block (ablation: no control-thread-driven loops)
};

// Returns false and writes a diagnostic to Err if the kernel uses something unsupported.
bool generateKernel(llvm::Function &Kernel, KernelAnalysis &KA, llvm::Module &CTModule,
                    llvm::raw_ostream &WT, llvm::raw_ostream &Err, const CodeGenOptions &Opts);

// Kernel preprocessing that must run before analysis: id range metadata + cleanup passes.
void prepareKernel(llvm::Function &F);

} // namespace hwacha

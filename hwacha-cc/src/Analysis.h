// Kernel analysis for the Hwacha vector-fetch backend:
//  - kernel discovery (clang OpenCL metadata)
//  - uniformity: which values vary per work-item (-> vv) vs uniform (-> vs)
//  - address classification: which memory operands are unit/constant-stride
//    streams (-> va register) vs uniform scalars vs gathers/scatters
#pragma once
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/UniformityAnalysis.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/CycleInfo.h"
#include <map>
#include <memory>
#include <vector>

namespace hwacha {

// True for OpenCL kernels as emitted by clang (skips the __clang_ocl_kern_imp_ stubs).
bool isKernel(const llvm::Function &F);
// GPU-dialect entry (MLIR convert-gpu-to-nvvm + mlir-translate output): rewrite NVVM conventions into the
// OpenCL ones above and run the -O2 pipeline. Adapted is set when the module contained ptx kernels.
// Inline erff calls (erff / llvm.erf.f32 / _Z3erff) as vector arithmetic (A-S 7.1.26); emits an
// llvm.exp.f32, so run before expandExpf.
void expandErff(llvm::Function &F);
// Inline expf calls (expf / llvm.exp.f32 / _Z3expf) as vector arithmetic; run before analysis.
void expandExpf(llvm::Function &F);
// Contract fmul+fadd/fsub into fmuladd (-ffp-contract=fast); changes rounding.
void contractFMA(llvm::Function &F);
// Run the clang-equivalent -O2 pipeline (no vectorization / unrolling) on a module.
void optimizeModule(llvm::Module &M);
bool adaptGPUModule(llvm::Module &M, bool ForceBlock1, bool NoOpt, llvm::raw_ostream &Err, bool &Adapted);
// Work-item id calls: get_global_id(0) / get_local_id(0). Both are divergence sources.
bool isWorkItemId(const llvm::Value *V);
bool isGlobalId(const llvm::Value *V);
bool isLocalId(const llvm::Value *V);
// Uniform per-group queries (get_local_size(0), get_group_id(0)) and barrier().
bool isLocalSizeCall(const llvm::Value *V);
bool isGroupIdCall(const llvm::Value *V);
bool isBarrierCall(const llvm::Value *V);
// work_group_reduce_{add,min,max}(x): cross-lane reduction over the work-group; Op receives "add"/"min"/"max"
bool isWorkGroupReduce(const llvm::Value *V, llvm::StringRef *Op = nullptr);

enum class AddrKind { Stream, Uniform, Gather };

struct MemAccess {
  llvm::Instruction *I;          // load or store
  AddrKind Kind;
  llvm::Value *Base = nullptr;   // Stream/Uniform: uniform base pointer expression (SCEV-expandable)
  const llvm::SCEV *BaseSCEV = nullptr;
  int64_t Stride = 0;            // Stream: bytes between consecutive work-items (0 if not constant)
  bool Local = false;            // Stream: indexed by the local id (base does not advance per group)
  const llvm::SCEV *StrideSCEV = nullptr; // Stream: stride, uniform but maybe not constant
  llvm::Value *Index = nullptr;  // Gather: the divergent index value
};

class KernelAnalysis {
public:
  explicit KernelAnalysis(llvm::Function &F);
  ~KernelAnalysis();

  bool isUniform(const llvm::Value *V) const;
  // Re-run the uniformity analysis after new instructions were inserted (no CFG changes).
  void recomputeUniformity();
  bool hasDivergentBranch(const llvm::BasicBlock *BB) const;
  const std::vector<MemAccess> &memAccesses() const { return Accesses; }
  llvm::ScalarEvolution &scalarEvolution() { return *SE; }
  llvm::TargetLibraryInfo &targetLibraryInfo() { return *TLI; }
  llvm::AssumptionCache &assumptionCache() { return *AC; }

  void print(llvm::raw_ostream &OS) const;

private:
  void classifyAccesses();
  bool decompose(const llvm::SCEV *S, const llvm::SCEV *&Base, const llvm::SCEV *&Stride, bool &Local);

  llvm::Function &F;
  llvm::TargetLibraryInfoImpl TLII;
  std::unique_ptr<llvm::TargetLibraryInfo> TLI;
  std::unique_ptr<llvm::AssumptionCache> AC;
  std::unique_ptr<llvm::DominatorTree> DT;
  std::unique_ptr<llvm::LoopInfo> LI;
  std::unique_ptr<llvm::ScalarEvolution> SE;
  std::unique_ptr<llvm::TargetTransformInfo> TTI;
  std::unique_ptr<llvm::CycleInfo> CI;
  std::unique_ptr<llvm::UniformityInfo> UI;
  std::vector<MemAccess> Accesses;
};

} // namespace hwacha

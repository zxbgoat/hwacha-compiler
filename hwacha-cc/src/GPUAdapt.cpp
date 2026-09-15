// GPU-dialect entry: rewrite LLVM IR produced by MLIR's convert-gpu-to-nvvm + mlir-translate into
// the OpenCL-style conventions the rest of hwacha-cc understands.
//
//   ptx_kernel calling convention           -> C, plus !kernel_arg_addr_space (marks the kernel)
//   llvm.nvvm.read.ptx.sreg.tid.x           -> trunc(_Z12get_local_idj(0))
//   llvm.nvvm.read.ptx.sreg.ctaid.x         -> trunc(_Z12get_group_idj(0))
//   llvm.nvvm.read.ptx.sreg.ntid.x          -> trunc(_Z14get_local_sizej(0))
//   (block size 1, i.e. "nvvm.maxntid"="1,1,1" or --gpu-block1: ctaid.x -> get_global_id(0),
//    tid.x -> 0, ntid.x -> 1 — the shape gpu-map-parallel-loops produces for 1-D loops)
//   y/z dimensions                          -> 0 / 1 (the launch is 1-D; collapse loops first)
//   llvm.nvvm.barrier*                      -> _Z7barrierj(1)
//   addrspace(3) globals (gpu workgroup)    -> addrspace(0) internal globals (__local buffers)
//   __nv_<f>                                -> <f> (libdevice math -> libm)
// then the module gets the same -O2 (no vectorize / unroll) pipeline clang would have run.
#include "Analysis.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

Function *builtinDecl(Module &M, StringRef Name) {
  LLVMContext &C = M.getContext();
  FunctionType *FT = FunctionType::get(Type::getInt64Ty(C), {Type::getInt32Ty(C)}, false);
  Function *F = cast<Function>(M.getOrInsertFunction(Name, FT).getCallee());
  F->setMemoryEffects(MemoryEffects::none());
  F->setDoesNotThrow(); F->setWillReturn();
  return F;
}

// call <builtin>(0) as i64 with a [0, 2^30) range, truncated to i32 (what the NVVM sreg reads return)
Value *builtinCall(Module &M, IRBuilder<> &B, StringRef Name) {
  LLVMContext &C = M.getContext();
  CallInst *CI = B.CreateCall(builtinDecl(M, Name), {B.getInt32(0)});
  Type *I64 = Type::getInt64Ty(C);
  Metadata *R[] = {ConstantAsMetadata::get(ConstantInt::get(I64, 0)), ConstantAsMetadata::get(ConstantInt::get(I64, 1u << 30))};
  CI->setMetadata(LLVMContext::MD_range, MDNode::get(C, R));
  return B.CreateTrunc(CI, B.getInt32Ty());
}

bool blockSizeIsOne(const Function &F) {
  for (StringRef A : {"nvvm.reqntid", "nvvm.maxntid"}) {
    Attribute At = F.getFnAttribute(A);
    if (At.isValid() && At.isStringAttribute()) {
      StringRef S = At.getValueAsString();
      if (S == "1" || S == "1,1,1" || S == "1,1" ) return true;
    }
  }
  return false;
}

// Replace every reference to an addrspace(3) global by an addrspace(0) copy and fix the pointer
// types of the instructions that flow from it.
bool flattenWorkgroupGlobals(Module &M, raw_ostream &Err) {
  SmallVector<GlobalVariable *, 8> WG;
  for (GlobalVariable &G : M.globals()) if (G.getAddressSpace() == 3) WG.push_back(&G);
  if (WG.empty()) return true;
  SmallVector<Constant *, 8> Cs(WG.begin(), WG.end());
  convertUsersOfConstantsToInstructions(Cs);
  PointerType *P0 = PointerType::get(M.getContext(), 0);
  for (GlobalVariable *G : WG) {
    Constant *Init = G->hasInitializer() && !isa<UndefValue>(G->getInitializer()) && !isa<PoisonValue>(G->getInitializer())
                     ? G->getInitializer() : Constant::getNullValue(G->getValueType());
    auto *G0 = new GlobalVariable(M, G->getValueType(), G->isConstant(), GlobalValue::InternalLinkage, Init, G->getName() + ".local", nullptr, GlobalValue::NotThreadLocal, 0);
    G0->setAlignment(G->getAlign().value_or(Align(8)));
    for (Use &U : llvm::make_early_inc_range(G->uses())) {
      if (!isa<Instruction>(U.getUser())) { Err << "hwacha-cc: unsupported non-instruction use of workgroup global " << G->getName() << "\n"; return false; }
      U.set(G0);
    }
    G->eraseFromParent();
  }
  // propagate: any instruction still typed ptr addrspace(3) becomes ptr (or disappears if it is a cast)
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (Function &F : M)
      for (Instruction &I : llvm::make_early_inc_range(instructions(F))) {
        auto *PT = dyn_cast<PointerType>(I.getType());
        if (!PT || PT->getAddressSpace() != 3) continue;
        if (auto *AC = dyn_cast<AddrSpaceCastInst>(&I)) {
          if (AC->getOperand(0)->getType() != P0) { Err << "hwacha-cc: addrspacecast from a non-flat pointer in " << F.getName() << "\n"; return false; }
          AC->replaceAllUsesWith(AC->getOperand(0)); AC->eraseFromParent(); Changed = true; continue;
        }
        if (isa<GetElementPtrInst>(I) || isa<PHINode>(I) || isa<SelectInst>(I)) { I.mutateType(P0); Changed = true; continue; }
        Err << "hwacha-cc: cannot flatten workgroup pointer produced by " << I << "\n"; return false;
      }
    // casts 0 -> 3 whose result was mutated away, or now-trivial casts
    for (Function &F : M)
      for (Instruction &I : llvm::make_early_inc_range(instructions(F)))
        if (auto *AC = dyn_cast<AddrSpaceCastInst>(&I); AC && AC->getType() == AC->getOperand(0)->getType()) {
          AC->replaceAllUsesWith(AC->getOperand(0)); AC->eraseFromParent(); Changed = true;
        }
  }
  return true;
}

bool adaptFunction(Function &F, bool Block1, raw_ostream &Err) {
  Module &M = *F.getParent();
  LLVMContext &C = M.getContext();
  F.setCallingConv(CallingConv::C);
  // kernel marker + argument address spaces (1 = global pointer)
  SmallVector<Metadata *, 8> AS;
  for (Argument &A : F.args()) AS.push_back(ConstantAsMetadata::get(ConstantInt::get(Type::getInt32Ty(C), A.getType()->isPointerTy() ? 1 : 0)));
  F.setMetadata("kernel_arg_addr_space", MDNode::get(C, AS));
  for (StringRef A : {"nvvm.reqntid", "nvvm.maxntid", "nvvm.kernel", "nvvm.minctasm", "nvvm.maxnreg", "nvvm.cluster_dim"}) F.removeFnAttr(A);

  SmallVector<CallInst *, 16> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (CI->getCalledFunction()) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    StringRef N = CI->getCalledFunction()->getName();
    IRBuilder<> B(CI);
    Value *R = nullptr;
    auto dimOf = [&](StringRef S) -> char { return S.empty() ? 0 : S.back(); };
    if (N.consume_front("llvm.nvvm.read.ptx.sreg.")) {
      StringRef Reg = N.substr(0, N.rfind('.'));
      char D = dimOf(N);
      if (D != 'x' && D != 'y' && D != 'z') { Err << "hwacha-cc: unsupported NVVM special register " << N << "\n"; return false; }
      bool X = D == 'x';
      if (Reg == "tid")        R = !X ? B.getInt32(0) : Block1 ? B.getInt32(0) : builtinCall(M, B, "_Z12get_local_idj");
      else if (Reg == "ntid")  R = !X ? B.getInt32(1) : Block1 ? B.getInt32(1) : builtinCall(M, B, "_Z14get_local_sizej");
      else if (Reg == "ctaid") R = !X ? B.getInt32(0) : Block1 ? builtinCall(M, B, "_Z13get_global_idj") : builtinCall(M, B, "_Z12get_group_idj");
      else if (Reg == "nctaid" && !X) R = B.getInt32(1);
      else if (Reg == "nctaid") {   // grid size: the lowered gpu.launch_func stores it in hwacha_grid_size
        Type *I64 = Type::getInt64Ty(C);
        GlobalVariable *GS = M.getGlobalVariable("hwacha_grid_size", true);
        if (!GS) { GS = new GlobalVariable(M, I64, false, GlobalValue::ExternalLinkage, nullptr, "hwacha_grid_size"); GS->setAlignment(Align(8)); }
        R = B.CreateTrunc(B.CreateLoad(I64, GS), B.getInt32Ty());
      } else { Err << "hwacha-cc: unsupported NVVM special register " << N << "\n"; return false; }
    } else if (N.starts_with("llvm.nvvm.barrier")) {
      FunctionType *FT = FunctionType::get(Type::getVoidTy(C), {Type::getInt32Ty(C)}, false);
      Function *Bar = cast<Function>(M.getOrInsertFunction("_Z7barrierj", FT).getCallee());
      Bar->setConvergent();
      B.CreateCall(Bar, {B.getInt32(1)});
      CI->eraseFromParent();
      continue;
    } else if (N.starts_with("llvm.nvvm.")) {
      Err << "hwacha-cc: unsupported NVVM intrinsic " << N << "\n"; return false;
    } else if (N.starts_with("__nv_")) {
      // libdevice math -> libm name (hwacha-cc vectorizes expf/sqrtf/... by name)
      Function *Callee = CI->getCalledFunction();
      StringRef Base = N.drop_front(5);
      bool F32 = Base.ends_with("f") && Base != "fabs"; // fabs (f64) vs fabsf
      StringRef Root = F32 ? Base.drop_back() : Base;
      Intrinsic::ID ID = StringSwitch<Intrinsic::ID>(Root)
          .Case("sqrt", Intrinsic::sqrt).Case("fma", Intrinsic::fma).Case("fabs", Intrinsic::fabs)
          .Case("fmin", Intrinsic::minnum).Case("fmax", Intrinsic::maxnum).Case("floor", Intrinsic::floor)
          .Case("ceil", Intrinsic::ceil).Case("trunc", Intrinsic::trunc).Case("rint", Intrinsic::rint)
          .Case("exp", Intrinsic::exp).Case("exp2", Intrinsic::exp2).Case("log", Intrinsic::log)
          .Case("log2", Intrinsic::log2).Case("sin", Intrinsic::sin).Case("cos", Intrinsic::cos)
          .Case("pow", Intrinsic::pow).Default(Intrinsic::not_intrinsic);
      if (ID != Intrinsic::not_intrinsic && CI->getType()->isFloatingPointTy()) {
        SmallVector<Value *, 3> Args(CI->args());
        R = B.CreateIntrinsic(CI->getType(), ID, Args);
      } else {
        FunctionCallee Lib = M.getOrInsertFunction(Base, Callee->getFunctionType(), Callee->getAttributes());
        CI->setCalledFunction(Lib);
        continue;
      }
    } else continue;
    CI->replaceAllUsesWith(R);
    CI->eraseFromParent();
  }
  return true;
}

// group_id*local_size + local_id is the global id: fold it so the address becomes a unit-stride stream.
Value *stripIdCast(Value *V) {
  while (auto *C = dyn_cast<CastInst>(V)) {
    if (!isa<TruncInst>(C) && !isa<SExtInst>(C) && !isa<ZExtInst>(C)) break;
    V = C->getOperand(0);
  }
  return V;
}
void foldGlobalId(Function &F) {
  using namespace PatternMatch;
  SmallVector<Instruction *, 8> Work;
  for (Instruction &I : instructions(F)) {
    Value *A, *B, *C;
    if (!match(&I, m_c_Add(m_c_Mul(m_Value(A), m_Value(B)), m_Value(C)))) continue;
    A = stripIdCast(A); B = stripIdCast(B); C = stripIdCast(C);
    if (!hwacha::isLocalId(C)) continue;
    if (!((hwacha::isGroupIdCall(A) && hwacha::isLocalSizeCall(B)) || (hwacha::isGroupIdCall(B) && hwacha::isLocalSizeCall(A)))) continue;
    Work.push_back(&I);
  }
  for (Instruction *I : Work) {
    IRBuilder<> Bld(I);
    CallInst *G = Bld.CreateCall(builtinDecl(*F.getParent(), "_Z13get_global_idj"), {Bld.getInt32(0)});
    LLVMContext &C = F.getContext(); Type *I64 = Type::getInt64Ty(C);
    Metadata *R[] = {ConstantAsMetadata::get(ConstantInt::get(I64, 0)), ConstantAsMetadata::get(ConstantInt::get(I64, 1u << 30))};
    G->setMetadata(LLVMContext::MD_range, MDNode::get(C, R));
    Value *Rep = Bld.CreateZExtOrTrunc(G, I->getType());
    I->replaceAllUsesWith(Rep);
    RecursivelyDeleteTriviallyDeadInstructions(I);
  }
}

void runO2(Module &M);
} // namespace
void hwacha::optimizeModule(Module &M) { runO2(M); }
namespace {
void runO2(Module &M) {
  PipelineTuningOptions PTO;
  PTO.LoopVectorization = false; PTO.SLPVectorization = false; PTO.LoopUnrolling = false;
  PassBuilder PB(nullptr, PTO);
  LoopAnalysisManager LAM; FunctionAnalysisManager FAM; CGSCCAnalysisManager CGAM; ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM); PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM); PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OptimizationLevel::O2);
  MPM.run(M, MAM);
}

} // namespace

// Returns false on error; sets Adapted when the module came from the GPU dialect.
bool hwacha::adaptGPUModule(Module &M, bool ForceBlock1, bool NoOpt, raw_ostream &Err, bool &Adapted) {
  Adapted = false;
  SmallVector<Function *, 8> Kernels;
  for (Function &F : M)
    if (!F.isDeclaration() && (F.getCallingConv() == CallingConv::PTX_Kernel || F.hasFnAttribute("nvvm.kernel"))) Kernels.push_back(&F);
  if (Kernels.empty()) return true;
  Adapted = true;
  // the MLIR module carries no target: give it the RISC-V one the rest of the pipeline expects
  if (M.getTargetTriple().empty() || M.getTargetTriple().str().find("riscv64") == std::string::npos)
    M.setTargetTriple(Triple("riscv64-unknown-elf"));
  if (M.getDataLayoutStr().empty() || M.getDataLayoutStr().find("n32:64") == std::string::npos)
    M.setDataLayout("e-m:e-p:64:64-i64:64-i128:128-n32:64-S128");
  if (!flattenWorkgroupGlobals(M, Err)) return false;
  for (Function *F : Kernels) if (!adaptFunction(*F, ForceBlock1 || blockSizeIsOne(*F), Err)) return false;
  for (Function *F : Kernels) foldGlobalId(*F);
  // leftover NVVM declarations
  for (Function &F : llvm::make_early_inc_range(M)) if (F.isDeclaration() && F.use_empty() && F.getName().starts_with("llvm.nvvm.")) F.eraseFromParent();
  if (verifyModule(M, &Err)) return false;
  if (!NoOpt) runO2(M);
  for (Function *F : Kernels) foldGlobalId(*F);
  return true;
}

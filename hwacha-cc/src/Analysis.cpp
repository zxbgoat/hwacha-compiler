#include "Analysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Operator.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace hwacha;

bool hwacha::isKernel(const Function &F) {
  return !F.isDeclaration() && F.hasMetadata("kernel_arg_addr_space") &&
         !F.getName().starts_with("__clang_ocl_kern_imp_");
}

static const CallBase *dim0Call(const Value *V, StringRef Name) {
  auto *CB = dyn_cast<CallBase>(V);
  if (!CB || !CB->getCalledFunction() || CB->getCalledFunction()->getName() != Name) return nullptr;
  auto *Dim = dyn_cast<ConstantInt>(CB->getArgOperand(0));
  return (Dim && Dim->isZero()) ? CB : nullptr;   // only dimension 0 is supported
}
bool hwacha::isGlobalId(const Value *V) { return dim0Call(V, "_Z13get_global_idj") || dim0Call(V, "hwacha.veidx"); }
bool hwacha::isLocalId(const Value *V) { return dim0Call(V, "_Z12get_local_idj"); }
bool hwacha::isWorkItemId(const Value *V) { return isGlobalId(V) || isLocalId(V); }
bool hwacha::isLocalSizeCall(const Value *V) { return dim0Call(V, "_Z14get_local_sizej"); }
bool hwacha::isGroupIdCall(const Value *V) { return dim0Call(V, "_Z12get_group_idj"); }
bool hwacha::isBarrierCall(const Value *V) {
  auto *CB = dyn_cast<CallBase>(V);
  return CB && CB->getCalledFunction() && CB->getCalledFunction()->getName() == "_Z7barrierj";
}

namespace {
// Minimal TTI whose only job is to tell UniformityAnalysis where divergence comes from.
struct HwachaTTIImpl : public TargetTransformInfoImplBase {
  explicit HwachaTTIImpl(const DataLayout &DL) : TargetTransformInfoImplBase(DL) {}
  bool hasBranchDivergence(const Function *) const override { return true; }
  ValueUniformity getValueUniformity(const Value *V) const override {
    return isWorkItemId(V) ? ValueUniformity::NeverUniform : ValueUniformity::Default;
  }
};
} // namespace

// A Hwacha work-item id always fits in 31 bits (vector lengths are far smaller); telling
// SCEV so lets it fold the trunc/sext pairs that "int i = get_global_id(0)" produces.

// int i = get_global_id(0); a[i*n + j]: clang computes the index in i32 (trunc id; mul/add nsw; sext or
// zext nneg back to i64). SCEV cannot push the extension through the i32 arithmetic (the nsw flags
// only count inside a UB context), so the address looks non-linear in id and classifies as a gather.
// sext of nsw arithmetic equals nsw arithmetic of the sext'ed operands, and the id is < 2^30, so rebuild
// every such chain in i64 and drop the extension.
static bool chainHasIdTrunc(Value *V, unsigned Depth = 0) {
  if (Depth > 16) return false;
  if (auto *T = dyn_cast<TruncInst>(V)) return T->getSrcTy()->isIntegerTy(64) && isWorkItemId(T->getOperand(0));
  auto *BO = dyn_cast<OverflowingBinaryOperator>(V);
  if (!BO || !BO->hasNoSignedWrap()) return false;
  unsigned Op = BO->getOpcode();
  if (Op != Instruction::Add && Op != Instruction::Sub && Op != Instruction::Mul && Op != Instruction::Shl) return false;
  return chainHasIdTrunc(BO->getOperand(0), Depth + 1) || chainHasIdTrunc(BO->getOperand(1), Depth + 1);
}
static Value *widenToI64(Value *V, Function &F, DenseMap<Value *, Value *> &Cache) {
  if (auto It = Cache.find(V); It != Cache.end()) return It->second;
  Type *I64 = Type::getInt64Ty(F.getContext());
  Value *R = nullptr;
  if (auto *C = dyn_cast<ConstantInt>(V)) R = ConstantInt::get(I64, C->getSExtValue());
  else if (auto *T = dyn_cast<TruncInst>(V); T && T->getSrcTy() == I64 && isWorkItemId(T->getOperand(0))) R = T->getOperand(0);
  else if (auto *BO = dyn_cast<OverflowingBinaryOperator>(V); BO && BO->hasNoSignedWrap() && chainHasIdTrunc(V)) {
    Value *A = widenToI64(BO->getOperand(0), F, Cache), *B = widenToI64(BO->getOperand(1), F, Cache);
    IRBuilder<> Bld(cast<Instruction>(V));
    R = Bld.CreateBinOp((Instruction::BinaryOps)BO->getOpcode(), A, B);
    if (auto *NB = dyn_cast<BinaryOperator>(R)) NB->setHasNoSignedWrap();
  } else {
    Instruction *IP;
    if (auto *P = dyn_cast<PHINode>(V)) IP = &*P->getParent()->getFirstInsertionPt();
    else if (auto *VI = dyn_cast<Instruction>(V)) IP = VI->getNextNode();
    else IP = &*F.getEntryBlock().getFirstInsertionPt();
    IRBuilder<> Bld(IP);
    R = Bld.CreateSExt(V, I64);
  }
  Cache[V] = R;
  return R;
}
static void widenIdArithmetic(Function &F) {
  SmallVector<CastInst *, 8> Exts;
  for (Instruction &I : instructions(F)) {
    auto *CI = dyn_cast<CastInst>(&I);
    if (!CI || !CI->getSrcTy()->isIntegerTy(32) || !CI->getDestTy()->isIntegerTy(64)) continue;
    bool Signed = isa<SExtInst>(CI) || (isa<ZExtInst>(CI) && cast<PossiblyNonNegInst>(CI)->hasNonNeg());
    if (Signed && chainHasIdTrunc(CI->getOperand(0))) Exts.push_back(CI);
  }
  DenseMap<Value *, Value *> Cache;
  for (CastInst *CI : Exts) {
    Value *W = widenToI64(CI->getOperand(0), F, Cache);
    CI->replaceAllUsesWith(W);
    Value *Src = CI->getOperand(0);
    CI->eraseFromParent();
    RecursivelyDeleteTriviallyDeadInstructions(Src);
  }
}

static void annotateIdRange(Function &F) {
  LLVMContext &C = F.getContext();
  for (Instruction &I : instructions(F))
    if (isWorkItemId(&I) && !I.getMetadata(LLVMContext::MD_range)) {
      Type *T = I.getType();
      Metadata *R[] = {ConstantAsMetadata::get(ConstantInt::get(T, 0)),
                       ConstantAsMetadata::get(ConstantInt::get(T, 1u << 30))};
      I.setMetadata(LLVMContext::MD_range, MDNode::get(C, R));
    }
}

KernelAnalysis::KernelAnalysis(Function &F)
    : F(F), TLII(F.getParent()->getTargetTriple()) {
  widenIdArithmetic(F);
  annotateIdRange(F);
  TLI = std::make_unique<TargetLibraryInfo>(TLII);
  AC = std::make_unique<AssumptionCache>(F);
  DT = std::make_unique<DominatorTree>(F);
  LI = std::make_unique<LoopInfo>(*DT);
  SE = std::make_unique<ScalarEvolution>(F, *TLI, *AC, *DT, *LI);
  TTI = std::make_unique<TargetTransformInfo>(
      std::make_unique<HwachaTTIImpl>(F.getParent()->getDataLayout()));
  CI = std::make_unique<CycleInfo>();
  CI->compute(F);
  UI = std::make_unique<UniformityInfo>(*DT, *CI, TTI.get());
  UI->compute();
  classifyAccesses();
}

KernelAnalysis::~KernelAnalysis() = default;

void KernelAnalysis::recomputeUniformity() {
  // ScalarEvolution keeps a reference to the dominator tree: refresh it in place
  DT->recalculate(F);
  CI = std::make_unique<CycleInfo>(); CI->compute(F);
  UI = std::make_unique<UniformityInfo>(*DT, *CI, TTI.get());
  UI->compute();
}

bool KernelAnalysis::isUniform(const Value *V) const { return UI->isUniformAtDef(V); }
bool KernelAnalysis::hasDivergentBranch(const BasicBlock *BB) const {
  return UI->hasDivergentTerminator(*BB);  // non-const in LLVM; UI is a pointer member
}

// Write S as Base + Stride * id for one flavour of id (global or local), where Base and Stride
// are id-free:  Stride = S[id := id+1] - S[id],  Base = S[id := 0].  A pointer that depends on
// both flavours (x[gid + lid]) is not a stream.
bool KernelAnalysis::decompose(const SCEV *S, const SCEV *&Base, const SCEV *&Stride, bool &Local) {
  auto containsId = [&](const SCEV *X) {
    return SCEVExprContains(X, [&](const SCEV *Y) {
      auto *U = dyn_cast<SCEVUnknown>(Y);
      if (!U) return false;
      if (isWorkItemId(U->getValue())) return true;
      auto *UI_ = dyn_cast<Instruction>(U->getValue());   // sdiv/srem/select... of the id: opaque to SCEV but divergent
      return UI_ && !UI->isUniformAtDef(static_cast<const Value *>(UI_)); });
  };
  for (int Flavour = 0; Flavour < 2; Flavour++) {
    bool WantLocal = Flavour == 1;
    ValueToSCEVMapTy PlusOne, Zero; bool Any = false;
    for (Instruction &I : instructions(F)) {
      if (!(WantLocal ? isLocalId(&I) : isGlobalId(&I))) continue;
      const SCEV *IdS = SE->getSCEV(&I); Any = true;
      PlusOne[&I] = SE->getAddExpr(IdS, SE->getOne(IdS->getType()));
      Zero[&I] = SE->getZero(IdS->getType());
    }
    if (!Any) continue;
    const SCEV *S1 = SCEVParameterRewriter::rewrite(S, *SE, PlusOne);
    const SCEV *D = SE->getMinusSCEV(S1, S);
    if (D->isZero()) continue;                                   // does not depend on this flavour
    if (containsId(D)) return false;                            // not linear
    const SCEV *B = SCEVParameterRewriter::rewrite(S, *SE, Zero);
    if (containsId(B)) return false;                            // the other flavour is in there too
    Base = B; Stride = D; Local = WantLocal;
    return true;
  }
  return false;
}

void KernelAnalysis::classifyAccesses() {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB) {
      Value *Ptr = nullptr;
      if (auto *L = dyn_cast<LoadInst>(&I)) Ptr = L->getPointerOperand();
      else if (auto *St = dyn_cast<StoreInst>(&I)) Ptr = St->getPointerOperand();
      else continue;
      MemAccess A; A.I = &I;
      const SCEV *S = SE->getSCEV(Ptr);
      const SCEV *Base; const SCEV *Stride; bool Local = false;
      if (isUniform(Ptr)) { A.Kind = AddrKind::Uniform; A.BaseSCEV = S; }
      else if (decompose(S, Base, Stride, Local) && !Stride->isZero()) {
        A.Kind = AddrKind::Stream; A.BaseSCEV = Base; A.StrideSCEV = Stride; A.Local = Local;
        if (auto *C = dyn_cast<SCEVConstant>(Stride)) A.Stride = C->getAPInt().getSExtValue();
      }
      else { A.Kind = AddrKind::Gather; A.Index = Ptr; }
      Accesses.push_back(A);
    }
}

void KernelAnalysis::print(raw_ostream &OS) const {
  OS << "== kernel " << F.getName() << "\n";
  for (const Argument &A : F.args())
    OS << "  arg " << A.getName() << " #" << A.getArgNo() << ": " << (isUniform(&A) ? "uniform" : "DIVERGENT") << "\n";
  for (const BasicBlock &BB : F) {
    OS << "  block " << BB.getName() << (hasDivergentBranch(&BB) ? "  [divergent terminator]" : "") << "\n";
    for (const Instruction &I : BB) {
      OS << "    " << (I.getType()->isVoidTy() ? "  -   " : (isUniform(&I) ? "  vs  " : "  vv  "));
      I.print(OS, /*IsForDebug=*/true); OS << "\n";
    }
  }
  for (const MemAccess &A : Accesses) {
    OS << "  mem: ";
    switch (A.Kind) {
    case AddrKind::Stream:  OS << (A.Local ? "LSTREAM stride=" : "STREAM  stride="); if (A.Stride) OS << A.Stride; else A.StrideSCEV->print(OS); OS << " base="; A.BaseSCEV->print(OS); break;
    case AddrKind::Uniform: OS << "UNIFORM base="; A.BaseSCEV->print(OS); break;
    case AddrKind::Gather:  OS << "GATHER  index=" << *A.Index; break;
    }
    OS << "   <- " << *A.I << "\n";
  }
}

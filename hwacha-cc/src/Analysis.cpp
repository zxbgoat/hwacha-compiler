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
bool hwacha::isWorkGroupReduce(const Value *V, StringRef *Op) {
  auto *CB = dyn_cast<CallBase>(V);
  if (!CB || !CB->getCalledFunction() || CB->arg_size() != 1) return false;
  StringRef N = CB->getCalledFunction()->getName();
  if (!N.consume_front("_Z21work_group_reduce_")) return false;
  StringRef O = N.take_front(3);
  if (O != "add" && O != "min" && O != "max") return false;
  if (Op) *Op = O;
  return true;
}
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
    if (isWorkItemId(V)) return ValueUniformity::NeverUniform;
    if (isWorkGroupReduce(V)) return ValueUniformity::AlwaysUniform;   // the reduced value is the same in every lane
    return ValueUniformity::Default;
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

// expf(x) as straight-line vector arithmetic (same algorithm as test/llama/hwacha_math.h hw_expf):
// x = k*ln2 + r with k = round(x*log2e) via the 1.5*2^23 magic add, degree-6 polynomial in r,
// scale by 2^k by adding k to the exponent field. Applies to expf / llvm.exp.f32 / _Z3expf / __nv_expf.
static bool isExpfCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return false;
  StringRef N = Callee->getName();
  return N == "expf" || N == "llvm.exp.f32" || N == "_Z3expf" || N == "__nv_expf";
}
// tanhf(x) = 1 - 2/(exp(2x) + 1). Emits an llvm.exp.f32 (which clamps its argument), so run before
// expandExpf. Saturates correctly: 2x above ~88 -> exp huge -> ~1; below ~-87 -> exp ~0 -> -1.
static bool isTanhfCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return false;
  StringRef N = Callee->getName();
  return N == "tanhf" || N == "llvm.tanh.f32" || N == "_Z4tanhf" || N == "__nv_tanhf";
}
void hwacha::expandTanhf(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (isTanhfCall(CI)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    Value *X = CI->getArgOperand(0);
    Value *E = B.CreateIntrinsic(Intrinsic::exp, {FT}, {B.CreateFMul(X, C(2.0))});
    Value *R = B.CreateFSub(C(1.0), B.CreateFDiv(C(2.0), B.CreateFAdd(E, C(1.0))));
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// floorf(x) for the value range models actually use (upsample index math): trunc toward zero, then step
// down by one where truncation rounded up (negatives with a fraction). Pure vector arithmetic.
static bool isFloorfCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return false;
  StringRef N = Callee->getName();
  return N == "floorf" || N == "llvm.floor.f32" || N == "_Z5floorf" || N == "__nv_floorf";
}
void hwacha::expandFloorf(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (isFloorfCall(CI)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy(), *IT = B.getInt32Ty();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    Value *X = CI->getArgOperand(0);
    Value *T = B.CreateSIToFP(B.CreateFPToSI(X, IT), FT);          // trunc toward zero
    Value *R = B.CreateSelect(B.CreateFCmpOGT(T, X), B.CreateFSub(T, C(1.0)), T);
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// logf(x) as straight-line vector arithmetic (Cephes single-precision logf): frexp x = m * 2^e with
// m in [sqrt(1/2), sqrt(2)), then a degree-8 polynomial in (m-1). No calls, so it needs no follow-up pass.
static bool isLogfCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return false;
  StringRef N = Callee->getName();
  return N == "logf" || N == "llvm.log.f32" || N == "_Z4logf" || N == "__nv_logf";
}
void hwacha::expandLogf(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (isLogfCall(CI)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy(), *IT = B.getInt32Ty();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    Value *X = CI->getArgOperand(0);
    Value *IX = B.CreateBitCast(X, IT);
    // frexp: e = ((ix >> 23) & 0xff) - 126; m = (ix & 0x807fffff) | 0x3f000000  -> m in [0.5, 1)
    Value *E = B.CreateSub(B.CreateAnd(B.CreateAShr(IX, 23), B.getInt32(0xff)), B.getInt32(126));
    Value *M = B.CreateBitCast(B.CreateOr(B.CreateAnd(IX, B.getInt32(0x807fffff)), B.getInt32(0x3f000000)), FT);
    Value *EF = B.CreateSIToFP(E, FT);
    // if m < SQRTHF (0.70710678): e -= 1; m = m + m - 1; else m -= 1
    Value *Lt = B.CreateFCmpOLT(M, C(0.70710678118654752440));
    EF = B.CreateSelect(Lt, B.CreateFSub(EF, C(1.0)), EF);
    Value *M2 = B.CreateSelect(Lt, B.CreateFSub(B.CreateFAdd(M, M), C(1.0)), B.CreateFSub(M, C(1.0)));
    Value *Z = B.CreateFMul(M2, M2);
    Value *Y = C(7.0376836292E-2);
    for (double c : {-1.1514610310E-1, 1.1676998740E-1, -1.2420140846E-1, 1.4249322787E-1,
                     -1.6668057665E-1, 2.0000714765E-1, -2.4999993993E-1, 3.3333331174E-1})
      Y = B.CreateFAdd(B.CreateFMul(Y, M2), C(c));
    Y = B.CreateFMul(B.CreateFMul(Y, M2), Z);
    Y = B.CreateFAdd(Y, B.CreateFMul(EF, C(-2.12194440E-4)));
    Y = B.CreateFSub(Y, B.CreateFMul(Z, C(0.5)));
    Value *R = B.CreateFAdd(M2, Y);
    R = B.CreateFAdd(R, B.CreateFMul(EF, C(0.693359375)));
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// log1pf/expm1f/powf via the exp and log primitives above (adequate for the tolerances here); each emits
// an llvm.log.f32 / llvm.exp.f32, so run these before expandLogf / expandExpf.
static bool isNamed(const CallInst *CI, std::initializer_list<StringRef> names) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee) return false;
  StringRef N = Callee->getName();
  for (StringRef n : names) if (N == n) return true;
  return false;
}
void hwacha::expandLogExpM1Pow(Function &F) {
  SmallVector<CallInst *, 8> log1p, expm1, pow, powi;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (CI->getType()->isFloatTy()) {
      if (CI->arg_size() == 1 && isNamed(CI, {"log1pf", "llvm.log1p.f32", "__nv_log1pf"})) log1p.push_back(CI);
      else if (CI->arg_size() == 1 && isNamed(CI, {"expm1f", "llvm.expm1.f32", "__nv_expm1f"})) expm1.push_back(CI);
      else if (CI->arg_size() == 2 && isNamed(CI, {"powf", "llvm.pow.f32", "__nv_powf"})) pow.push_back(CI);
      else if (CI->arg_size() == 2 && isNamed(CI, {"powif", "llvm.powi.f32", "__nv_powif"})) powi.push_back(CI);
    }
  }
  auto FT = Type::getFloatTy(F.getContext());
  for (CallInst *CI : powi) {     // integer power: constant exponent -> repeated multiply, else exp(n*log(x))
    IRBuilder<> B(CI);
    Value *X = CI->getArgOperand(0), *N = CI->getArgOperand(1), *R;
    if (auto *CN = dyn_cast<ConstantInt>(N)) {
      int64_t n = CN->getSExtValue(), an = n < 0 ? -n : n;
      if (an == 0) R = ConstantFP::get(FT, 1.0);
      else { R = X; for (int64_t i = 1; i < an; i++) R = B.CreateFMul(R, X); }
      if (n < 0) R = B.CreateFDiv(ConstantFP::get(FT, 1.0), R);
    } else
      R = B.CreateIntrinsic(Intrinsic::exp, {FT}, {B.CreateFMul(B.CreateSIToFP(N, FT), B.CreateIntrinsic(Intrinsic::log, {FT}, {X}))});
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
  for (CallInst *CI : log1p) {   // log1p(x) = log(1 + x)
    IRBuilder<> B(CI);
    Value *R = B.CreateIntrinsic(Intrinsic::log, {FT}, {B.CreateFAdd(CI->getArgOperand(0), ConstantFP::get(FT, 1.0))});
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
  for (CallInst *CI : expm1) {   // expm1(x) = exp(x) - 1
    IRBuilder<> B(CI);
    Value *R = B.CreateFSub(B.CreateIntrinsic(Intrinsic::exp, {FT}, {CI->getArgOperand(0)}), ConstantFP::get(FT, 1.0));
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
  for (CallInst *CI : pow) {      // pow(x, y) = exp(y * log(x))  (bases are positive here)
    IRBuilder<> B(CI);
    Value *L = B.CreateIntrinsic(Intrinsic::log, {FT}, {CI->getArgOperand(0)});
    Value *R = B.CreateIntrinsic(Intrinsic::exp, {FT}, {B.CreateFMul(CI->getArgOperand(1), L)});
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// erff(x) via Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7): erf(x) = sign(x) * (1 - p(t)*exp(-x^2)),
// t = 1/(1 + 0.3275911*|x|). Emits an llvm.exp.f32 that expandExpf then inlines, so run this first.
static bool isErffCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return false;
  StringRef N = Callee->getName();
  return N == "erff" || N == "llvm.erf.f32" || N == "_Z3erff" || N == "__nv_erff";
}
void hwacha::expandErff(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (isErffCall(CI)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    Value *X = CI->getArgOperand(0);
    Value *Neg = B.CreateFCmpOLT(X, C(0.0));
    Value *A = B.CreateSelect(Neg, B.CreateFNeg(X), X);
    Value *S = B.CreateSelect(Neg, C(-1.0), C(1.0));
    Value *T = B.CreateFDiv(C(1.0), B.CreateFAdd(C(1.0), B.CreateFMul(C(0.3275911), A)));
    Value *P = C(1.061405429);
    for (double c : {-1.453152027, 1.421413741, -0.284496736, 0.254829592})
      P = B.CreateFAdd(B.CreateFMul(P, T), C(c));
    P = B.CreateFMul(P, T);
    Value *E = B.CreateIntrinsic(Intrinsic::exp, {FT}, {B.CreateFNeg(B.CreateFMul(A, A))});
    Value *Res = B.CreateFMul(S, B.CreateFSub(C(1.0), B.CreateFMul(P, E)));
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
  }
}
void hwacha::expandExpf(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (isExpfCall(CI)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy(), *IT = B.getInt32Ty();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    Value *X = CI->getArgOperand(0);
    X = B.CreateMinNum(B.CreateMaxNum(X, C(-87.0)), C(88.0));
    Value *T = B.CreateFAdd(B.CreateFMul(X, C(1.44269504088896341)), C(12582912.0));
    Value *K = B.CreateSub(B.CreateBitCast(T, IT), B.getInt32(0x4B400000));
    Value *KF = B.CreateFSub(T, C(12582912.0));
    Value *R = B.CreateFSub(X, B.CreateFMul(KF, C(0.693145751953125)));
    R = B.CreateFSub(R, B.CreateFMul(KF, C(1.428606765330187e-06)));
    Value *P = C(1.9875691500E-4);
    for (double c : {1.3981999507E-3, 8.3334519073E-3, 4.1665795894E-2, 1.6666665459E-1, 5.0000001201E-1})
      P = B.CreateFAdd(B.CreateFMul(P, R), C(c));
    P = B.CreateFAdd(B.CreateFAdd(B.CreateFMul(P, B.CreateFMul(R, R)), R), C(1.0));
    Value *Res = B.CreateBitCast(B.CreateAdd(B.CreateBitCast(P, IT), B.CreateShl(K, 23)), FT);
    CI->replaceAllUsesWith(Res);
    CI->eraseFromParent();
  }
}

// fadd(fmul(a, b), c) with a single-use product -> llvm.fmuladd (Hwacha's vfmadd), like -ffp-contract=fast.
void hwacha::contractFMA(Function &F) {
  SmallVector<BinaryOperator *, 16> Adds;
  for (Instruction &I : instructions(F))
    if (auto *BO = dyn_cast<BinaryOperator>(&I); BO && (BO->getOpcode() == Instruction::FAdd || BO->getOpcode() == Instruction::FSub) && BO->getType()->isFloatingPointTy())
      Adds.push_back(BO);
  for (BinaryOperator *BO : Adds) {
    bool Sub = BO->getOpcode() == Instruction::FSub;
    Value *A = BO->getOperand(0), *B = BO->getOperand(1);
    auto mul = [](Value *V) -> BinaryOperator * { auto *M = dyn_cast<BinaryOperator>(V); return M && M->getOpcode() == Instruction::FMul && M->hasOneUse() ? M : nullptr; };
    IRBuilder<> Bld(BO);
    Value *R = nullptr;
    if (BinaryOperator *M = mul(A)) {          // a*b + c  |  a*b - c
      Value *C = Sub ? Bld.CreateFNeg(B) : B;
      R = Bld.CreateIntrinsic(BO->getType(), Intrinsic::fmuladd, {M->getOperand(0), M->getOperand(1), C});
    } else if (BinaryOperator *M = mul(B)) {   // c + a*b  |  c - a*b
      Value *X = Sub ? Bld.CreateFNeg(M->getOperand(0)) : M->getOperand(0);
      R = Bld.CreateIntrinsic(BO->getType(), Intrinsic::fmuladd, {X, M->getOperand(1), A});
    } else continue;
    BO->replaceAllUsesWith(R);
    BO->eraseFromParent();
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

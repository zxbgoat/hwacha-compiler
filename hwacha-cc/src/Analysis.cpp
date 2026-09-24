#include "Analysis.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfoImpl.h"
#include "llvm/ADT/GenericUniformityImpl.h"
#include "llvm/IR/IntrinsicInst.h"
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

// sinf / cosf (SHOC's fft twiddles: exp_i(phi) = (cos(phi), sin(phi))) as straight-line arithmetic:
// k = round(x * 2/pi) (floor(t + 0.5) via truncation, corrected for negatives), r = x - k*pi/2 in
// two Cody-Waite parts (|x| up to a few thousand keeps r accurate to ~1e-6), Taylor polynomials on
// |r| <= pi/4 (degree 9 for sin, 8 for cos: error ~2e-7), quadrant q = k & 3 selects and negates.
// Applies to sinf / cosf, llvm.sin/cos.f32, _Z3sinf / _Z3cosf and their native_ / __nv_ variants.
static int sinCosKind(const CallInst *CI) {   // 1 = sin, 2 = cos, 0 = neither
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return 0;
  StringRef N = Callee->getName();
  if (N == "sinf" || N == "llvm.sin.f32" || N == "_Z3sinf" || N == "_Z10native_sinf" || N == "_Z8half_sinf" || N == "__nv_sinf") return 1;
  if (N == "cosf" || N == "llvm.cos.f32" || N == "_Z3cosf" || N == "_Z10native_cosf" || N == "_Z8half_cosf" || N == "__nv_cosf") return 2;
  return 0;
}
void hwacha::expandSinCosf(Function &F) {
  SmallVector<std::pair<CallInst *, int>, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (int K = sinCosKind(CI)) Calls.push_back({CI, K});
  for (auto &[CI, Kind] : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy(), *IT = B.getInt32Ty();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    Value *X = CI->getArgOperand(0);
    Value *T = B.CreateFAdd(B.CreateFMul(X, C(0.63661977236758134)), C(0.5));          // x * 2/pi + 0.5
    Value *Ti = B.CreateFPToSI(T, IT);
    Value *Tf = B.CreateSIToFP(Ti, FT);
    Value *Neg = B.CreateFCmpOGT(Tf, T);                                                // truncation rounded up: floor is one less
    Value *K = B.CreateSelect(Neg, B.CreateSub(Ti, ConstantInt::get(IT, 1)), Ti);
    Value *Kf = B.CreateSIToFP(K, FT);
    Value *R = B.CreateFSub(X, B.CreateFMul(Kf, C(1.5707963705062866)));               // pi/2 high part (float)
    R = B.CreateFSub(R, B.CreateFMul(Kf, C(-4.3711388286737929e-08)));                 // pi/2 low part
    Value *R2 = B.CreateFMul(R, R);
    Value *S = B.CreateFAdd(B.CreateFMul(R2, C(2.7557319223985893e-06)), C(-1.9841269841269841e-04));   // sin: r (1 - r2/6 + r4/120 - r6/5040 + r8/362880)
    S = B.CreateFAdd(B.CreateFMul(S, R2), C(8.3333333333333332e-03));
    S = B.CreateFAdd(B.CreateFMul(S, R2), C(-1.6666666666666666e-01));
    S = B.CreateFMul(R, B.CreateFAdd(B.CreateFMul(S, R2), C(1.0)));
    Value *Co = B.CreateFAdd(B.CreateFMul(R2, C(2.4801587301587302e-05)), C(-1.3888888888888889e-03));   // cos: 1 - r2/2 + r4/24 - r6/720 + r8/40320
    Co = B.CreateFAdd(B.CreateFMul(Co, R2), C(4.1666666666666664e-02));
    Co = B.CreateFAdd(B.CreateFMul(Co, R2), C(-0.5));
    Co = B.CreateFAdd(B.CreateFMul(Co, R2), C(1.0));
    Value *Q = B.CreateAnd(K, ConstantInt::get(IT, 3));
    Value *Odd = B.CreateICmpNE(B.CreateAnd(Q, ConstantInt::get(IT, 1)), ConstantInt::get(IT, 0));
    Value *Hi = B.CreateICmpNE(B.CreateAnd(Q, ConstantInt::get(IT, 2)), ConstantInt::get(IT, 0));
    Value *Res;
    if (Kind == 1) { Res = B.CreateSelect(Odd, Co, S); Res = B.CreateSelect(Hi, B.CreateFNeg(Res), Res); }             // sin: q0 s, q1 c, q2 -s, q3 -c
    else { Res = B.CreateSelect(Odd, B.CreateFNeg(S), Co); Res = B.CreateSelect(Hi, B.CreateFNeg(Res), Res); }          // cos: q0 c, q1 -s, q2 -c, q3 s
    CI->replaceAllUsesWith(Res); CI->eraseFromParent();
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
// rint / nearbyint / roundeven (round half to even: torch.round, torch.special.round) and round (half
// away from zero) are built on the same trunc-and-fix floor; |x| < 2^31.
static int roundKind(const CallInst *CI) {   // 0: not a rounding call, 1: floor, 2: half to even, 3: half away from zero
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return 0;
  StringRef N = Callee->getName();
  if (isFloorfCall(CI)) return 1;
  if (N == "llvm.rint.f32" || N == "llvm.nearbyint.f32" || N == "llvm.roundeven.f32" || N == "rintf" || N == "nearbyintf" || N == "roundevenf" || N == "_Z4rintf") return 2;
  if (N == "llvm.round.f32" || N == "roundf" || N == "_Z5roundf" || N == "__nv_roundf") return 3;
  return 0;
}
void hwacha::expandFloorf(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (roundKind(CI)) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Type *FT = B.getFloatTy(), *IT = B.getInt32Ty();
    auto C = [&](double v) { return ConstantFP::get(FT, v); };
    auto floorOf = [&](Value *X) {
      Value *T = B.CreateSIToFP(B.CreateFPToSI(X, IT), FT);          // trunc toward zero
      return B.CreateSelect(B.CreateFCmpOGT(T, X), B.CreateFSub(T, C(1.0)), T);
    };
    Value *X = CI->getArgOperand(0), *R;
    int K = roundKind(CI);
    if (K == 1) R = floorOf(X);
    else if (K == 2) {   // f = floor(x + 0.5); a tie (x + 0.5 integral) with f odd rounds down to the even f - 1
      Value *T = B.CreateFAdd(X, C(0.5)), *Fl = floorOf(T);
      Value *Half = B.CreateFMul(Fl, C(0.5)), *Odd = B.CreateFCmpOEQ(B.CreateFSub(Fl, B.CreateFMul(floorOf(Half), C(2.0))), C(1.0));
      Value *Tie = B.CreateFCmpOEQ(T, Fl);
      R = B.CreateSelect(B.CreateAnd(Tie, Odd), B.CreateFSub(Fl, C(1.0)), Fl);
    } else {             // trunc(x + copysign(0.5, x))
      Value *T = B.CreateFAdd(X, B.CreateSelect(B.CreateFCmpOLT(X, C(0.0)), C(-0.5), C(0.5)));
      R = B.CreateSIToFP(B.CreateFPToSI(T, IT), FT);
    }
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// llvm.abs.iN(x) -> x < 0 ? -x : x. NVVM canonicalization forms this integer-abs intrinsic (from
// reflection padding's index math); hwacha-cc's codegen has no call lowering, so expand it to arith here.
void hwacha::expandAbsI(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) {
    const Function *Callee = CI->getCalledFunction();
    if (Callee && Callee->getName().starts_with("llvm.abs.") && CI->getType()->isIntegerTy()) Calls.push_back(CI);
  }
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    Value *X = CI->getArgOperand(0);
    Value *Neg = B.CreateNeg(X);
    Value *R = B.CreateSelect(B.CreateICmpSLT(X, ConstantInt::get(X->getType(), 0)), Neg, X);
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// llvm.fshl / llvm.fshr (rotates: SHOC's md5 LEFTROTATE, InstCombine forms them) -> shifts and an or:
// fshl(a, b, c) = (a << c) | (b >> (w - c)), fshr(a, b, c) = (a >> c) | (b << (w - c)), c mod w.
void hwacha::expandFunnelShift(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) {
    const Function *Callee = CI->getCalledFunction();
    if (Callee && (Callee->getName().starts_with("llvm.fshl.") || Callee->getName().starts_with("llvm.fshr.")) && CI->getType()->isIntegerTy()) Calls.push_back(CI);
  }
  for (CallInst *CI : Calls) {
    IRBuilder<> B(CI);
    bool Left = CI->getCalledFunction()->getName().starts_with("llvm.fshl.");
    Value *A = CI->getArgOperand(0), *Bv = CI->getArgOperand(1), *C = CI->getArgOperand(2);
    Type *T = CI->getType(); unsigned W = T->getIntegerBitWidth();
    Value *Sh = B.CreateAnd(C, ConstantInt::get(T, W - 1));
    Value *Inv = B.CreateAnd(B.CreateSub(ConstantInt::get(T, W), Sh), ConstantInt::get(T, W - 1));
    // c == 0 (mod w): the result is a (fshl) / b (fshr); the (w - 0) shift would be undefined
    Value *Zero = B.CreateICmpEQ(Sh, ConstantInt::get(T, 0));
    Value *R = Left ? B.CreateOr(B.CreateShl(A, Sh), B.CreateLShr(Bv, Inv)) : B.CreateOr(B.CreateLShr(Bv, Sh), B.CreateShl(A, Inv));
    R = B.CreateSelect(Zero, Left ? A : Bv, R);
    CI->replaceAllUsesWith(R); CI->eraseFromParent();
  }
}

// logf(x) as straight-line vector arithmetic (Cephes single-precision logf): frexp x = m * 2^e with
// m in [sqrt(1/2), sqrt(2)), then a degree-8 polynomial in (m-1). No calls, so it needs no follow-up pass.
static bool isLogfCall(const CallInst *CI) {
  const Function *Callee = CI->getCalledFunction();
  if (!Callee || CI->arg_size() != 1 || !CI->getType()->isFloatTy() || !CI->getArgOperand(0)->getType()->isFloatTy()) return false;
  StringRef N = Callee->getName();
  return N == "logf" || N == "llvm.log.f32" || N == "_Z4logf" || N == "_Z3logf" || N == "__nv_logf";   // _Z3logf: OpenCL log(float)
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
      else if (CI->arg_size() == 2 && isNamed(CI, {"powf", "llvm.pow.f32", "__nv_powf", "_Z3powff"})) pow.push_back(CI);
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


// ---- OpenCL / Rodinia odds and ends -------------------------------------------------------------
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"

// ceil(float) -> -floor(-x) (expandFloorf then inlines the floor), mul24 -> mul, abs(int) and
// llvm.usub.sat -> selects. Run before expandFloorf / expandAbsI.
// log10(x) = log(x) * log10(e) and fmod(x, y) = x - trunc(x / y) * y (trunc via the integer
// conversion, exact while the quotient fits an i32), emitted before expandLogf runs.
static void expandLog10Fmod(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (CI->getCalledFunction()) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    StringRef N = CI->getCalledFunction()->getName();
    if (!CI->getType()->isFloatTy()) continue;
    IRBuilder<> B(CI); Value *R = nullptr;
    if ((N == "_Z5log10f" || N == "log10f" || N == "llvm.log10.f32") && CI->arg_size() == 1) {
      Function *LogF = cast<Function>(F.getParent()->getOrInsertFunction("logf", B.getFloatTy(), B.getFloatTy()).getCallee());
      R = B.CreateFMul(B.CreateCall(LogF, {CI->getArgOperand(0)}), ConstantFP::get(B.getFloatTy(), 0.43429448190325182765));
    } else if ((N == "_Z5exp10f" || N == "exp10f" || N == "llvm.exp10.f32") && CI->arg_size() == 1) {   // exp10(x) = exp(x ln 10) (s3d's ratx)
      R = B.CreateIntrinsic(Intrinsic::exp, {B.getFloatTy()}, {B.CreateFMul(CI->getArgOperand(0), ConstantFP::get(B.getFloatTy(), 2.30258509299404568402))});
    } else if ((N == "_Z4fmodff" || N == "fmodf") && CI->arg_size() == 2) {
      Value *X = CI->getArgOperand(0), *Y = CI->getArgOperand(1);
      Value *Q = B.CreateSIToFP(B.CreateFPToSI(B.CreateFDiv(X, Y), B.getInt32Ty()), B.getFloatTy());
      R = B.CreateFSub(X, B.CreateFMul(Q, Y));
    }
    if (R) { CI->replaceAllUsesWith(R); CI->eraseFromParent(); }
  }
}

void hwacha::expandOpenCLMisc(Function &F) {
  expandLog10Fmod(F);
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (CI->getCalledFunction()) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    StringRef N = CI->getCalledFunction()->getName();
    IRBuilder<> B(CI); Value *R = nullptr;
    if (N == "_Z4ceilf" && CI->arg_size() == 1 && CI->getType()->isFloatTy()) {
      Value *X = CI->getArgOperand(0);
      R = B.CreateFNeg(B.CreateUnaryIntrinsic(Intrinsic::floor, B.CreateFNeg(X)));
    } else if (N == "_Z5mul24ii" || N == "_Z5mul24jj") {
      R = B.CreateMul(CI->getArgOperand(0), CI->getArgOperand(1));
    } else if (N == "_Z3absi" || N == "_Z3absl") {
      Value *X = CI->getArgOperand(0);
      R = B.CreateSelect(B.CreateICmpSLT(X, ConstantInt::get(X->getType(), 0)), B.CreateNeg(X), X);
    } else if (N.starts_with("llvm.usub.sat.")) {
      Value *A = CI->getArgOperand(0), *Bv = CI->getArgOperand(1);
      R = B.CreateSelect(B.CreateICmpUGT(A, Bv), B.CreateSub(A, Bv), ConstantInt::get(A->getType(), 0));
    }
    if (R) { CI->replaceAllUsesWith(R); CI->eraseFromParent(); }
  }
}

// switch -> branches (the worker-thread codegen only knows conditional branches).
void hwacha::lowerSwitches(Function &F) {
  bool Any = false;
  for (BasicBlock &BB : F) if (isa<SwitchInst>(BB.getTerminator())) { Any = true; break; }
  if (!Any) return;
  PassBuilder PB; FunctionAnalysisManager FAM; PB.registerFunctionAnalyses(FAM);
  FunctionPassManager FPM; FPM.addPass(LowerSwitchPass()); FPM.run(F, FAM);
}

bool hwacha::isGlobalSizeCall(const Value *V) { return dim0Call(V, "_Z15get_global_sizej"); }
bool hwacha::isNumGroupsCall(const Value *V) { return dim0Call(V, "_Z14get_num_groupsj"); }

// 2-D NDRanges: a work-group of LS0 x LS1 work-items is flattened onto LS0*LS1 lanes and the NG0 x NG1
// groups onto NG0*NG1 sequential groups. (Barriers keep their meaning only while the whole group fits
// one stripmine, i.e. LS0*LS1 <= the kernel's maxvl; hwacha_vl_short reports when it does not.) The lane / group index the
// codegen provides (get_local_id(0) / get_group_id(0)) becomes the flattened one, and every
// work-item query is rewritten in terms of it and of the host-set globals hwacha_ls0, hwacha_ls1,
// hwacha_ng0 (the host also sets hwacha_group_size = LS0*LS1 and passes n = the flattened total):
//   lid0 = lane % LS0   lid1 = lane / LS0   grp0 = group % NG0   grp1 = group / NG0
//   gid_d = grp_d * LS_d + lid_d   local_size(d) = LS_d   global_size(0) = NG0*LS0
// Only kernels that query dimension 1 are touched.
void hwacha::flattenNDRange(Function &F) {
  auto dimOf = [](const CallInst *CI) -> int { auto *C = dyn_cast<ConstantInt>(CI->getArgOperand(0)); return C ? (int)C->getZExtValue() : -1; };
  SmallVector<CallInst *, 16> Q; bool Uses1 = false;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (Function *Callee = CI->getCalledFunction()) {
    StringRef N = Callee->getName();
    if (N == "_Z12get_local_idj" || N == "_Z12get_group_idj" || N == "_Z13get_global_idj" || N == "_Z14get_local_sizej" || N == "_Z15get_global_sizej" || N == "_Z14get_num_groupsj") {
      Q.push_back(CI); if (dimOf(CI) == 1) Uses1 = true;
      if (dimOf(CI) >= 2) { errs() << "hwacha-cc: " << F.getName() << ": 3-D NDRange queries are not supported\n"; }
    }
  }
  if (!Uses1) return;
  Module &M = *F.getParent(); LLVMContext &Ctx = F.getContext(); Type *I64 = Type::getInt64Ty(Ctx);
  auto glob = [&](StringRef Name) { GlobalVariable *G = M.getGlobalVariable(Name, true); if (!G) G = new GlobalVariable(M, I64, false, GlobalValue::ExternalLinkage, nullptr, Name); return G; };
  auto decl = [&](StringRef Name) { return M.getOrInsertFunction(Name, FunctionType::get(I64, {Type::getInt32Ty(Ctx)}, false)); };
  IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
  // The flattened work-item index is derived from the *global* id: a LS0*LS1 group larger than the
  // vector length Hwacha grants this kernel (maxvl depends on its register count) is executed as
  // several stripmines, and get_local_id(0) / get_group_id(0) then count within a stripmine, not
  // within the work-group. gid = group*LS0*LS1 + lane is unaffected.
  Value *Gid = B.CreateCall(decl("_Z13get_global_idj"), {B.getInt32(0)}, "gid");
  Value *LS0 = B.CreateLoad(I64, glob("hwacha_ls0"), "ls0"), *LS1 = B.CreateLoad(I64, glob("hwacha_ls1"), "ls1"), *NG0 = B.CreateLoad(I64, glob("hwacha_ng0"), "ng0");
  Value *GS = B.CreateMul(LS0, LS1, "gs");
  Value *Lane = B.CreateURem(Gid, GS, "lane"), *Grp = B.CreateUDiv(Gid, GS, "grp");
  Value *Lid0 = B.CreateURem(Lane, LS0, "lid0"), *Lid1 = B.CreateUDiv(Lane, LS0, "lid1");
  Value *Grp0 = B.CreateURem(Grp, NG0, "grp0"), *Grp1 = B.CreateUDiv(Grp, NG0, "grp1");
  Value *Gid0 = B.CreateAdd(B.CreateMul(Grp0, LS0), Lid0, "gid0"), *Gid1 = B.CreateAdd(B.CreateMul(Grp1, LS1), Lid1, "gid1");
  Value *NG1 = B.CreateLoad(I64, glob("hwacha_ng1"), "ng1");   // loaded here: the entry builder's insertion point may be a query call erased below
  for (CallInst *CI : Q) {
    int D = dimOf(CI); if (D < 0 || D > 1) continue;
    StringRef N = CI->getCalledFunction()->getName(); Value *R = nullptr;
    if (N == "_Z12get_local_idj") R = D ? Lid1 : Lid0;
    else if (N == "_Z12get_group_idj") R = D ? Grp1 : Grp0;
    else if (N == "_Z13get_global_idj") R = D ? Gid1 : Gid0;
    else if (N == "_Z14get_local_sizej") R = D ? LS1 : LS0;
    else if (N == "_Z15get_global_sizej") { IRBuilder<> Bi(CI); R = D ? Bi.CreateMul(NG1, LS1) : Bi.CreateMul(NG0, LS0); }
    else if (N == "_Z14get_num_groupsj") R = D ? NG1 : NG0;
    if (R) { CI->replaceAllUsesWith(R); CI->eraseFromParent(); }
  }
}


// Drop `nuw` from the kernel's integer arithmetic. clang infers it from a dominating guard (nw's
// `if (tx <= m)` makes `(m - tx) * 17 + 17` unsigned-no-wrap), and SCEV then turns the sext of such an
// expression into a zext and distributes it over its re-associated parts (`zext(17 - 17*tx) + 17*m`),
// which is wrong as soon as a part wraps on its own (tx = 2: zext(-17) + 34 = 2^32 + 17). Registers
// hold sign-extended values and sext distributes over nsw adds, so nothing is lost without nuw.
void hwacha::dropNUW(Function &F) {
  for (Instruction &I : instructions(F))
    if (auto *BO = dyn_cast<OverflowingBinaryOperator>(&I))
      if (BO->hasNoUnsignedWrap()) cast<Instruction>(BO)->setHasNoUnsignedWrap(false);
}


// llvm.memcpy / llvm.memset with a constant size (clang emits them for struct assignments such as
// lavaMD's `rA_shared[wtx] = d_rv_gpu[first_i + wtx]`) -> element-wise loads and stores. The codegen
// only knows loads and stores; a memcpy it does not recognise is silently dropped (the destination
// stays undef). Widths: 8-byte units while the size and both alignments allow, else 4, else 1.
void hwacha::expandMemIntrinsics(Function &F) {
  SmallVector<CallInst *, 8> Calls;
  for (Instruction &I : instructions(F)) if (auto *CI = dyn_cast<CallInst>(&I)) if (auto *II = dyn_cast<IntrinsicInst>(CI))
    if (II->getIntrinsicID() == Intrinsic::memcpy || II->getIntrinsicID() == Intrinsic::memmove || II->getIntrinsicID() == Intrinsic::memset)
      if (isa<ConstantInt>(II->getArgOperand(2))) Calls.push_back(CI);
  for (CallInst *CI : Calls) {
    auto *II = cast<IntrinsicInst>(CI);
    uint64_t N = cast<ConstantInt>(II->getArgOperand(2))->getZExtValue();
    IRBuilder<> B(CI); LLVMContext &C = F.getContext();
    Value *Dst = II->getArgOperand(0);
    bool IsSet = II->getIntrinsicID() == Intrinsic::memset;
    Value *Src = IsSet ? nullptr : II->getArgOperand(1);
    uint64_t DA = cast<MemIntrinsic>(II)->getDestAlign().valueOrOne().value();
    uint64_t SA = IsSet ? 8 : cast<MemTransferInst>(II)->getSourceAlign().valueOrOne().value();
    uint64_t W = 8; while (W > 1 && (N % W || DA % W || SA % W)) W /= 2;
    Type *ET = W == 8 ? Type::getInt64Ty(C) : W == 4 ? Type::getInt32Ty(C) : Type::getInt8Ty(C);
    Value *SetV = nullptr;
    if (IsSet) {   // splat the byte
      uint64_t Byte = cast<ConstantInt>(II->getArgOperand(1))->getZExtValue() & 0xff, V = 0;
      for (unsigned i = 0; i < W; i++) V |= Byte << (8 * i);
      SetV = ConstantInt::get(ET, V);
    }
    for (uint64_t Off = 0; Off < N; Off += W) {
      Value *D = B.CreateInBoundsGEP(B.getInt8Ty(), Dst, B.getInt64(Off));
      Value *V = IsSet ? SetV : B.CreateAlignedLoad(ET, B.CreateInBoundsGEP(B.getInt8Ty(), Src, B.getInt64(Off)), Align(W));
      B.CreateAlignedStore(V, D, Align(W));
    }
    CI->eraseFromParent();
  }
}


// Private (per-work-item) memory: an alloca in a kernel becomes a slice of a per-kernel buffer,
// indexed by the lane (get_local_id(0) = the element index within the stripmine; groups run one
// after another, so one buffer per kernel is enough). The buffer is a kernel-module global the
// control-thread module clones like the __local arrays.
void hwacha::expandAllocas(Function &F) {
  SmallVector<AllocaInst *, 8> Allocas;
  for (Instruction &I : instructions(F)) if (auto *A = dyn_cast<AllocaInst>(&I)) Allocas.push_back(A);
  if (Allocas.empty()) return;
  Module &M = *F.getParent(); LLVMContext &Ctx = F.getContext(); const DataLayout &DL = M.getDataLayout();
  const uint64_t Lanes = 512;   // >= any vector length Hwacha grants (maxvl <= 2048 / registers)
  FunctionCallee Lid = M.getOrInsertFunction("_Z12get_local_idj", FunctionType::get(Type::getInt64Ty(Ctx), {Type::getInt32Ty(Ctx)}, false));
  unsigned n = 0;
  for (AllocaInst *A : Allocas) {
    if (!A->isStaticAlloca()) { errs() << "hwacha-cc: " << F.getName() << ": dynamic alloca is not supported\n"; continue; }
    uint64_t S = (DL.getTypeAllocSize(A->getAllocatedType()) * cast<ConstantInt>(A->getArraySize())->getZExtValue() + 7) & ~7ULL;
    auto *AT = ArrayType::get(Type::getInt8Ty(Ctx), Lanes * S);
    auto *G = new GlobalVariable(M, AT, false, GlobalValue::InternalLinkage, ConstantAggregateZero::get(AT), (F.getName() + ".priv" + Twine(n++)).str());
    G->setAlignment(Align(8));
    IRBuilder<> B(A);
    Value *Lane = B.CreateCall(Lid, {B.getInt32(0)}, "lane");
    Value *P = B.CreateInBoundsGEP(B.getInt8Ty(), G, B.CreateMul(Lane, B.getInt64(S)), "priv");
    A->replaceAllUsesWith(P); A->eraseFromParent();
  }
}


// Inline every defined non-kernel function a kernel calls (clang keeps large helpers such as dwt2d's
// `transform` as calls; hwacha-cc only emits kernels, and a dropped call means dropped work), then
// SROA the kernels so the callees' small structs become registers before expandAllocas.
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Scalar/SROA.h"
#include "llvm/Transforms/Scalar/Scalarizer.h"
void hwacha::inlineCallees(Module &M) {
  bool Any = false;
  for (Function &F : M) {
    if (F.isDeclaration() || isKernel(F)) continue;
    F.removeFnAttr(Attribute::NoInline); F.removeFnAttr(Attribute::OptimizeNone);
    F.addFnAttr(Attribute::AlwaysInline); Any = true;
  }
  if (!Any) return;
  PassBuilder PB;
  LoopAnalysisManager LAM; FunctionAnalysisManager FAM; CGSCCAnalysisManager CGAM; ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM); PB.registerCGSCCAnalyses(CGAM); PB.registerFunctionAnalyses(FAM); PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ModulePassManager MPM; MPM.addPass(AlwaysInlinerPass()); MPM.run(M, MAM);
  for (Function &F : M) if (!F.isDeclaration() && isKernel(F)) {
    FunctionPassManager FPM; FPM.addPass(SROAPass(SROAOptions::ModifyCFG)); FPM.run(F, FAM);
  }
}

// OpenCL vector types (float2 / float4 / uint4: SHOC's fft, md, scan, sort) -> scalar operations,
// including vector loads and stores (LLVM's scalarizer with load-store scalarization). The codegen
// only knows scalar values.
void hwacha::scalarizeVectors(Module &M) {
  PassBuilder PB;
  LoopAnalysisManager LAM; FunctionAnalysisManager FAM; CGSCCAnalysisManager CGAM; ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM); PB.registerCGSCCAnalyses(CGAM); PB.registerFunctionAnalyses(FAM); PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  ScalarizerPassOptions Opts; Opts.ScalarizeLoadStore = true;
  for (Function &F : M) if (!F.isDeclaration()) { FunctionPassManager FPM; FPM.addPass(ScalarizerPass(Opts)); FPM.run(F, FAM); }
}

#include <cstdlib>
#include "CodeGen.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/TypeBasedAliasAnalysis.h"
#include "llvm/Analysis/MemoryLocation.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar/SimplifyCFG.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LCSSA.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ScalarEvolutionExpander.h"
#include "llvm/Support/FormatVariadic.h"
#include <functional>

using namespace llvm;
using namespace hwacha;

// ---------------------------------------------------------------- preprocessing

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

void hwacha::prepareKernel(Function &F) {
  annotateIdRange(F);
  PassBuilder PB;
  LoopAnalysisManager LAM; FunctionAnalysisManager FAM; CGSCCAnalysisManager CGAM; ModuleAnalysisManager MAM;
  PB.registerModuleAnalyses(MAM); PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM); PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
  FunctionPassManager FPM;
  FPM.addPass(InstCombinePass());
  FPM.addPass(EarlyCSEPass());
  FPM.addPass(SimplifyCFGPass());
  FPM.addPass(DCEPass());
  FPM.addPass(LoopSimplifyPass());
  FPM.addPass(LCSSAPass());
  FPM.run(F, FAM);
}

// ---------------------------------------------------------------- worker-thread codegen

namespace {

enum class RC { VV, VS, VP, VW };   // VW: vector register configured as 32-bit (numbered after the 64-bit ones)

struct Reg {
  RC Class; unsigned Idx;
  std::string str() const {
    return (Class == RC::VV ? "vv" : Class == RC::VS ? "vs" : Class == RC::VP ? "vp" : "vw") + std::to_string(Idx);
  }
};

class WTGen {
public:
  WTGen(Function &F, KernelAnalysis &KA, raw_ostream &Err, const CodeGenOptions &O)
      : F(F), KA(KA), Err(Err), Verbose(O.Verbose), Opts(O), DL(F.getParent()->getDataLayout()) {}

  // Results consumed by the control-thread generator.
  // Stride: constant lane stride, or 0 with StrideV holding the uniform runtime stride (bytes).
  // PerIter >= 0: the base advances inside control-thread loop #PerIter (re-sent before every body vf).
  struct Stream { Value *Base; int64_t Stride; unsigned VA; bool Local; unsigned StrideVA = 0; bool Unit = true; Value *StrideV = nullptr; int PerIter = -1;
                  SmallPtrSet<const BasicBlock *, 4> Blocks; };   // Blocks: where a per-iteration stream is accessed
  std::vector<Stream> Streams;
  // Uniform values the control thread must send with vmcs, in vs-register order (index = vs number).
  // Each is either a kernel Argument, a Constant, or an instruction depending only on those.
  std::vector<std::pair<unsigned, Value *>> VSInputs;
  // Numeric constants beyond what the shared register file holds live in a per-kernel constant pool
  // in memory (an i64 payload each, laid out by generateCT); PoolVS holds its address and a use
  // reloads the constant into a temporary vs (freed at the next position boundary).
  std::vector<Value *> PoolConsts;   // constants and (when vs registers run short) arguments, loaded on demand
  DenseMap<const Value *, unsigned> PoolIdx;
  unsigned PoolVS = 0;
  std::vector<unsigned> PendingTemps;
  // Spilling of uniform temporaries: when the shared register file is full, the live vs value with
  // the farthest last use is stored to a per-kernel spill area (SpillVS holds its address, SpillAddrVS
  // is the address scratch) and reloaded into a temporary at each later use.
  unsigned SpillVS = 0, SpillAddrVS = 0, NumSpillSlots = 0;
  size_t SpillReserveOff = 0;   // Text offset at which the spill base/address registers were reserved: a
                                // write-through store can only be inserted after it (the registers may
                                // have been temporaries before)
  void reserveSpillVS() { if (SpillVS) return; SpillVS = allocInputVS().Idx; SpillAddrVS = allocInputVS().Idx; SpillReserveOff = Text.size(); }
  DenseMap<const Value *, unsigned> Spilled;
  DenseSet<const Value *> CurOperands;   // operands of the instruction being emitted: never spilled
  bool SpillInBlock = false;             // a spill store was emitted in this block: no skip jump around it
  bool Pinning = false;                  // the input pre-allocation pass: rematerialized values are only registered
  bool spillOne(bool Diag = false);
  // Predicate spilling: an i1 value whose vp register is needed is parked as 0/1 words in a 32-bit
  // vector register and compared back into a temporary vp at each later use.
  DenseMap<const Value *, unsigned> SpilledVP;   // value -> vw register holding it
  DenseMap<std::pair<const BasicBlock *, const BasicBlock *>, unsigned> EdgeSpilled;   // edge -> vw register
  DenseMap<unsigned, unsigned> VWSpillRef;       // vw register -> number of spilled holders
  const BasicBlock *RestoringFor = nullptr;
  // Vector spilling: with a register cap (-vregs, or 2048/reqd_work_group_size so the group fits in
  // one stripmine) live vv/vw values are parked in a per-lane slot of <kernel>_vspill
  // (lane * stride + 8 * slot, indexed through a lane-offset vector register).
  unsigned VPLimit = 16;   // -vpregs: predicate registers usable (testing the predicate spiller)
  unsigned VCap = 256, VSpillVS = 0, VStrideVS = 0, VSpillAddrVS = 0, NumVSlots = 0;
  Reg LaneOffVV{RC::VV, 0}; bool HaveLaneOff = false;
  DenseMap<const Value *, unsigned> SpilledVec;   // value -> slot
  std::vector<Reg> PendingVec;
  bool spillOneVec(RC C, bool Diag = false);
  unsigned freeVec() const {   // registers available without raising the vv+vw count past the cap
    unsigned N = VCap > NumVV + NumVW ? VCap - NumVV - NumVW : 0;
    for (unsigned i = 0; i < NumVV; i++) if (!VVUsed[i]) N++;
    for (unsigned i = 0; i < NumVW; i++) if (!VWUsed[i]) N++;
    return N;
  }
  Reg poolLoad(Value *V);
  void vecSlotAddr(unsigned Slot);
  std::string vecSlotAddrText(unsigned Slot);
  void phiStore(unsigned Mask, PHINode *Phi, Value *Src);   // @Mask slot(Phi) = Src, for a spilled header phi
  std::string DbgWhere; std::string VPAllocSite[16];
  const Instruction *CurInst = nullptr;   // being emitted: its destination is never a spill victim
  // A no-op cast of a value that is not register-resident (pooled constant / argument, spilled
  // value, rematerialized address) cannot share a register: regOfValue re-derives the source at
  // every use instead. (dwt2d: `sext sx` captured the pool-load temporary of sx, which was dead
  // and reused by the time the sext was read.)
  DenseMap<const Value *, Value *> Forward;
  void aliasTo(Instruction &I, Value *Src) {
    Reg R = regOfValue(Src);
    auto It = RegOf.find(Src);
    if (It != RegOf.end() && It->second.Class == R.Class && It->second.Idx == R.Idx) RegOf[&I] = R;
    else Forward[&I] = Src;
    Alias.insert(&I);
    if (auto *SI = dyn_cast<Instruction>(Src)) LastUse[SI] = std::max(LastUse[SI], LastUse[&I]);
  }
  unsigned TraceVS = 0;                   // HWCC_TRACE: written with the position before every instruction
  // Predicate spills are hoisted to the top of the current block (before its skip jump) so the
  // block can still be skipped when no lane is active: the victim must have been defined before
  // the block, and the vw register receiving it must have been free at the block top too.
  bool InBlock = false; const BasicBlock *CurBB = nullptr;
  std::vector<bool> VWUsedAtTop, VVUsedAtTop;
  unsigned OneVS = 0;                     // input register holding 1 (the hoisted spill code cannot use a temporary)
  std::vector<unsigned> PendingVP;
  bool spillOneVP(bool Diag = false);
  void restoreEdgesTo(BasicBlock *BB);
  unsigned VSOffset = 0;      // vs register that receives the stripmine element offset, 0 if unused
  unsigned VSLocalSize = 0;   // vs register that receives vl (get_local_size), 0 if unused
  unsigned VSGlobalSize = 0;  // vs register that receives n (get_global_size), 0 if unused
  unsigned VSGroupId = 0;     // vs register that receives the stripmine iteration index, 0 if unused
  bool UsesBarrier = false;
  uint64_t GroupSize = 0;     // reqd_work_group_size(X,1,1) if present
  // ---- control-thread loops. A depth-1 single-block loop with a uniform exit condition that is
  // reached under uniform control flow only is run by the control thread: it evaluates the loop's
  // uniform slice in scalar code and issues the body as one vf block per iteration. Vector state
  // persists across vf blocks, uniform values the body needs arrive by vmcs, stream bases that
  // advance with the loop by vmca. No masks, no consensual jumps, unit-stride streams inside loops.
  struct CTLoop {
    Loop *L = nullptr;
    std::string AfterLabel;
    std::vector<std::pair<unsigned, Value *>> IterInputs;   // vs <- uniform value, sent right after the control thread computes it
    unsigned Mask = 0;                                        // predicate the region runs under (lanes that entered the loop)
    bool NeedFence = false;                                   // a vector store issued before the region may alias its scalar loads
    DenseMap<const BasicBlock *, std::string> BlockSeg;       // one vf block per kernel block
    DenseMap<std::pair<const BasicBlock *, const BasicBlock *>, std::string> EdgeSeg;   // phi moves that could not be merged into the source block
    DenseMap<const BasicBlock *, std::vector<std::pair<int, std::string>>> BlockTail;   // (reduction, next segment) pairs splitting a block
  };
  std::vector<CTLoop> CTLoops;
  DenseMap<const Loop *, int> CTLoopOf;
  struct Segment { std::string Label; int CT; int Reduce = -1; };   // vf blocks in issue order; CT >= 0: region issued before it; Reduce >= 0: reduction done before it
  // ---- cross-lane reductions (work_group_reduce_*). The worker ISA has no reduction, so the lanes
  // store their values (identity where masked off) to a scratch buffer, the vf block ends, and the
  // control thread reduces the vl values in scalar code after a fence and sends the result by vmcs.
  // Reduction tree in the vector unit: the control thread issues log2(vl) steps of TreeLabel with
  // shrinking vl (vsetvl) and the upper half's address in TreeVA; the segment after the reduction
  // starts with a vfence and a scalar load of slot 0 into VS. No fence on the control thread.
  struct Reduction { unsigned VS; Value *V; std::string Op; Type *Ty; std::string TreeLabel; };
  std::vector<Reduction> Reductions;
  unsigned ScratchVA = 0;     // va register holding the scratch buffer (set when a reduction exists)
  unsigned TreeVA = 0;        // va register for the upper half during the tree steps
  unsigned ScratchVS = 0;     // vs register holding the scratch address (scalar load of the result)
  unsigned PredLoadVS = 0;    // vs temp for uniform i1 loads (vlsb cannot target a vector register)
  bool BlockSplit = false;    // the current block was split by a reduction (no skip jump around it)
  std::vector<Segment> Segments;
  unsigned NumVV = 0, NumVP = 1, NumVS = 0, NumVW = 0;
  unsigned NumInsts = 0, NumJumps = 0, NumMasks = 0, NumMoves = 0;
  std::string Text;

  bool run();
  bool isUniformValue(const Value *V) const { return isUniform(V); }

private:
  Function &F; KernelAnalysis &KA; raw_ostream &Err; bool Verbose;
public:
  const CodeGenOptions &Opts;
private:
  const DataLayout &DL;
  raw_string_ostream Out{Text};
  DenseMap<const Value *, Reg> RegOf;
  DenseMap<const Instruction *, const MemAccess *> AccessOf;
  DenseMap<const Instruction *, AddrKind> KindOf;          // effective kind (streams may be demoted to indexed)
  DenseMap<const Instruction *, unsigned> StreamOfInst;   // access -> stream index
  DenseMap<const Instruction *, Value *> GatherIndex;     // access -> divergent byte-offset value
  DenseMap<const Instruction *, Value *> GatherBase;      // access -> uniform base value (vs)
  DenseSet<const Value *> Needed;                          // instructions that must be emitted
  DenseMap<const Value *, unsigned> LastUse;
  DenseSet<const Value *> Alias;                           // no-op casts sharing their source's register
  DenseSet<const Value *> Transferred;                     // operands whose register was taken over by a destination
  DenseSet<const Value *> UnsignedLoads;                   // narrow loads emitted as vl*u (all users are zext)
  DenseMap<const SelectInst *, Reg> SelectReg;             // select whose arm was computed straight into its register
  DenseMap<const Value *, Reg> PreferReg;                  // loop-carried value -> its header phi's register
  DenseMap<const Value *, const Value *> PreferPhi;        // loop-carried value -> its header phi
  DenseMap<const Value *, unsigned> PhiRealLast;           // header phi -> last read inside the loop body
  std::vector<int> VPRef;                                  // refcount per vp register (masks)
  std::string CurPred;                                     // predicate applied to vector/memory ops of the current block
  std::string ArmPred;                                     // set by dest(): predicate for an instruction computed into a select's register

  // ---- control flow
  struct Item { enum Kind { Block, LoopBegin, LoopEnd } K; BasicBlock *BB; Loop *L; };
  std::vector<Item> Order;
  std::unique_ptr<DominatorTree> DT; std::unique_ptr<LoopInfo> LI;
  DenseMap<const BasicBlock *, unsigned> BlockPos;         // position of the block's phis
  DenseMap<const Loop *, std::pair<unsigned, unsigned>> LoopRange;
  DenseMap<const BasicBlock *, unsigned> BlockPred;        // vp index of the block predicate (0 = vp0)
  DenseMap<std::pair<const BasicBlock *, const BasicBlock *>, unsigned> EdgeMask;
  struct LoopState { Loop *L; unsigned Active; std::string Label; DenseMap<const BasicBlock *, unsigned> ExitTotal; };
  std::vector<LoopState> LoopStack;
  DenseMap<const PHINode *, Reg> Captured;                 // LCSSA exit phis resolved at exit edges
  unsigned LinkVS = 0;                                     // vs scratch for vcjal link writes
  unsigned ZeroVP = 0;                                     // a predicate register kept all-zero (i1 false)
  unsigned LabelCounter = 0;

  void linearize(Loop *L, const SmallPtrSetImpl<BasicBlock *> &Members, BasicBlock *Entry);
  void computePositions();
  unsigned allocVP() { Reg R = alloc(RC::VP); if (VPRef.size() <= R.Idx) VPRef.resize(R.Idx + 1, 0); VPRef[R.Idx] = 1; return R.Idx; }
  unsigned shareVP(unsigned Idx) { if (Idx) VPRef[Idx]++; return Idx; }
  void freeVP(unsigned Idx) { if (!Idx) return; if (--VPRef[Idx] == 0) VPUsed[Idx] = false; }
  static std::string vp(unsigned Idx) { return "vp" + std::to_string(Idx); }
  unsigned andMask(unsigned A, unsigned B, bool NegB);      // new vp = A & (Neg ? !B : B)
  unsigned orInto(unsigned Acc, unsigned B);               // Acc |= B (Acc may be 0 -> allocate)
  void movePred(unsigned Pred, Reg Dst, Value *Src);       // @Pred Dst = Src (any class)
  bool emitBlock(BasicBlock *BB, unsigned &Pos);
  bool beginLoop(Loop *L, unsigned &Pos);
  bool endLoop(Loop *L, unsigned &Pos);
  bool handleEdge(BasicBlock *From, BasicBlock *To, unsigned Mask, unsigned Pos);
  DenseMap<const Value *, unsigned> DefPos;                 // position of the defining instruction / phi
  DenseMap<const Value *, SmallVector<unsigned, 8>> UsePos; // every position that reads the value
  unsigned CurPos = 0;                                      // position being emitted
  // Spilling inside a loop: a value may leave its register only if the loop re-defines it every
  // iteration (defined inside), or if nothing in the loop has read the register yet (every use
  // from here on reloads). NeedInside: the spill emits code at this point (predicate spills), so
  // the register must be re-defined before the point is reached again.
  bool loopSafe(const Value *V, bool NeedInside) const {
    unsigned D = DefPos.lookup(V);
    for (auto &KV : LoopRange) {
      unsigned B = KV.second.first, E = KV.second.second;
      if (!(B < CurPos && CurPos <= E)) continue;
      if (D > B) continue;                                    // defined inside (a header phi has D == B)
      if (NeedInside) return false;
      auto It = UsePos.find(V);
      if (It != UsePos.end()) for (unsigned U : It->second) if (U >= B && U < CurPos) return false;   // a use at CurPos reloads
    }
    return true;
  }
  bool edgeLoopSafe(const BasicBlock *From) const {
    unsigned D = BlockPos.lookup(From);
    for (auto &KV : LoopRange) { unsigned B = KV.second.first, E = KV.second.second; if (B < CurPos && CurPos <= E && D <= B) return false; }
    return true;
  }
  // Write-through spilling: the store into the slot is inserted right after the value's definition
  // (recorded as a Text offset), so the slot is valid wherever the register was.
  DenseMap<const Value *, size_t> DefEnd;
  size_t BodyStart = 0;   // Text offset where the current block's body begins (the skip jump goes there)
  void insertText(size_t At, const std::string &S) {
    Text.insert(At, S);
    for (auto &KV : DefEnd) if (KV.second > At) KV.second += S.size();
    if (BodyStart >= At) BodyStart += S.size();
  }
  static std::string fmt(StringRef Pred, StringRef Op, ArrayRef<std::string> Ops) {
    std::string S = "    " + (Pred.empty() ? std::string() : ("@" + Pred + " ").str()) + Op.str();
    for (size_t i = 0; i < Ops.size(); i++) S += (i ? ", " : " ") + Ops[i];
    return S + "\n";
  }
  unsigned LastReleasePos = 0;   // RegOf keeps dead entries; a value with LastUse <= this is dead
  bool liveInst(const Value *V) const { return isa<Instruction>(V) && LastUse.lookup(V) > LastReleasePos; }
  void releaseAt(unsigned Pos) { LastReleasePos = Pos; for (auto &KV : LastUse) if (KV.second == Pos) release(KV.first); for (unsigned T : PendingTemps) VSUsed[T] = false; PendingTemps.clear(); for (unsigned P : PendingVP) { VPUsed[P] = false; if (VPRef.size() > P) VPRef[P] = 0; } PendingVP.clear(); for (Reg &R : PendingVec) (R.Class == RC::VV ? VVUsed : VWUsed)[R.Idx] = false; PendingVec.clear(); }
  std::vector<bool> VVUsed, VSUsed, VPUsed, VWUsed;
  std::vector<bool> VSEverUsed;   // vs registers a temporary has ever occupied (unsafe for a late input)
  DenseSet<const Instruction *> SpecialCalls;             // id / local-size / group-id calls: registers pre-assigned

  bool fail(const Twine &Msg, const Value *V = nullptr) {
    Err << "hwacha-cc: " << F.getName() << ": " << Msg;
    if (V) { Err << ": "; V->print(Err); }
    Err << "\n"; return false;
  }
  bool materializeAddresses();
  void selectCTLoops();
  bool ctCloneable(Value *V, DenseSet<Value *> &Seen, unsigned Depth = 0);
  bool ctCloneableImpl(Value *V, DenseSet<Value *> &Seen, unsigned Depth);
  bool ctLoadOK(LoadInst *LD);
  bool reachUniform(BasicBlock *BB, DenseMap<BasicBlock *, int> &Memo, DenseSet<Value *> &Conds);
  bool emitCTRegion(size_t &Idx, unsigned &Pos);
  bool emitCTBlock(BasicBlock *BB, unsigned &Pos, CTLoop &C);
  Reg phiReg(PHINode *Phi) { auto It = RegOf.find(Phi); if (It != RegOf.end()) return It->second; Reg R = alloc(classOf(Phi)); RegOf[Phi] = R; return R; }
  void registerIterInput(Instruction *I, Loop *Region, CTLoop &C) {
    // a uniform value the control thread computes: only vector code inside the region (or anything after it) needs it in a vs register
    bool Used = ImplicitUsers.count(I);
    for (User *U : I->users()) { auto *UI = dyn_cast<Instruction>(U); if (UI && UI->isTerminator()) continue; if (!UI || !Region->contains(UI) || !ctSkip(UI)) Used = true; }
    if (!Used) return;
    Reg R = alloc(RC::VS); RegOf[I] = R; C.IterInputs.push_back({R.Idx, I}); CTPinned.insert(I);
  }
  Loop *outermost(const BasicBlock *BB) const { Loop *L = LI ? LI->getLoopFor(BB) : nullptr; while (L && L->getParentLoop()) L = L->getParentLoop(); return L; }
  bool inCTBody(const Instruction *I) const { Loop *L = outermost(I->getParent()); return L && CTLoopOf.count(L); }
  bool ctSkip(const Instruction *I, unsigned Depth = 0) const {   // computed by the control thread instead of the vf body
    if (!inCTBody(I) || isa<StoreInst>(I) || isa<CallBase>(I) || I->isTerminator() || I->getType()->isVoidTy()) return false;
    if (I->getType()->isIntegerTy(1)) {
      // a uniform predicate stays on the control thread only if nothing in the vf block reads it
      if (!isUniform(I) || Depth > 8) return false;
      for (const User *U : I->users()) {
        auto *UI = dyn_cast<Instruction>(U);
        if (!UI || UI->isTerminator()) continue;
        if (!ctSkip(UI, Depth + 1)) return false;
      }
      return true;
    }
    return classOf(I) == RC::VS;
  }
  DenseSet<const Value *> CTPinned;                        // iteration inputs: registers held until the loop ends
  DenseSet<const Value *> ImplicitUsers;                   // values referenced by GatherBase/GatherIndex
  std::unique_ptr<BasicAAResult> BAA; std::unique_ptr<TypeBasedAAResult> TBAA; std::unique_ptr<AAResults> AA;
  Loop *CandRegion = nullptr; bool CandFence = false;      // region whose uniform slice is being checked
  void computeNeeded();
  Reg alloc(RC C);
  Reg allocInputVS();
  unsigned freeVS() { if (VSUsed.size() < 64) VSUsed.resize(64, false); unsigned n = 0; for (unsigned i = 1; i < 64; i++) if (!VSUsed[i]) n++; return n; }
  void release(const Value *V);
  Reg regOfValue(Value *V);             // allocate/lookup register for an operand
  bool emitInst(Instruction &I, unsigned Pos);
  bool emitInstImpl(Instruction &I, unsigned Pos);
  void emit(StringRef Pred, StringRef Op, ArrayRef<std::string> Ops) {
    NumInsts++;
    if (Op == "vcjal") NumJumps++;
    else if (Op == "vpop" || Op == "vpclear" || Op == "vpset") NumMasks++;
    else if (Op == "vadd" && Ops.size() == 3 && Ops[2] == "vs0") NumMoves++;
    Out << fmt(Pred, Op, Ops);
  }
  bool isUniform(const Value *V) const { return isa<Constant>(V) || KA.isUniform(V); }
  bool isNarrow(const Value *V) const {
    Type *T = V->getType();
    return (T->isIntegerTy() && T->getIntegerBitWidth() <= 32) || T->isFloatTy() || T->isHalfTy();
  }
  // Any computation that needs the FPU must live in a vector register even when it is uniform:
  // scalar (vs-destination) floating-point ops in a vf block go to the shared Rocket FPU, which
  // never services Hwacha on the Chipyard RTL (the scalar unit hangs). Loads, arguments and
  // constants of floating-point type may stay in vs as broadcast operands.
  bool needsFPU(const Value *V) const {
    auto *I = dyn_cast<Instruction>(V);
    if (!I || isa<LoadInst>(I) || isWorkGroupReduce(I)) return false;   // a reduction result arrives by vmcs
    if (I->getType()->isFloatingPointTy()) return true;                 // FP result (arith, phi, select, cvt)
    for (const Value *Op : I->operands()) if (Op->getType()->isFloatingPointTy()) return true;   // fptosi etc.
    return false;
  }
  // Values assembled from predicated moves (select, join-block phi, integer min/max) cannot live
  // in a vs register: scalar ops ignore predicates, so both moves would execute and the last one
  // would win. Loop-header phis are fine (exactly one move per iteration).
  bool mergedByPredicates(const Value *V) const {
    if (isa<SelectInst>(V)) return true;
    if (auto *C = dyn_cast<CastInst>(V)) if (C->getSrcTy()->isIntegerTy(1)) return true;   // predicate -> int
    if (auto *Phi = dyn_cast<PHINode>(V)) {
      if (!LI) return true;
      Loop *L = LI->getLoopFor(Phi->getParent());
      return !(L && L->getHeader() == Phi->getParent());
    }
    if (auto *II = dyn_cast<IntrinsicInst>(V)) {
      auto Id = II->getIntrinsicID();
      return Id == Intrinsic::smin || Id == Intrinsic::smax || Id == Intrinsic::umin || Id == Intrinsic::umax;
    }
    return false;
  }
  RC vecClass(const Value *V) const { return (isNarrow(V) && !Opts.NoV32) ? RC::VW : RC::VV; }
  RC baseClass(const Value *V) const {
    if (V->getType()->isIntegerTy(1)) return RC::VP;
    if (isUniform(V) && !(needsFPU(V) && !Opts.ScalarFP) && !mergedByPredicates(V)) return RC::VS;
    return vecClass(V);
  }
  DenseMap<const Value *, RC> ClassMap;
  // Vector-ness propagates: a scalar (vs) op cannot read a vector register, so every consumer
  // of a vector-class value becomes vector-class too (fixpoint over the whole function).
  void computeClasses() {
    for (Instruction &I : instructions(F)) ClassMap[&I] = baseClass(&I);
    bool Changed = true;
    while (Changed) {
      Changed = false;
      for (Instruction &I : instructions(F)) {
        if (ClassMap[&I] != RC::VS || isWorkGroupReduce(&I)) continue;   // a reduction legitimately reads vectors
        for (Value *Op : I.operands()) {
          auto It = ClassMap.find(Op);
          if (It != ClassMap.end() && isVec(It->second)) { ClassMap[&I] = vecClass(&I); Changed = true; break; }
        }
      }
    }
  }
  RC classOf(const Value *V) const {
    auto It = ClassMap.find(V);
    return It != ClassMap.end() ? It->second : baseClass(V);
  }
  bool isVec(RC C) const { return C == RC::VV || C == RC::VW; }
  std::string memSuffix(Type *T, bool &IsFloat) {
    IsFloat = T->isFloatingPointTy();
    switch (DL.getTypeStoreSize(T)) { case 1: return "b"; case 2: return "h"; case 4: return "w"; default: return "d"; }
  }
  std::string fpSuffix(Type *T) { return T->isFloatTy() ? ".s" : T->isDoubleTy() ? ".d" : ".h"; }
  bool is32(Type *T) { return T->isIntegerTy(32); }
};

Reg WTGen::alloc(RC C) {
  std::vector<bool> &Used = C == RC::VV ? VVUsed : C == RC::VS ? VSUsed : C == RC::VP ? VPUsed : VWUsed;
  unsigned Start = (C == RC::VV || C == RC::VW) ? 0 : 1, Limit = (C == RC::VV || C == RC::VW) ? 256 : C == RC::VS ? 64 : VPLimit;
  if (Used.size() < 16) Used.resize(std::max(16u, Limit), false);
  if (C == RC::VS) {
    if (!SpillVS && freeVS() < 6) reserveSpillVS();   // reserve while there still are two
    if (freeVS() == 0 && !spillOne(true)) report_fatal_error(Twine("out of Hwacha registers of class vs (nothing to spill) in ") + F.getName());
  }
  if ((C == RC::VV || C == RC::VW) && VCap < 256) {
    unsigned &Num = C == RC::VV ? NumVV : NumVW;
    for (;;) {
      unsigned i = Start; while (i < Limit && Used[i]) i++;
      if (i < Num || NumVV + NumVW < VCap) break;   // reuses a register below the high-water mark, or room to grow
      if (spillOneVec(C)) continue;
      // a narrow value may live in a free 64-bit register (every vw is renamed to a vv in the end anyway)
      if (C == RC::VW) {
        for (unsigned j = 0; j < NumVV; j++) if (!VVUsed[j]) { VVUsed[j] = true; return Reg{RC::VV, j}; }
        if (spillOneVec(RC::VV)) { for (unsigned j = 0; j < NumVV; j++) if (!VVUsed[j]) { VVUsed[j] = true; return Reg{RC::VV, j}; } }
      }
      if (!spillOneVec(C, true)) report_fatal_error(Twine("out of Hwacha vector registers under the register cap (nothing to spill) in ") + F.getName());
    }
  }
  if (C == RC::VP) {
    bool Free = false; for (unsigned i = 1; i < VPLimit; i++) if (!Used[i]) { Free = true; break; }
    if (!Free && !spillOneVP(true)) report_fatal_error(Twine("out of Hwacha registers of class vp (nothing to spill) in ") + F.getName());
    if (getenv("HWCC_DEBUG_VS")) for (unsigned i = 1; i < 16; i++) if (!Used[i]) { VPAllocSite[i] = DbgWhere; break; }
  }
  for (unsigned i = Start; i < Limit; i++)
    if (!Used[i]) {
      Used[i] = true;
      if (C == RC::VV) NumVV = std::max(NumVV, i + 1);
      if (C == RC::VP) NumVP = std::max(NumVP, i + 1);
      if (C == RC::VS) { NumVS = std::max(NumVS, i + 1); if (VSEverUsed.size() < 64) VSEverUsed.resize(64, false); VSEverUsed[i] = true; }
      if (C == RC::VW) NumVW = std::max(NumVW, i + 1);
      return Reg{C, i};
    }
  if (getenv("HWCC_DEBUG_VS") && C == RC::VP) { unsigned used = 0; for (unsigned i = 1; i < 16; i++) if (VPUsed[i]) used++; errs() << "vp exhausted: used " << used << ", edge masks " << EdgeMask.size() << ", loop depth " << LoopStack.size() << ", captured " << Captured.size() << "\n"; for (auto &KV : EdgeMask) errs() << "  edge " << KV.first.first->getName() << " -> " << KV.first.second->getName() << " vp" << KV.second << "\n"; }
  if (getenv("HWCC_DEBUG_VS") && C == RC::VS) { unsigned live = 0, inputs = VSInputs.size(); for (unsigned i = 1; i < 64; i++) if (VSUsed[i]) live++; errs() << "vs temps exhausted: used " << live << " of 63, inputs " << inputs << ", pool " << PoolConsts.size() << ", pending temps " << PendingTemps.size() << "\n"; }
  report_fatal_error(Twine("out of Hwacha registers of class ") + (C == RC::VV ? "vv" : C == RC::VS ? "vs" : C == RC::VP ? "vp" : "vw") + " in " + F.getName());
}

void WTGen::release(const Value *V) {
  if (auto SI = SpilledVP.find(V); SI != SpilledVP.end()) { unsigned T = SI->second; SpilledVP.erase(SI); if (--VWSpillRef[T] == 0) VWUsed[T] = false; return; }
  if (SpilledVec.erase(V)) return;
  auto It = RegOf.find(V);
  if (It == RegOf.end()) return;
  if (isa<Argument>(V) || isa<Constant>(V)) return;  // inputs live for the whole block
  if (Alias.count(V)) return;                        // the register belongs to the cast's source
  if (CTPinned.count(V)) return;                     // sent by the control thread before every body vf
  if (Transferred.count(V)) return;                  // the register now belongs to another value
  if (auto *I = dyn_cast<Instruction>(V)) if (SpecialCalls.count(I)) return;
  Reg R = It->second;
  (R.Class == RC::VV ? VVUsed : R.Class == RC::VS ? VSUsed : R.Class == RC::VP ? VPUsed : VWUsed)[R.Idx] = false;
}

// Inputs (arguments, constants) live for the whole block and may be discovered lazily while
// emitting (shift amounts, GEP scales ...). They are allocated from the top of the vs file so
// they can never collide with a temporary that was handed out earlier from the bottom.
Reg WTGen::allocInputVS() {
  if (VSUsed.size() < 64) VSUsed.resize(64, false);
  if (VSEverUsed.size() < 64) VSEverUsed.resize(64, false);
  // An input is written once by the control thread before the first vf, so its register must not
  // have been written by a temporary earlier in the instruction stream: prefer never-used registers
  // (the pool / spill bases are allocated lazily, after temporaries have come and gone).
  for (int i = 63; i >= 1; i--)
    if (!VSUsed[i] && !VSEverUsed[i]) { VSUsed[i] = VSEverUsed[i] = true; NumVS = std::max(NumVS, (unsigned)i + 1); return Reg{RC::VS, (unsigned)i}; }
  if (getenv("HWCC_DEBUG_VS")) errs() << "hwacha-cc: " << F.getName() << ": no never-used vs register left for a late input\n";
  if (getenv("HWCC_DEBUG_VS")) { unsigned ci = 0, cf = 0, ce = 0, cg = 0, ca = 0, co = 0; for (auto &KV : VSInputs) { Value *V = KV.second; if (isa<ConstantInt>(V)) ci++; else if (isa<ConstantFP>(V)) cf++; else if (isa<ConstantExpr>(V)) ce++; else if (isa<GlobalValue>(V)) cg++; else if (isa<Argument>(V)) ca++; else co++; } errs() << "vs inputs: int " << ci << " fp " << cf << " cexpr " << ce << " global " << cg << " arg " << ca << " other " << co << "; pool " << PoolConsts.size() << "\n"; for (auto &KV : VSInputs) if (isa<ConstantExpr>(KV.second)) errs() << "  cexpr: " << *KV.second << "\n"; }
  report_fatal_error(Twine("out of Hwacha shared registers in ") + F.getName());
}

// Free one vs register by spilling the live uniform temporary with the farthest last use. Several
// values may share a register (no-op casts, freeze, an operand whose register a destination took
// over): the register is spilled as a whole and every value mapped to it reloads from the slot.
bool WTGen::spillOne(bool Diag) {
  DenseMap<unsigned, unsigned> Far; DenseSet<unsigned> Excluded;
  for (auto &KV : PreferReg) if (KV.second.Class == RC::VS) Excluded.insert(KV.second.Idx);
  for (auto &KV : SelectReg) if (KV.second.Class == RC::VS && (!RegOf.count(KV.first) || liveInst(KV.first))) Excluded.insert(KV.second.Idx);
  for (unsigned T : PendingTemps) Excluded.insert(T);
  for (auto &KV : RegOf) {
    const Value *V = KV.first; Reg R = KV.second;
    if (R.Class != RC::VS || R.Idx == 0) continue;
    if (isa<Instruction>(V) && !liveInst(V)) continue;   // dead entry, register long since reused
    if (!isa<Instruction>(V) || isa<PHINode>(V) || CTPinned.count(V) || PreferReg.count(V) || CurOperands.count(V) || V == CurInst ||
        SpecialCalls.count(cast<Instruction>(V)) || Spilled.count(V) || !DefEnd.count(V) || DefEnd.lookup(V) < SpillReserveOff || !loopSafe(V, false)) { Excluded.insert(R.Idx); continue; }
    Far[R.Idx] = std::max(Far[R.Idx], LastUse.lookup(V));
  }
  unsigned Victim = 0, VFar = 0;
  for (auto &KV : Far) if (!Excluded.count(KV.first) && (!Victim || KV.second > VFar)) { Victim = KV.first; VFar = KV.second; }
  if (!Victim) {
    if (Diag && getenv("HWCC_DEBUG_VS")) {
      DenseMap<unsigned, std::string> Role;
      for (auto &KV : VSInputs) Role[KV.first] += (isa<Constant>(KV.second) ? "const " : isa<Argument>(KV.second) ? "arg " : "input ");
      for (auto &KV : RegOf) if (KV.second.Class == RC::VS && KV.second.Idx && (!isa<Instruction>(KV.first) || liveInst(KV.first))) {
        std::string r = isa<Instruction>(KV.first) ? (isa<PHINode>(KV.first) ? "phi" : "val") : isa<Constant>(KV.first) ? "constval" : "argval";
        if (CurOperands.count(KV.first)) r += "(curop)"; if (PreferReg.count(KV.first)) r += "(prefer)"; if (CTPinned.count(KV.first)) r += "(ctpinned)"; if (KV.first == CurInst) r += "(cur)";
        if (auto *I = dyn_cast<Instruction>(KV.first)) if (SpecialCalls.count(I)) r += "(special)";
        Role[KV.second.Idx] += r + " "; }
      for (unsigned T : PendingTemps) Role[T] += "pendingReload ";
      if (SpillVS) Role[SpillVS] += "spillBase "; if (SpillAddrVS) Role[SpillAddrVS] += "spillAddr "; if (PoolVS) Role[PoolVS] += "poolBase "; if (LinkVS) Role[LinkVS] += "link ";
      for (auto &KV : PreferReg) if (KV.second.Class == RC::VS) Role[KV.second.Idx] += "preferTarget ";
      errs() << "vs exhausted in " << F.getName() << " at " << DbgWhere << ":\n";
      for (unsigned i = 1; i < 64; i++) if (VSUsed[i]) errs() << "  vs" << i << ": " << (Role.count(i) ? Role[i] : "(temporary)") << "\n";
    }
    return false;
  }
  unsigned Slot = NumSpillSlots++;
  uint64_t Off = 8 * Slot; std::string Base = "vs" + std::to_string(SpillVS), A = "vs" + std::to_string(SpillAddrVS), Store;
  for (; Off > 2040; Off -= 2040) { Store += fmt("", "vaddi", {A, Base, "2040"}); Base = A; }
  Store += fmt("", "vaddi", {A, Base, std::to_string(Off)});
  Store += fmt("", "vssd", {A, "vs" + std::to_string(Victim)});   // address first
  SmallVector<const Value *, 4> Holders;
  for (auto &KV : RegOf) if (KV.second.Class == RC::VS && KV.second.Idx == Victim) Holders.push_back(KV.first);
  for (const Value *W : Holders) { RegOf.erase(W); if (liveInst(W)) { Spilled[W] = Slot; insertText(DefEnd[W], Store); NumInsts += 2; } }
  VSUsed[Victim] = false;
  return true;
}

// Free one predicate register: the vp holding the live i1 value(s) with the farthest last use is
// parked in a vw register as 0/1 words (@vp vaddw T, vs0, 1 over a cleared T); every value mapped to
// that vp (dead operands whose register a destination took over stay mapped) reloads from it.
bool WTGen::spillOneVP(bool Diag) {
  DenseSet<unsigned> Excluded;
  for (auto &KV : PreferReg) if (KV.second.Class == RC::VP) Excluded.insert(KV.second.Idx);
  for (unsigned P : PendingVP) Excluded.insert(P);
  if (ZeroVP) Excluded.insert(ZeroVP);
  for (auto &LS : LoopStack) { Excluded.insert(LS.Active); for (auto &E : LS.ExitTotal) Excluded.insert(E.second); }
  if (!CurPred.empty()) Excluded.insert(std::atoi(CurPred.c_str() + 2));
  DenseMap<unsigned, unsigned> Far;
  for (auto &KV : EdgeMask) {
    if (KV.first.second == RestoringFor || !edgeLoopSafe(KV.first.first) || (InBlock && KV.first.first == CurBB)) { Excluded.insert(KV.second); continue; }
    Far[KV.second] = std::max(Far[KV.second], BlockPos.lookup(KV.first.second));
  }
  for (auto &KV : RegOf) {
    const Value *V = KV.first; Reg R = KV.second;
    if (R.Class != RC::VP || R.Idx == 0) continue;
    if (isa<Instruction>(V) && !liveInst(V)) continue;   // dead entry, register long since reused
    if (!isa<Instruction>(V) || isa<PHINode>(V) || CTPinned.count(V) || PreferReg.count(V) || CurOperands.count(V) || V == CurInst ||
        SpecialCalls.count(cast<Instruction>(V)) || SpilledVP.count(V) || !loopSafe(V, true) ||
        (InBlock && DefPos.lookup(V) >= BlockPos.lookup(CurBB))) { Excluded.insert(R.Idx); continue; }
    Far[R.Idx] = std::max(Far[R.Idx], LastUse.lookup(V));
  }
  unsigned Victim = 0, VFar = 0;
  for (auto &KV : Far) if (!Excluded.count(KV.first) && (!Victim || KV.second > VFar)) { Victim = KV.first; VFar = KV.second; }
  if (!Victim) {
    if (Diag && getenv("HWCC_DEBUG_VS")) {
      DenseMap<unsigned, std::string> Role;
      for (auto &KV : RegOf) if (KV.second.Class == RC::VP && KV.second.Idx && (!isa<Instruction>(KV.first) || liveInst(KV.first))) { std::string r = isa<Instruction>(KV.first) ? (isa<PHINode>(KV.first) ? "phi" : "i1val") : "const"; if (CurOperands.count(KV.first)) r += "(curop)"; if (PreferReg.count(KV.first)) r += "(prefer)"; Role[KV.second.Idx] += r + " "; }
      for (auto &KV : EdgeMask) Role[KV.second] += "edge ";
      for (auto &LS : LoopStack) { Role[LS.Active] += "loopActive "; for (auto &E : LS.ExitTotal) Role[E.second] += "loopExit "; }
      if (ZeroVP) Role[ZeroVP] += "zero ";
      for (unsigned P : PendingVP) Role[P] += "pendingReload ";
      if (!CurPred.empty()) Role[std::atoi(CurPred.c_str() + 2)] += "curPred ";
      errs() << "vp exhausted in " << F.getName() << " (loop depth " << LoopStack.size() << "):\n";
      for (unsigned i = 1; i < 16; i++) if (VPUsed[i]) errs() << "  vp" << i << ": " << (Role.count(i) ? Role[i] : "(temporary mask)") << " ref=" << (VPRef.size() > i ? VPRef[i] : 0) << "  [alloc at: " << VPAllocSite[i] << "]\n";
      errs() << "  now at: " << DbgWhere << "\n";
      for (auto &KV : SpilledVP) errs() << "  spilledVP " << KV.first->getName() << " -> vw" << KV.second << " lastuse " << LastUse.lookup(KV.first) << "\n";
    }
    return false;
  }
  // the receiving register: one that was free at the block top as well, so the store can be hoisted
  Reg T{RC::VW, 0}; bool Hoist = false;
  if (InBlock && OneVS) {
    if (VWUsed.size() < 256) VWUsed.resize(256, false);
    for (unsigned i = 0; i < NumVW; i++) if (!VWUsed[i] && (VWUsedAtTop.size() <= i || !VWUsedAtTop[i])) { VWUsed[i] = true; T = Reg{RC::VW, i}; Hoist = true; break; }
    if (!Hoist && (VCap >= 256 || NumVV + NumVW < VCap)) { T = alloc(RC::VW); Hoist = true; }   // a fresh register was free all along
  }
  if (!Hoist) T = alloc(RC::VW);
  std::string One = OneVS ? "vs" + std::to_string(OneVS) : regOfValue(ConstantInt::get(Type::getInt64Ty(F.getContext()), 1)).str();
  if (Hoist) { insertText(BodyStart, fmt("", "vaddw", {T.str(), "vs0", "vs0"}) + fmt(vp(Victim), "vaddw", {T.str(), "vs0", One})); NumInsts += 2; }
  else {
    emit("", "vaddw", {T.str(), "vs0", "vs0"});
    emit(vp(Victim), "vaddw", {T.str(), "vs0", One});
    if (InBlock) SpillInBlock = true;   // the block must run for the store to happen
  }
  SmallVector<const Value *, 4> Holders;
  for (auto &KV : RegOf) if (KV.second.Class == RC::VP && KV.second.Idx == Victim) Holders.push_back(KV.first);
  for (const Value *W : Holders) { RegOf.erase(W); if (liveInst(W)) { SpilledVP[W] = T.Idx; VWSpillRef[T.Idx]++; } }
  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 4> Edges;
  for (auto &KV : EdgeMask) if (KV.second == Victim) Edges.push_back(KV.first);
  for (auto &E : Edges) { EdgeSpilled[E] = T.Idx; EdgeMask.erase(E); VWSpillRef[T.Idx]++; }
  VPUsed[Victim] = false; if (VPRef.size() > Victim) VPRef[Victim] = 0;
  return true;
}

std::string WTGen::vecSlotAddrText(unsigned Slot) {
  uint64_t Off = 8 * Slot; std::string Base = "vs" + std::to_string(VSpillVS), A = "vs" + std::to_string(VSpillAddrVS), S;
  for (; Off > 2040; Off -= 2040) { S += fmt("", "vaddi", {A, Base, "2040"}); Base = A; }
  return S + fmt("", "vaddi", {A, Base, std::to_string(Off)});
}
void WTGen::vecSlotAddr(unsigned Slot) { Out << vecSlotAddrText(Slot); NumInsts++; }

// Free one vector register of class C: the live value with the farthest last use is stored to its
// per-lane spill slot (unpredicated: inactive lanes park garbage in their own slot).
bool WTGen::spillOneVec(RC C, bool Diag) {
  std::vector<bool> &Used = C == RC::VV ? VVUsed : VWUsed;
  DenseSet<unsigned> Excluded; DenseMap<unsigned, unsigned> Far;
  for (auto &KV : SelectReg) if (KV.second.Class == C && (!RegOf.count(KV.first) || liveInst(KV.first))) Excluded.insert(KV.second.Idx);
  for (auto &KV : Captured) if (KV.second.Class == C) Excluded.insert(KV.second.Idx);
  for (Reg &R : PendingVec) if (R.Class == C) Excluded.insert(R.Idx);
  if (C == RC::VV && HaveLaneOff) Excluded.insert(LaneOffVV.Idx);
  // phis may spill once their moves are done (a spilled header phi is updated at the latch with a
  // predicated store into its slot); phis of control-thread regions and LCSSA captures keep their register
  auto phiOk = [&](const PHINode *Phi) { return !Captured.count(Phi) && !inCTBody(Phi); };
  for (auto &KV : RegOf) {
    const Value *V = KV.first; Reg R = KV.second;
    if (R.Class != C) continue;
    if (isa<Instruction>(V) && !liveInst(V)) continue;   // dead entry, register long since reused
    if (!isa<Instruction>(V) || (isa<PHINode>(V) && !phiOk(cast<PHINode>(V))) || CTPinned.count(V) || PreferReg.count(V) || CurOperands.count(V) || V == CurInst ||
        SpecialCalls.count(cast<Instruction>(V)) || SpilledVec.count(V) || !DefEnd.count(V) || !loopSafe(V, false)) { Excluded.insert(R.Idx); continue; }
    Far[R.Idx] = std::max(Far[R.Idx], LastUse.lookup(V));
  }
  unsigned Victim = 0; bool Have = false; unsigned VFar = 0;
  for (auto &KV : Far) if (!Excluded.count(KV.first) && (!Have || KV.second > VFar)) { Victim = KV.first; VFar = KV.second; Have = true; }
  if (!Have) {
    if (Diag && getenv("HWCC_DEBUG_VS")) {
      errs() << (C == RC::VV ? "vv" : "vw") << " exhausted in " << F.getName() << " at " << DbgWhere << " (cap " << VCap << ", vv " << NumVV << " vw " << NumVW << "):\n";
      if (HaveLaneOff) errs() << "  " << LaneOffVV.str() << ": lane offset (vector spill slots)\n";
      for (auto &KV : RegOf) if ((KV.second.Class == RC::VV || KV.second.Class == RC::VW) && (!isa<Instruction>(KV.first) || liveInst(KV.first))) {
        std::string r = isa<Instruction>(KV.first) ? (isa<PHINode>(KV.first) ? "phi" : "val") : "nonInst";
        if (CurOperands.count(KV.first)) r += "(curop)"; if (PreferReg.count(KV.first)) r += "(prefer)"; if (CTPinned.count(KV.first)) r += "(ctpinned)"; if (KV.first == CurInst) r += "(cur)";
        if (auto *I = dyn_cast<Instruction>(KV.first)) if (SpecialCalls.count(I)) r += "(special)";
        if (Excluded.count(KV.second.Idx)) r += "(excluded)";
        errs() << "  " << KV.second.str() << ": " << r << " lastuse " << LastUse.lookup(KV.first) << "\n"; }
    }
    return false;
  }
  unsigned Slot = NumVSlots++;
  std::string Store = vecSlotAddrText(Slot) + fmt("", C == RC::VV ? "vsxd" : "vsxw", {Reg{C, Victim}.str(), "vs" + std::to_string(VSpillAddrVS), LaneOffVV.str()});
  SmallVector<const Value *, 4> Holders;
  for (auto &KV : RegOf) if (KV.second.Class == C && KV.second.Idx == Victim) Holders.push_back(KV.first);
  for (const Value *W : Holders) { RegOf.erase(W); if (liveInst(W)) { SpilledVec[W] = Slot; insertText(DefEnd[W], Store); NumInsts += 2; } }
  // a loop-carried value coalesced into a spilled phi register computes into its own register now
  SmallVector<const Value *, 4> Coalesced;
  for (auto &KV : PreferReg) if (KV.second.Class == C && KV.second.Idx == Victim) Coalesced.push_back(KV.first);
  for (const Value *W : Coalesced) { PreferReg.erase(W); PreferPhi.erase(W); }
  Used[Victim] = false;
  return true;
}

void WTGen::phiStore(unsigned Mask, PHINode *Phi, Value *Src) {
  RC C = classOf(Phi); Reg S = regOfValue(Src);
  if (S.Class != C) { Reg T = alloc(C); PendingVec.push_back(T); emit("", C == RC::VW ? "vaddw" : "vadd", {T.str(), S.str(), "vs0"}); S = T; }
  vecSlotAddr(SpilledVec[Phi]);
  emit(Mask ? vp(Mask) : "", C == RC::VV ? "vsxd" : "vsxw", {S.str(), "vs" + std::to_string(VSpillAddrVS), LaneOffVV.str()});
}

// Bring every spilled edge mask targeting BB back into a vp before the block consumes them.
void WTGen::restoreEdgesTo(BasicBlock *BB) {
  SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 4> Edges;
  for (auto &KV : EdgeSpilled) if (KV.first.second == BB) Edges.push_back(KV.first);
  if (Edges.empty()) return;
  RestoringFor = BB;
  DenseMap<unsigned, unsigned> Loaded;   // vw -> vp
  for (auto &E : Edges) {
    unsigned T = EdgeSpilled[E]; EdgeSpilled.erase(E);
    unsigned P;
    if (auto LI = Loaded.find(T); LI != Loaded.end()) P = shareVP(LI->second);
    else {
      P = allocVP(); Loaded[T] = P;
      std::string TS = Reg{RC::VW, T}.str();
      emit("", "vcmpeq", {vp(P), TS, "vs0"});
      emit("", "vpop", {vp(P), vp(P), vp(P), vp(P), "0x55"});   // not
    }
    EdgeMask[E] = P;
    if (--VWSpillRef[T] == 0) VWUsed[T] = false;
  }
  RestoringFor = nullptr;
}

// Uniform pool: keep >= 32 vs registers for temporaries by loading constants (and arguments, when
// there are many) from a per-kernel i64 array on each use.
Reg WTGen::poolLoad(Value *V) {
  if (!PoolVS) PoolVS = allocInputVS().Idx;
  auto It = PoolIdx.find(V);
  unsigned Idx = It != PoolIdx.end() ? It->second : (PoolIdx[V] = PoolConsts.size(), PoolConsts.push_back(V), PoolConsts.size() - 1);
  if (Pinning) return Reg{RC::VS, 0};
  Reg T = alloc(RC::VS); PendingTemps.push_back(T.Idx);
  uint64_t Off = 8 * Idx; std::string Base = "vs" + std::to_string(PoolVS);
  for (; Off > 2040; Off -= 2040) { emit("", "vaddi", {T.str(), Base, "2040"}); Base = T.str(); }   // 12-bit immediate
  emit("", "vaddi", {T.str(), Base, std::to_string(Off)});
  emit("", "vlsd", {T.str(), T.str()});
  return T;
}

// Operand lookup. Constants and arguments get a vs register that the control thread fills.
Reg WTGen::regOfValue(Value *V) {
  auto It = RegOf.find(V);
  if (It != RegOf.end()) return It->second;
  if (auto FI = Forward.find(V); FI != Forward.end()) return regOfValue(FI->second);
  if (auto SI = SpilledVP.find(V); SI != SpilledVP.end()) {   // reload a spilled predicate: T != 0
    Reg P = alloc(RC::VP); if (VPRef.size() <= P.Idx) VPRef.resize(P.Idx + 1, 0); VPRef[P.Idx] = 1; PendingVP.push_back(P.Idx);
    std::string T = Reg{RC::VW, SI->second}.str();
    emit("", "vcmpeq", {P.str(), T, "vs0"});
    emit("", "vpop", {P.str(), P.str(), P.str(), P.str(), "0x55"});   // not
    return P;
  }
  if (auto SI = SpilledVec.find(V); SI != SpilledVec.end()) {   // reload a spilled vector value into a temporary
    RC C = classOf(V); Reg T = alloc(C); PendingVec.push_back(T);
    vecSlotAddr(SI->second);
    emit("", C == RC::VV ? "vlxd" : "vlxw", {T.str(), "vs" + std::to_string(VSpillAddrVS), LaneOffVV.str()});
    return T;
  }
  if (auto SI = Spilled.find(V); SI != Spilled.end()) {   // reload a spilled temporary
    Reg T = alloc(RC::VS); PendingTemps.push_back(T.Idx);
    uint64_t Off = 8 * SI->second; std::string Base = "vs" + std::to_string(SpillVS);
    for (; Off > 2040; Off -= 2040) { emit("", "vaddi", {T.str(), Base, "2040"}); Base = T.str(); }
    emit("", "vaddi", {T.str(), Base, std::to_string(Off)});
    emit("", "vlsd", {T.str(), T.str()});
    return T;
  }
  if (auto *C = dyn_cast<Constant>(V)) {
    if (V->getType()->isIntegerTy(1)) {
      if (C->isOneValue()) { Reg R{RC::VP, 0}; RegOf[V] = R; return R; }          // vp0 is all ones
      if (!ZeroVP) { ZeroVP = alloc(RC::VP).Idx; emit("", "vpclear", {vp(ZeroVP)}); }
      Reg R{RC::VP, ZeroVP}; RegOf[V] = R; return R;
    }
    if (C->isNullValue()) { Reg R{RC::VS, 0}; RegOf[V] = R; return R; }
    if (freeVS() < 32)   // a constant address into a global (dwt2d: 55 offsets into one __local struct): base register + immediate
      if (auto *GEP = dyn_cast<GEPOperator>(C)) {
        APInt Off(64, 0);
        if (GEP->accumulateConstantOffset(DL, Off) && Off.isNonNegative()) {
          if (Pinning) { (void)regOfValue(GEP->getPointerOperand()); return Reg{RC::VS, 0}; }
          std::string Base = regOfValue(GEP->getPointerOperand()).str();
          Reg T = alloc(RC::VS); PendingTemps.push_back(T.Idx);
          uint64_t O = Off.getZExtValue();
          for (; O > 2040; O -= 2040) { emit("", "vaddi", {T.str(), Base, "2040"}); Base = T.str(); }
          emit("", "vaddi", {T.str(), Base, std::to_string(O)});
          return T;
        }
      }
    if ((isa<ConstantInt>(C) || isa<ConstantFP>(C)) && (PoolIdx.count(V) || freeVS() < 32)) return poolLoad(V);
    Reg R = allocInputVS(); RegOf[V] = R; VSInputs.push_back({R.Idx, V}); return R;
  }
  if (isa<Argument>(V)) {
    if (PoolIdx.count(V) || freeVS() < 32) return poolLoad(V);   // heartwall: 33 scalar arguments
    Reg R = allocInputVS(); RegOf[V] = R; VSInputs.push_back({R.Idx, V}); return R;
  }
  std::string Str; raw_string_ostream OS(Str); V->print(OS);
  report_fatal_error(Twine("operand has no register: ") + Str);
}

// Split S = U + D into a uniform part U and a divergent part D (either may be zero).
// Add expressions split per operand; add-recurrences split start and step separately, so a
// row address {A + f(id), +, 4} becomes U = {A,+,4} (scalar induction variable) and D = f(id).
static void splitUniform(ScalarEvolution &SE, const SCEV *S, const std::function<bool(const SCEV *)> &IsUniform,
                         const SCEV *&U, const SCEV *&D) {
  Type *T = S->getType();
  if (IsUniform(S)) { U = S; D = SE.getZero(T); return; }
  if (auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    SmallVector<SCEVUse, 4> Us, Ds;
    for (const SCEV *Op : Add->operands()) { const SCEV *u, *d; splitUniform(SE, Op, IsUniform, u, d); Us.push_back(u); Ds.push_back(d); }
    U = SE.getAddExpr(Us); D = SE.getAddExpr(Ds); return;
  }
  if (auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    if (AR->isAffine()) {
      const SCEV *Us, *Ds, *Ust, *Dst;
      splitUniform(SE, AR->getStart(), IsUniform, Us, Ds);
      splitUniform(SE, AR->getStepRecurrence(SE), IsUniform, Ust, Dst);
      U = SE.getAddRecExpr(Us, Ust, AR->getLoop(), SCEV::FlagAnyWrap);
      D = Dst->isZero() ? Ds : SE.getAddRecExpr(Ds, Dst, AR->getLoop(), SCEV::FlagAnyWrap);
      return;
    }
  }
  U = SE.getZero(T); D = S;
}

// ---- control-thread loop selection
bool WTGen::ctLoadOK(LoadInst *LD) {
  // the control thread's scalar load must not race with a vector store issued earlier in this
  // kernel: refuse if any store that may alias it can execute before it
  BasicBlock *BL = LD->getParent();
  if (!AA) {
    BAA = std::make_unique<BasicAAResult>(DL, F, KA.targetLibraryInfo(), KA.assumptionCache(), DT.get());
    TBAA = std::make_unique<TypeBasedAAResult>(/*UsingTypeSanitizer=*/false);
    AA = std::make_unique<AAResults>(KA.targetLibraryInfo());
    AA->addAAResult(*BAA); AA->addAAResult(*TBAA);
  }
  for (Instruction &I : instructions(F)) {
    if (!I.mayWriteToMemory() || isWorkGroupReduce(&I)) continue;
    if (isa<StoreInst>(I) && !isModSet(AA->getModRefInfo(&I, MemoryLocation::get(LD)))) continue;
    BasicBlock *BS = I.getParent();
    bool Reaches = BS == BL ? (I.comesBefore(LD) || LI->getLoopFor(BL) != nullptr) : isPotentiallyReachable(BS, BL, nullptr, DT.get(), LI.get());
    if (!Reaches) continue;
    // a store inside the region itself races with the load every iteration; one issued before the
    // region is drained by a fence on the control thread at region entry
    if (CandRegion && CandRegion->contains(BS)) return false;
    if (CandRegion) CandFence = true; else return false;
  }
  return true;
}
bool WTGen::ctCloneable(Value *V, DenseSet<Value *> &Seen, unsigned Depth) {
  bool R = ctCloneableImpl(V, Seen, Depth);
  if (!R && Verbose) { errs() << "hwacha-cc:   not cloneable: " << *V; if (auto *I = dyn_cast<Instruction>(V)) errs() << " uniform=" << isUniform(I) << " ka=" << KA.isUniform(I) << " mem=" << I->mayReadFromMemory(); errs() << "\n"; }
  return R;
}
bool WTGen::ctCloneableImpl(Value *V, DenseSet<Value *> &Seen, unsigned Depth) {
  if (Depth > 64 || !Seen.insert(V).second) return true;
  if (isa<Constant>(V) || isa<Argument>(V)) return !isa<GlobalVariable>(V) || true;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || !isUniform(I)) return false;
  if (isWorkItemId(I)) return false;
  if (isLocalSizeCall(I) || isGroupIdCall(I) || isGlobalSizeCall(I)) return true;
  if (isWorkGroupReduce(I)) return true;   // the control thread computes it itself
  if (isa<CallBase>(I)) return false;
  if (auto *LD = dyn_cast<LoadInst>(I)) { if (!ctLoadOK(LD)) return false; return ctCloneable(LD->getPointerOperand(), Seen, Depth + 1); }
  if (auto *Phi = dyn_cast<PHINode>(I)) {
    // inside a loop nest the control thread mirrors the whole CFG, so any phi there is fine
    if (LI->getLoopFor(Phi->getParent())) {
      for (Value *In : Phi->incoming_values()) if (!ctCloneable(In, Seen, Depth + 1)) return false;
      return true;
    }
    // LCSSA phi of a depth-1 loop (single incoming)
    if (Phi->getNumIncomingValues() == 1) {
      Loop *PL = LI->getLoopFor(Phi->getIncomingBlock(0));
      if (PL && !PL->contains(Phi->getParent())) return ctCloneable(Phi->getIncomingValue(0), Seen, Depth + 1);
    }
    return false;
  }
  if (I->mayHaveSideEffects() || I->mayReadFromMemory()) return false;
  for (Value *Op : I->operands()) if (!ctCloneable(Op, Seen, Depth + 1)) return false;
  return true;
}
// Is BB reached under uniform control flow only? Collects the branch conditions involved.
bool WTGen::reachUniform(BasicBlock *BB, DenseMap<BasicBlock *, int> &Memo, DenseSet<Value *> &Conds) {
  if (auto It = Memo.find(BB); It != Memo.end()) return It->second > 0;
  Memo[BB] = 0;   // in progress: treated as not uniform if we come back around
  bool Ok = true;
  if (BB != &F.getEntryBlock()) {
    for (BasicBlock *P : predecessors(BB)) {
      Loop *PL = LI->getLoopFor(P);
      if (PL && !PL->contains(BB)) {              // loop exit edge: every lane that entered leaves here
        while (PL->getParentLoop()) PL = PL->getParentLoop();
        SmallVector<BasicBlock *, 4> Exits; PL->getUniqueExitBlocks(Exits);
        if (Exits.size() != 1 || !reachUniform(PL->getLoopPreheader(), Memo, Conds)) { Ok = false; break; }
        continue;
      }
      if (PL) { Ok = false; break; }              // BB itself is inside a loop
      auto *Br = dyn_cast<BranchInst>(P->getTerminator());
      if (!Br) { Ok = false; break; }
      if (Br->isConditional() && isUniform(Br->getCondition())) Conds.insert(Br->getCondition());   // divergent: masked on the vector side
      if (!reachUniform(P, Memo, Conds)) { Ok = false; break; }
    }
  }
  Memo[BB] = Ok ? 1 : -1;
  return Ok;
}
void WTGen::selectCTLoops() {
  if (Opts.NoCTLoops) return;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->getParentLoop()) continue;
    BasicBlock *Pre = L->getLoopPreheader();
    if (!Pre || !L->getExitingBlock() || !L->getUniqueExitBlock()) { if (Verbose) errs() << "hwacha-cc: loop not a control-thread region (shape)\n"; continue; }
    bool Ok = true;
    CandRegion = L; CandFence = false;
    for (Loop *Sub : L->getLoopsInPreorder())
      if (!Sub->getLoopPreheader() || !Sub->getLoopLatch() || !Sub->getExitingBlock() || !Sub->getUniqueExitBlock()) Ok = false;
    DenseSet<Value *> Seen;
    for (BasicBlock *BB : L->blocks()) {
      if (!Ok) break;
      auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
      if (!Br) { Ok = false; break; }
      if (Br->isConditional()) { if (!isUniform(Br->getCondition())) { Ok = false; break; } Ok = ctCloneable(Br->getCondition(), Seen); }
      for (Instruction &I : *BB) {
        if (!Ok) break;
        if (I.isTerminator() || isa<StoreInst>(I)) continue;
        if (isWorkGroupReduce(&I)) continue;                                                     // a segment boundary inside the region
        if (isBarrierCall(&I) || (isa<CallBase>(I) && I.mayWriteToMemory())) { Ok = false; break; }   // barriers, atomics
        if (isa<CallBase>(I) && !isLocalSizeCall(&I) && !isGroupIdCall(&I) && !isGlobalSizeCall(&I)) { if (isUniform(&I)) Ok = false; continue; }   // uniform builtin
        if (isUniform(&I)) Ok = ctCloneable(&I, Seen);
      }
    }
    if (!Ok) { if (Verbose) errs() << "hwacha-cc: loop at " << L->getHeader()->getName() << " not a control-thread region (body)\n"; continue; }
    DenseMap<BasicBlock *, int> Memo; DenseSet<Value *> Conds;
    if (!reachUniform(Pre, Memo, Conds)) { if (Verbose) errs() << "hwacha-cc: loop not a control-thread region (reach)\n"; continue; }
    for (Value *C : Conds) Ok = Ok && ctCloneable(C, Seen);
    if (!Ok) { if (Verbose) errs() << "hwacha-cc: loop not a control-thread region (reach conditions)\n"; continue; }
    int Idx = CTLoops.size();
    CTLoop C; C.L = L; C.NeedFence = CandFence;
    C.AfterLabel = F.getName().str() + "_wt_a" + std::to_string(Idx);
    CTLoops.push_back(C); CTLoopOf[L] = Idx;
  }
  CandRegion = nullptr;
}

// Rewrite stream and gather addresses so the instruction selector only sees registers.
bool WTGen::materializeAddresses() {
  ScalarEvolution &SE = KA.scalarEvolution();
  Instruction *InsertPt = &*F.getEntryBlock().getFirstInsertionPt();
  SCEVExpander Exp(SE, "hw", /*PreserveLCSSA=*/false);
  auto isUniformSCEV = [&](const SCEV *S) {
    return !SCEVExprContains(S, [&](const SCEV *X) {
      auto *U = dyn_cast<SCEVUnknown>(X); return U && !isUniform(U->getValue()); });
  };
  // indexed access: uniform base in a vs register (scalar code in the block), divergent byte offset in vv
  auto indexed = [&](const MemAccess &A) -> bool {
    Value *Ptr = isa<LoadInst>(A.I) ? cast<LoadInst>(A.I)->getPointerOperand() : cast<StoreInst>(A.I)->getPointerOperand();
    const SCEV *S = SE.getSCEV(Ptr), *U, *D;
    splitUniform(SE, S, isUniformSCEV, U, D);
    GatherBase[A.I] = Exp.expandCodeFor(U, PointerType::get(F.getContext(), 0), A.I);
    GatherIndex[A.I] = Exp.expandCodeFor(D, Type::getInt64Ty(F.getContext()), A.I);
    KindOf[A.I] = AddrKind::Gather;
    return true;
  };
  // Hwacha has 32 address registers (va0-va31). Each new stream costs one (a strided stream two),
  // plus ScratchVA/TreeVA when the kernel reduces. When the budget is spent, further streams fall
  // back to indexed (gather) accesses, which use a vs base + a vv index and no va.
  unsigned vaUsed = 0; const unsigned vaBudget = 29;
  for (const MemAccess &A : KA.memAccesses()) {
    AccessOf[A.I] = &A;
    KindOf[A.I] = A.Kind;
    bool LoopCarried = A.Kind == AddrKind::Stream &&
                       SCEVExprContains(A.BaseSCEV, [](const SCEV *X) { return isa<SCEVAddRecExpr>(X); });
    // A base that advances with a control-thread loop is re-sent by the control thread before every
    // body vf: the access stays a real (unit-stride or strided) stream inside the loop.
    int PerIter = -1; BasicBlock *ExpandAt = nullptr;
    if (A.Kind == AddrKind::Stream && LoopCarried && isUniformSCEV(A.BaseSCEV) && (A.Stride || isUniformSCEV(A.StrideSCEV))) {
      // every add-recurrence must belong to one control-thread region; the base is evaluated at the
      // header of the innermost of those loops (SCEV carries the analysis' own LoopInfo: match by header)
      bool Ok = true; int Region = -1; Loop *Inner = nullptr;
      SCEVExprContains(A.BaseSCEV, [&](const SCEV *X) {
        auto *AR = dyn_cast<SCEVAddRecExpr>(X);
        if (!AR) return false;
        if (!AR->isAffine()) { Ok = false; return false; }
        Loop *ML = LI->getLoopFor(AR->getLoop()->getHeader());
        if (!ML || ML->getHeader() != AR->getLoop()->getHeader()) { Ok = false; return false; }
        Loop *Top = ML; while (Top->getParentLoop()) Top = Top->getParentLoop();
        auto It = CTLoopOf.find(Top);
        if (It == CTLoopOf.end() || (Region >= 0 && Region != It->second)) { Ok = false; return false; }
        Region = It->second;
        if (!Inner || ML->getLoopDepth() > Inner->getLoopDepth()) Inner = ML;
        if (A.StrideSCEV && !SE.isLoopInvariant(A.StrideSCEV, AR->getLoop())) Ok = false;
        return false; });
      if (Ok && Region >= 0) { PerIter = Region; ExpandAt = Inner->getHeader(); }
    }
    bool RuntimeStride = A.Kind == AddrKind::Stream && !A.Stride && isUniformSCEV(A.StrideSCEV) && !SCEVExprContains(A.StrideSCEV, [](const SCEV *X) { return isa<SCEVAddRecExpr>(X); });
    if (Verbose && A.Kind == AddrKind::Stream && LoopCarried) errs() << "hwacha-cc: loop-carried stream PerIter=" << PerIter << " ctloops=" << CTLoopOf.size() << " uniformBase=" << isUniformSCEV(A.BaseSCEV) << ": " << *A.I << "\n";
    if (A.Kind == AddrKind::Stream && PerIter < 0 && (LoopCarried || (!A.Stride && !RuntimeStride))) {
      // va registers are read-only inside the vf block and hold no stride at all, so a base that
      // advances inside the block (x[i+k]) or a runtime stride (a[i*ld]) becomes an indexed access:
      //   vs_base = uniform part (scalar induction variable in the block), vv_idx = S - base
      if (!indexed(A)) return false;
      continue;
    }
    bool EntryInvariant = A.Kind != AddrKind::Stream || !SCEVExprContains(A.BaseSCEV, [](const SCEV *X) {
      if (isa<SCEVAddRecExpr>(X)) return true;
      auto *U = dyn_cast<SCEVUnknown>(X); return U && isa<Instruction>(U->getValue()); });
    if (A.Kind == AddrKind::Stream && !EntryInvariant && PerIter < 0) { if (!indexed(A)) return false; continue; }
    if (A.Kind == AddrKind::Stream) {
      if (!isUniformSCEV(A.BaseSCEV)) return fail("stream base is not uniform", A.I);
      Value *Base, *StrideV = nullptr;
      if (PerIter >= 0)   // evaluate the base inside the loop (uniform scalar code, cloned to the control thread)
        Base = Exp.expandCodeFor(A.BaseSCEV, PointerType::get(F.getContext(), 0), &*ExpandAt->getFirstInsertionPt());
      else Base = Exp.expandCodeFor(A.BaseSCEV, PointerType::get(F.getContext(), 0), InsertPt);
      if (!A.Stride) StrideV = Exp.expandCodeFor(A.StrideSCEV, Type::getInt64Ty(F.getContext()), InsertPt);
      // the control thread must be able to compute these (the expander's new instructions are
      // unknown to the uniformity analysis until it is recomputed)
      KA.recomputeUniformity();
      DenseSet<Value *> Seen;
      CandRegion = PerIter >= 0 ? CTLoops[PerIter].L : nullptr; CandFence = false;
      bool Cloneable = ctCloneable(Base, Seen) && (!StrideV || ctCloneable(StrideV, Seen));
      if (PerIter >= 0 && CandFence) CTLoops[PerIter].NeedFence = true;
      CandRegion = nullptr;
      if (!Cloneable) {
        if (Verbose) errs() << "hwacha-cc: stream base not evaluable by the control thread, using indexed access: " << *Base << "\n";
        if (!indexed(A)) return false; continue;
      }
      // share a va register between accesses with the same base/stride
      unsigned Idx = Streams.size();
      for (unsigned i = 0; i < Streams.size(); i++)
        if (Streams[i].Base == Base && Streams[i].Stride == A.Stride && Streams[i].StrideV == StrideV && Streams[i].Local == A.Local && Streams[i].PerIter == PerIter) { Idx = i; break; }
      Type *ET = isa<LoadInst>(A.I) ? A.I->getType() : cast<StoreInst>(A.I)->getValueOperand()->getType();
      bool strided = (int64_t)DL.getTypeStoreSize(ET) != A.Stride;
      if (Idx == Streams.size()) {
        unsigned cost = strided ? 2 : 1;   // a strided stream also needs a va for its stride
        if (vaUsed + cost > vaBudget) {     // out of address registers: spill this access to a gather
          if (Verbose) errs() << "hwacha-cc: address registers exhausted, using indexed access for " << *A.I << "\n";
          if (!indexed(A)) return false;
          continue;
        }
        vaUsed += cost;
        Stream S{Base, A.Stride, Idx, A.Local}; S.StrideV = StrideV; S.PerIter = PerIter; Streams.push_back(S);
      }
      // a stream whose stride is not the element size needs the strided form (stride in a va register)
      if (strided) Streams[Idx].Unit = false;
      Streams[Idx].Blocks.insert(A.I->getParent());
      StreamOfInst[A.I] = Idx;
    } else if (A.Kind == AddrKind::Gather) {
      if (!indexed(A)) return false;
    }
  }
  return true;
}

// Backward liveness from side effects; stream/gather pointer operands are replaced by registers.
void WTGen::computeNeeded() {
  SmallVector<const Value *, 32> Work;
  for (Instruction &I : instructions(F))
    if (isa<StoreInst>(I) || I.isTerminator() || isBarrierCall(&I) ||
        (isa<CallBase>(I) && cast<CallBase>(I).getCalledFunction() && cast<CallBase>(I).getCalledFunction()->getName().contains("atom")) ||
        // a call with memory side effects that is not a known builtin is work, not dead code: keep it so
        // that codegen reports it ("unsupported call") instead of silently dropping it
        (isa<CallBase>(I) && cast<CallBase>(I).getCalledFunction() && !cast<CallBase>(I).getCalledFunction()->isIntrinsic() &&
         !cast<CallBase>(I).getCalledFunction()->getName().starts_with("_Z") && cast<CallBase>(I).mayWriteToMemory()))
      Work.push_back(&I);
  while (!Work.empty()) {
    const Value *V = Work.pop_back_val();
    auto *I = dyn_cast<Instruction>(V);
    if (!I || !Needed.insert(I).second) continue;
    auto It = AccessOf.find(I);
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      Value *Op = I->getOperand(i);
      if (It != AccessOf.end()) {
        bool IsPtr = (isa<LoadInst>(I) && i == 0) || (isa<StoreInst>(I) && i == 1);
        if (IsPtr && KindOf.lookup(I) != AddrKind::Uniform) continue;   // replaced by va / vs+index
      }
      Work.push_back(Op);
    }
    if (It != AccessOf.end() && KindOf.lookup(I) == AddrKind::Gather) {
      Work.push_back(GatherIndex[I]);
      Work.push_back(GatherBase[I]);
    }
  }
}

// ---- linear order: blocks in topological order, each loop contiguous (LoopBegin ... LoopEnd)
void WTGen::linearize(Loop *L, const SmallPtrSetImpl<BasicBlock *> &Members, BasicBlock *Entry) {
  // nodes: member blocks not in a sub-loop, plus sub-loops represented by their header
  auto repOf = [&](BasicBlock *B) -> BasicBlock * {
    Loop *Inner = LI->getLoopFor(B);
    while (Inner && Inner->getParentLoop() != L && Inner != L) Inner = Inner->getParentLoop();
    return (Inner && Inner != L) ? Inner->getHeader() : B;
  };
  DenseMap<BasicBlock *, unsigned> InDeg; std::vector<BasicBlock *> Nodes;
  for (BasicBlock &B : F) if (Members.count(&B) && repOf(&B) == &B) { Nodes.push_back(&B); InDeg[&B] = 0; }
  auto succsOf = [&](BasicBlock *N, SmallVectorImpl<BasicBlock *> &Out) {
    Loop *Sub = LI->getLoopFor(N); bool IsSub = Sub && Sub != L && Sub->getHeader() == N && (L ? L->contains(Sub) : true) && repOf(N) == N && LI->getLoopFor(N) != L;
    SmallVector<BasicBlock *, 8> Raw;
    if (IsSub && Sub->getHeader() == N && (!L || Sub->getParentLoop() == L)) Sub->getExitBlocks(Raw);
    else for (BasicBlock *S : successors(N)) Raw.push_back(S);
    for (BasicBlock *S : Raw) {
      if (!Members.count(S)) continue;                 // leaves this region (loop exit)
      if (L && S == L->getHeader()) continue;          // back edge
      BasicBlock *R = repOf(S);
      if (R != N) Out.push_back(R);
    }
  };
  for (BasicBlock *N : Nodes) { SmallVector<BasicBlock *, 8> Ss; succsOf(N, Ss); for (BasicBlock *S : Ss) InDeg[S]++; }
  std::vector<BasicBlock *> Ready{Entry};
  DenseSet<BasicBlock *> Done;
  while (!Ready.empty()) {
    BasicBlock *N = Ready.front(); Ready.erase(Ready.begin());
    if (!Done.insert(N).second) continue;
    Loop *Sub = LI->getLoopFor(N);
    if (Sub && Sub != L && Sub->getHeader() == N && Sub->getParentLoop() == L) {
      Order.push_back({Item::LoopBegin, N, Sub});
      SmallPtrSet<BasicBlock *, 16> SubMembers(Sub->block_begin(), Sub->block_end());
      linearize(Sub, SubMembers, N);
      Order.push_back({Item::LoopEnd, N, Sub});
    } else {
      Order.push_back({Item::Block, N, nullptr});
    }
    SmallVector<BasicBlock *, 8> Ss; succsOf(N, Ss);
    for (BasicBlock *S : Ss) if (--InDeg[S] == 0) Ready.push_back(S);
  }
}

// ---- positions & last uses over the linear order (loop-aware)
void WTGen::computePositions() {
  unsigned Pos = 0;
  std::vector<std::pair<Loop *, unsigned>> Open;
  auto use = [&](Value *V, unsigned P) { LastUse[V] = std::max(LastUse[V], P); UsePos[V].push_back(P); };
  for (Item &It : Order) {
    Pos++;
    if (It.K == Item::LoopBegin) {
      LoopRange[It.L].first = Pos; Open.push_back({It.L, Pos});
      for (PHINode &Phi : It.BB->phis()) { DefPos[&Phi] = Pos; use(Phi.getIncomingValueForBlock(It.L->getLoopPreheader()), Pos); }
      continue;
    }
    if (It.K == Item::LoopEnd) {
      LoopRange[It.L].second = Pos;
      for (PHINode &Phi : It.BB->phis()) { PhiRealLast[&Phi] = LastUse.lookup(&Phi); use(Phi.getIncomingValueForBlock(It.L->getLoopLatch()), Pos); use(&Phi, Pos); }
      Open.pop_back();
      continue;
    }
    BasicBlock *BB = It.BB; BlockPos[BB] = Pos;
    Loop *CurL = Open.empty() ? nullptr : Open.back().first;
    bool IsHeader = CurL && CurL->getHeader() == BB;
    for (PHINode &Phi : BB->phis()) {
      if (IsHeader) continue;                         // handled at LoopBegin/LoopEnd
      DefPos[&Phi] = Pos;
      for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
        BasicBlock *P = Phi.getIncomingBlock(i);
        Loop *PL = LI->getLoopFor(P);
        // LCSSA exit phi: the value is captured at the exiting block's terminator
        if (PL && !PL->contains(BB)) use(Phi.getIncomingValue(i), BlockPos.lookup(P) + 1 + P->size());
        else use(Phi.getIncomingValue(i), Pos);
      }
    }
    for (Instruction &I : *BB) {
      if (isa<PHINode>(I)) continue;
      Pos++; DefPos[&I] = Pos;
      for (Value *Op : I.operands()) use(Op, Pos);
      // implicit operands of indexed accesses (materialized base / offset values)
      if (auto It = GatherIndex.find(&I); It != GatherIndex.end()) { use(It->second, Pos); use(GatherBase[&I], Pos); }
    }
  }
  // aliases: a no-op cast (sext of a wider int, trunc, bitcast, ptrtoint, inttoptr, freeze) shares
  // its operand's register, so the operand lives as long as the alias does. Emission also extends
  // it, but too late: releaseAt(operand's last use) has run by the time the cast is emitted (srad:
  // `ei` was freed one instruction before its sext, and the register was handed to a phi).
  for (bool Changed = true; Changed;) {
    Changed = false;
    for (Instruction &I : instructions(F)) {
      Value *Src = nullptr;
      if (auto *C = dyn_cast<CastInst>(&I)) {
        auto Op = C->getOpcode();
        bool Alias = Op == Instruction::PtrToInt || Op == Instruction::IntToPtr || Op == Instruction::BitCast ||
                     (Op == Instruction::SExt && !C->getSrcTy()->isIntegerTy(1)) || (Op == Instruction::Trunc && !C->getDestTy()->isIntegerTy(1));
        if (Alias) Src = C->getOperand(0);
      } else if (auto *Fr = dyn_cast<FreezeInst>(&I)) Src = Fr->getOperand(0);
      if (!Src || !isa<Instruction>(Src)) continue;
      unsigned L = LastUse.lookup(&I);
      if (L > LastUse.lookup(Src)) { LastUse[Src] = L; Changed = true; }
    }
  }
  // values defined before a loop and used inside it live until the loop ends
  for (auto &KV : LastUse) {
    unsigned D = DefPos.lookup(KV.first);
    for (auto &LR : LoopRange)
      if (KV.second >= LR.second.first && KV.second <= LR.second.second && D < LR.second.first)
        KV.second = std::max(KV.second, LR.second.second);
  }
}

unsigned WTGen::andMask(unsigned A, unsigned B, bool NegB) {
  unsigned D = allocVP();
  // vpop truth table over (a=A, b=B, c=B): index = a | b<<1 | c<<2; with c==b only 0,1,6,7 occur
  // a&b -> index 7 ; a&!b -> index 1 ; (vp0 is all-ones so A==0 uses the same tables)
  emit("", "vpop", {vp(D), vp(A), vp(B), vp(B), NegB ? "0x02" : "0x80"});
  return D;
}
unsigned WTGen::orInto(unsigned Acc, unsigned B) {
  if (!Acc) { unsigned D = allocVP(); emit("", "vpop", {vp(D), vp(B), vp(B), vp(B), "0xAA"}); return D; }
  emit("", "vpop", {vp(Acc), vp(Acc), vp(B), vp(B), "0xEE"});
  return Acc;
}
void WTGen::movePred(unsigned Pred, Reg Dst, Value *Src) {
  std::string P = Pred ? vp(Pred) : "";
  std::string S = regOfValue(Src).str();
  if (Dst.str() == S) return;                                  // coalesced: nothing to move
  if (Dst.Class == RC::VP) {
    // predicate-logic ops are never masked (Spike writes the PRF unpredicated, the RTL issues vipred
    // without a predicate operand), so a conditional predicate move is a 3-input mux: D = M ? S : D
    if (Pred == 0) emit("", "vpop", {Dst.str(), S, S, S, "0xAA"});
    else emit("", "vpop", {Dst.str(), vp(Pred), S, Dst.str(), "0xD8"});
  }
  else emit(P, (Dst.Class == RC::VW || isNarrow(Src)) ? "vaddw" : "vadd", {Dst.str(), S, "vs0"});
}

// exiting edge bookkeeping for every loop the edge leaves
bool WTGen::handleEdge(BasicBlock *From, BasicBlock *To, unsigned Mask, unsigned Pos) {
  EdgeMask[{From, To}] = shareVP(Mask);
  for (int i = (int)LoopStack.size() - 1; i >= 0; i--) {
    LoopState &LS = LoopStack[i];
    if (LS.L->contains(To)) break;
    // capture LCSSA phi values for lanes leaving now
    for (PHINode &Phi : To->phis()) {
      int Idx = Phi.getBasicBlockIndex(From);
      if (Idx < 0) continue;
      Reg &Cap = Captured[&Phi];
      if (!Cap.Idx && Cap.Class != RC::VP && !RegOf.count(&Phi)) { Cap = alloc(classOf(&Phi)); RegOf[&Phi] = Cap; }
      Cap = RegOf[&Phi];
      movePred(Mask, Cap, Phi.getIncomingValue(Idx));
    }
    emit("", "vpop", {vp(LS.ExitTotal[To]), vp(LS.ExitTotal[To]), vp(Mask), vp(Mask), "0xEE"});   // ExitTotal |= Mask
    emit("", "vpop", {vp(LS.Active), vp(LS.Active), vp(Mask), vp(Mask), "0x02"});   // Active &= !Mask, in place
  }
  return true;
}

bool WTGen::beginLoop(Loop *L, unsigned &Pos) {
  Pos++; CurPos = Pos;
  // Loop entry is the one point where values from outside the loop can still leave their registers
  // (nothing in the loop has read them yet), so make room here for what the body will need.
  if (getenv("HWCC_DEBUG_VS")) DbgWhere = "beginLoop " + std::to_string(Pos) + " " + L->getHeader()->getName().str();
  if (VPUsed.size() < 16) VPUsed.resize(16, false);
  { unsigned Free = 0; for (unsigned i = 1; i < VPLimit; i++) if (!VPUsed[i]) Free++; while (Free < std::min(8u, VPLimit / 2) && spillOneVP()) Free++; }
  while (freeVS() < 12 && spillOne()) {}
  if (VCap < 256) while (freeVec() < 12 && (spillOneVec(RC::VW) || spillOneVec(RC::VV))) {}
  if (getenv("HWCC_DEBUG_VS")) DbgWhere = "beginLoop " + std::to_string(Pos) + " " + L->getHeader()->getName().str();
  BasicBlock *H = L->getHeader(), *Pre = L->getLoopPreheader();
  restoreEdgesTo(H);
  unsigned PreMask = EdgeMask.lookup({Pre, H});
  LoopState LS; LS.L = L;
  LS.Active = allocVP();
  if (PreMask == 0) emit("", "vpset", {vp(LS.Active)});
  else emit("", "vpop", {vp(LS.Active), vp(PreMask), vp(PreMask), vp(PreMask), "0xAA"});
  for (PHINode &Phi : H->phis()) {
    Reg D = alloc(classOf(&Phi)); RegOf[&Phi] = D;
    CurInst = &Phi;
    movePred(PreMask, D, Phi.getIncomingValueForBlock(Pre));
    CurInst = nullptr; DefEnd[&Phi] = Text.size();
    // coalesce: the latch value may be computed straight into the phi register if the phi is not
    // read after that point in the body (its register is kept allocated until LoopEnd anyway)
    Value *Next = Phi.getIncomingValueForBlock(L->getLoopLatch());
    if (auto *NI = dyn_cast<Instruction>(Next))
      if (!Opts.NoCoalesce && L->contains(NI) && !isa<PHINode>(NI) && classOf(NI) == classOf(&Phi) && !PreferReg.count(NI))
        { PreferReg[NI] = D; PreferPhi[NI] = &Phi; }
  }
  // one accumulator per exit block, cleared before the loop, OR-ed at every exit edge
  SmallVector<BasicBlock *, 4> Exits; L->getExitBlocks(Exits);
  for (BasicBlock *E : Exits) if (!LS.ExitTotal.count(E)) { LS.ExitTotal[E] = allocVP(); emit("", "vpclear", {vp(LS.ExitTotal[E])}); }
  LS.Label = ".L" + F.getName().str() + "_loop" + std::to_string(LabelCounter++);
  Out << LS.Label << ":\n";
  LoopStack.push_back(LS);
  releaseAt(Pos);
  return true;
}

bool WTGen::endLoop(Loop *L, unsigned &Pos) {
  Pos++; CurPos = Pos;
  if (getenv("HWCC_DEBUG_VS")) DbgWhere = "endLoop " + std::to_string(Pos) + " " + L->getHeader()->getName().str();
  LoopState LS = LoopStack.back(); LoopStack.pop_back();
  BasicBlock *H = L->getHeader(), *Latch = L->getLoopLatch();
  restoreEdgesTo(H);
  unsigned Back = EdgeMask.lookup({Latch, H});
  // lanes taking the back edge are exactly the still-active ones
  for (PHINode &Phi : H->phis()) {
    CurInst = &Phi;
    if (SpilledVec.count(&Phi)) phiStore(Back, &Phi, Phi.getIncomingValueForBlock(Latch));
    else movePred(Back, RegOf[&Phi], Phi.getIncomingValueForBlock(Latch));
  }
  CurInst = nullptr;
  if (!LinkVS) LinkVS = alloc(RC::VS).Idx;
  emit(vp(Back), "vcjal", {"1", "vs" + std::to_string(LinkVS), LS.Label});
  freeVP(Back); EdgeMask.erase({Latch, H});
  freeVP(LS.Active);
  // exit blocks get their predicate from the accumulated exit masks
  for (auto &KV : LS.ExitTotal) {
    // record as a synthetic edge (loop -> exit block); consumed when the exit block is emitted. A
    // real header -> exit edge is already folded into the total: drop it instead of leaking its vp.
    if (auto It = EdgeMask.find({H, KV.first}); It != EdgeMask.end()) { freeVP(It->second); EdgeMask.erase(It); }
    if (auto It = EdgeSpilled.find({H, KV.first}); It != EdgeSpilled.end()) { if (--VWSpillRef[It->second] == 0) VWUsed[It->second] = false; EdgeSpilled.erase(It); }
    EdgeMask[{H, KV.first}] = KV.second;
  }
  releaseAt(Pos);
  return true;
}

bool WTGen::emitBlock(BasicBlock *BB, unsigned &Pos) {
  Pos++; CurPos = Pos;
  if (getenv("HWCC_DEBUG_VS")) DbgWhere = "block entry " + std::to_string(Pos) + " " + BB->getName().str();
  Loop *CurL = LoopStack.empty() ? nullptr : LoopStack.back().L;
  bool IsHeader = CurL && CurL->getHeader() == BB;
  // ---- block predicate
  unsigned Pred = 0;
  if (IsHeader) Pred = shareVP(LoopStack.back().Active);
  else if (BB == &F.getEntryBlock()) Pred = 0;
  else {
    restoreEdgesTo(BB);
    SmallVector<unsigned, 4> In;
    for (BasicBlock *P : predecessors(BB)) {
      auto It = EdgeMask.find({P, BB});
      if (It != EdgeMask.end()) In.push_back(It->second);
    }
    // exits of a just-finished loop arrive as (header -> BB) synthetic edges
    for (auto &KV : EdgeMask) if (KV.first.second == BB && !is_contained(predecessors(BB), KV.first.first)) In.push_back(KV.second);
    if (In.empty()) return fail("block has no incoming edge mask (unsupported CFG shape)", BB);
    if (In.size() == 1) Pred = shareVP(In[0]);
    else { Pred = 0; for (unsigned M : In) Pred = orInto(Pred, M); }
  }
  BlockPred[BB] = Pred;
  CurPred = Pred ? vp(Pred) : "";
  // ---- phis
  if (!IsHeader)
    for (PHINode &Phi : BB->phis()) {
      if (Captured.count(&Phi)) continue;             // LCSSA phi already resolved at the exit edges
      Reg D = alloc(classOf(&Phi)); RegOf[&Phi] = D;
      CurInst = &Phi;
      for (unsigned i = 0; i < Phi.getNumIncomingValues(); i++) {
        unsigned M = EdgeMask.lookup({Phi.getIncomingBlock(i), BB});
        movePred(M, D, Phi.getIncomingValue(i));
      }
      CurInst = nullptr; DefEnd[&Phi] = Text.size();
    }
  // incoming edge masks are consumed
  for (BasicBlock *P : predecessors(BB)) { auto It = EdgeMask.find({P, BB}); if (It != EdgeMask.end()) { freeVP(It->second); EdgeMask.erase(It); } }
  {
    SmallVector<std::pair<const BasicBlock *, const BasicBlock *>, 4> Dead;
    for (auto &KV : EdgeMask) if (KV.first.second == BB) { freeVP(KV.second); Dead.push_back(KV.first); }
    for (auto &K : Dead) EdgeMask.erase(K);
  }
  releaseAt(Pos);
  // ---- body (skipped with a consensual jump when no lane is active in this block)
  BodyStart = Text.size();
  BlockSplit = false; SpillInBlock = false;
  InBlock = true; CurBB = BB; VWUsedAtTop = VWUsed; VVUsedAtTop = VVUsed;
  for (Instruction &I : *BB) {
    if (isa<PHINode>(I)) continue;
    Pos++;
    if (Needed.count(&I) && !SpecialCalls.count(&I) && !I.isTerminator()) { if (!emitInst(I, Pos)) return false; }
    if (!I.isTerminator()) releaseAt(Pos);
  }
  // A consensual jump costs a predicate reduction (~50 cycles on the RTL, the scalar unit stalls
  // until the vector unit answers). A loop header is executed every iteration and is normally only
  // fully inactive right before the loop exits, so skipping it is overhead -- except that a loop can
  // also be *entered* with no lane active: the guard before it was all-false and the consensual jump
  // around the guarded region lands on the loop, whose rotated body then runs once with garbage
  // uniform values (lud_diagonal's `if (tx > i) for (j < i)` at i = 0). Vector ops under a false
  // predicate are harmless, but a uniform load / store (vls* / vss*) is unpredicated and faults, so
  // headers with uniform memory ops get the jump too. Other blocks are skipped when they hold at
  // least two instructions.
  bool HasUniformMem = false;
  for (Instruction &I : *BB)
    if ((isa<LoadInst>(I) || isa<StoreInst>(I)) && Needed.count(&I) && KindOf.lookup(&I) == AddrKind::Uniform) HasUniformMem = true;
  InBlock = false; CurBB = nullptr;
  if (Pred != 0 && !Opts.NoSkip && (!IsHeader || HasUniformMem) && !BlockSplit && !SpillInBlock) {
    size_t Lines = std::count(Text.begin() + BodyStart, Text.end(), '\n');
    if (Lines >= 2 || (Lines >= 1 && HasUniformMem)) {
      if (!LinkVS) LinkVS = alloc(RC::VS).Idx;
      std::string Skip = ".L" + F.getName().str() + "_skip" + std::to_string(LabelCounter++);
      std::string Jump = "    @!" + vp(Pred) + " vcjal 0, vs" + std::to_string(LinkVS) + ", " + Skip + "\n";
      insertText(BodyStart, Jump);
      NumInsts++; NumJumps++;
      Out << Skip << ":\n";
    }
  }
  // ---- terminator -> edge masks
  Instruction *T = BB->getTerminator();
  if (auto *Br = dyn_cast<BranchInst>(T)) {
    if (Br->isConditional()) {
      unsigned C = regOfValue(Br->getCondition()).Idx;
      unsigned MT = andMask(Pred, C, false), MF = andMask(Pred, C, true);
      handleEdge(BB, Br->getSuccessor(0), MT, Pos); freeVP(MT);
      handleEdge(BB, Br->getSuccessor(1), MF, Pos); freeVP(MF);
    } else {
      handleEdge(BB, Br->getSuccessor(0), Pred, Pos);
    }
  } else if (!isa<ReturnInst>(T)) return fail("unsupported terminator", T);
  freeVP(Pred);
  releaseAt(Pos);     // the branch condition, if this was its last use
  CurPred.clear();
  return true;
}

// ---- control-thread regions: a depth-1 loop nest whose control flow is entirely uniform.
// Every kernel block becomes its own vf block; phi moves go at the tail of the source block when
// the moves of all successors can be ordered without clobbering (all lanes take the same edge, and
// phi registers are exclusively held), otherwise into a per-edge vf block. The whole region runs
// under the mask that reached the loop, so a divergent guard around the nest is fine.
bool WTGen::emitCTRegion(size_t &Idx, unsigned &Pos) {
  Loop *L = Order[Idx].L; BasicBlock *H = L->getHeader(), *Pre = L->getLoopPreheader();
  CTLoop &C = CTLoops[CTLoopOf[L]];
  restoreEdgesTo(H);
  unsigned PreMask = EdgeMask.lookup({Pre, H});
  C.Mask = shareVP(PreMask);
  Pos++;   // LoopBegin
  // header phis: initial values move in the segment before the region
  for (PHINode &Phi : H->phis()) {
    if (!Needed.count(&Phi)) continue;
    if (ctSkip(&Phi)) registerIterInput(&Phi, L, C);
    else movePred(PreMask, phiReg(&Phi), Phi.getIncomingValueForBlock(Pre));
  }
  { auto It = EdgeMask.find({Pre, H}); if (It != EdgeMask.end()) { freeVP(It->second); EdgeMask.erase(It); } }
  // coalesce loop-carried values into their header phi's register (every loop of the nest)
  SmallVector<Loop *, 4> Nest{L}; for (Loop *Sub : L->getLoopsInPreorder()) Nest.push_back(Sub);
  for (Loop *NL : Nest)
    for (PHINode &Phi : NL->getHeader()->phis()) {
      if (!Needed.count(&Phi) || ctSkip(&Phi)) continue;
      Reg D = phiReg(&Phi);
      Value *Next = Phi.getIncomingValueForBlock(NL->getLoopLatch());
      if (auto *NI = dyn_cast<Instruction>(Next))
        if (!Opts.NoCoalesce && NL->contains(NI) && !isa<PHINode>(NI) && classOf(NI) == classOf(&Phi) && !PreferReg.count(NI))
          { PreferReg[NI] = D; PreferPhi[NI] = &Phi; }
    }
  releaseAt(Pos);
  Out << "    vstop\n";
  for (Idx++; !(Order[Idx].K == Item::LoopEnd && Order[Idx].L == L); Idx++) {
    Item &It = Order[Idx];
    if (It.K == Item::Block) { if (!emitCTBlock(It.BB, Pos, C)) return false; }
    else { Pos++; releaseAt(Pos); }          // inner LoopBegin / LoopEnd: nothing to emit
  }
  Pos++;   // LoopEnd
  Out << "    .globl " << C.AfterLabel << "\n" << C.AfterLabel << ":\n";
  Segments.push_back({C.AfterLabel, CTLoopOf[L]});   // CT: the region issued right before this segment
  BasicBlock *Exiting = L->getExitingBlock(), *Exit = L->getUniqueExitBlock();
  EdgeMask[{Exiting, Exit}] = shareVP(C.Mask);
  freeVP(C.Mask);
  for (auto &[VS, V] : C.IterInputs) { CTPinned.erase(V); if (LastUse.lookup(V) <= Pos) release(V); }
  releaseAt(Pos);
  return true;
}

bool WTGen::emitCTBlock(BasicBlock *BB, unsigned &Pos, CTLoop &C) {
  Pos++; CurPos = Pos;
  Loop *L = C.L;
  CurPred = C.Mask ? vp(C.Mask) : "";
  std::string Label = F.getName().str() + "_wt_r" + std::to_string(CTLoopOf[L]) + "_b" + std::to_string(C.BlockSeg.size());
  C.BlockSeg[BB] = Label;
  Out << "    .globl " << Label << "\n" << Label << ":\n";
  // phis of this block: registers only (the moves happen on the incoming edges)
  for (PHINode &Phi : BB->phis()) {
    if (!Needed.count(&Phi)) continue;
    if (ctSkip(&Phi)) { if (!RegOf.count(&Phi)) registerIterInput(&Phi, L, C); }
    else (void)phiReg(&Phi);
  }
  releaseAt(Pos);
  for (Instruction &I : *BB) {
    if (isa<PHINode>(I)) continue;
    Pos++;
    if (Needed.count(&I) && !SpecialCalls.count(&I) && !I.isTerminator()) {
      if (ctSkip(&I)) registerIterInput(&I, L, C);
      else if (!emitInst(I, Pos)) return false;
    }
    if (!I.isTerminator()) releaseAt(Pos);
  }
  // successor phi moves
  struct Move { BasicBlock *S; PHINode *Phi; Value *V; Reg Dst; Reg Src; };
  std::vector<Move> Moves;
  for (BasicBlock *S : successors(BB)) {
    bool IsExit = !L->contains(S);
    Loop *BL = LI->getLoopFor(BB);
    bool InnerExit = !IsExit && BL && BL != L && !BL->contains(S);   // leaving an inner loop of the region
    for (PHINode &Phi : S->phis()) {
      int In = Phi.getBasicBlockIndex(BB);
      if (In < 0 || !Needed.count(&Phi)) continue;
      if (!IsExit && ctSkip(&Phi)) continue;                    // uniform: the control thread's phi
      Value *V = Phi.getIncomingValue(In);
      if ((IsExit || InnerExit) && !RegOf.count(&Phi)) {
        // the loop's LCSSA phi of a value coalesced into a header phi register: alias it, no copy
        // per iteration (the header phi is dead after the loop; keep its register until the exit
        // phi's own last use)
        auto PI = PreferPhi.find(V);
        if (PI != PreferPhi.end() && RegOf.lookup(V).Idx == RegOf.lookup(PI->second).Idx && RegOf.lookup(V).Class == RegOf.lookup(PI->second).Class && classOf(&Phi) == RegOf.lookup(V).Class) {
          Reg D = RegOf[V]; RegOf[&Phi] = D; Captured[&Phi] = D; Alias.insert(&Phi);
          LastUse[PI->second] = std::max(LastUse[PI->second], LastUse.lookup(&Phi));
          continue;
        }
      }
      Reg Dst = phiReg(&Phi);
      if (IsExit) Captured[&Phi] = Dst;
      Moves.push_back({S, &Phi, V, Dst, regOfValue(V)});
    }
  }
  // try one tail order: a move must not overwrite a register a later move still reads
  std::vector<Move> Ordered; std::vector<bool> Done(Moves.size(), false);
  for (size_t n = 0; n < Moves.size(); n++) {
    bool Found = false;
    for (size_t i = 0; i < Moves.size() && !Found; i++) {
      if (Done[i]) continue;
      bool Clobbers = false;
      for (size_t j = 0; j < Moves.size(); j++)
        if (!Done[j] && j != i && Moves[j].Src.Class == Moves[i].Dst.Class && Moves[j].Src.Idx == Moves[i].Dst.Idx) Clobbers = true;
      if (!Clobbers) { Ordered.push_back(Moves[i]); Done[i] = true; Found = true; }
    }
    if (!Found) break;
  }
  if (Ordered.size() == Moves.size()) {
    for (Move &M : Ordered) movePred(0, M.Dst, M.V);
    Out << "    vstop\n";
  } else {
    Out << "    vstop\n";
    for (BasicBlock *S : successors(BB)) {
      bool Any = false;
      for (Move &M : Moves) if (M.S == S) Any = true;
      if (!Any) continue;
      std::string E = F.getName().str() + "_wt_r" + std::to_string(CTLoopOf[L]) + "_e" + std::to_string(C.EdgeSeg.size());
      C.EdgeSeg[{BB, S}] = E;
      Out << "    .globl " << E << "\n" << E << ":\n";
      for (Move &M : Moves) if (M.S == S) movePred(0, M.Dst, M.V);
      Out << "    vstop\n";
    }
  }
  releaseAt(Pos);
  CurPred.clear();
  return true;
}

bool WTGen::run() {
  DT = std::make_unique<DominatorTree>(F);
  LI = std::make_unique<LoopInfo>(*DT);
  for (Loop *L : LI->getLoopsInPreorder())
    if (!L->getLoopPreheader() || !L->getLoopLatch()) return fail("loop is not in simplified form");
  selectCTLoops();
  // SCEV proves facts per thread (a dominating `tx <= m` guard, loop guards) and its expander turns a
  // sext into a zext, or splits an AddRec into zext(start) + step*i, on the strength of them. Under
  // predicated SIMT execution those proofs do not hold for the values as computed (nw: the start
  // 17 - 17*tx is negative for the lanes the guard admits, and zext(-17) + 34 != 17). The kernel's
  // own IR keeps its sext / zext as written; every integer zext of a 32-bit or wider value the
  // expander creates is turned into a sext, the value clang's signed index arithmetic means. A
  // narrower one is a bit-field extraction (`and x, 48` -> `16 * zext i2 (trunc (x /u 4))`,
  // lp_pool2d) and stays unsigned.
  DenseSet<const Instruction *> Before;
  for (Instruction &I : instructions(F)) Before.insert(&I);
  if (!materializeAddresses()) return false;
  for (Instruction &I : llvm::make_early_inc_range(instructions(F))) {
    auto *Z = dyn_cast<ZExtInst>(&I);
    if (!Z || Before.count(Z) || !Z->getSrcTy()->isIntegerTy() || Z->getSrcTy()->getIntegerBitWidth() < 32) continue;
    IRBuilder<> B(Z); Value *S = B.CreateSExt(Z->getOperand(0), Z->getDestTy());
    Z->replaceAllUsesWith(S); Z->eraseFromParent();
  }
  { unsigned Next = Streams.size(); for (auto &S : Streams) if (!S.Unit) S.StrideVA = Next++; ScratchVA = Next; TreeVA = Next + 1; }
  KA.recomputeUniformity();
  computeNeeded();
  for (auto &KV : GatherBase) ImplicitUsers.insert(KV.second);
  for (auto &KV : GatherIndex) ImplicitUsers.insert(KV.second);
  computeClasses();
  Segments.push_back({F.getName().str() + "_wt", -1});
  // the link register of consensual jumps: a skip jump is inserted in front of a block after the
  // block was emitted, so the register must not be one that a value dying in that block used
  LinkVS = alloc(RC::VS).Idx;
  SmallPtrSet<BasicBlock *, 32> All; for (BasicBlock &B : F) All.insert(&B);
  linearize(nullptr, All, &F.getEntryBlock());
  if (Order.size() < F.size()) return fail("could not linearize the CFG");
  computePositions();

  Out << "    vpset vp0\n";
  if (MDNode *MD = F.getMetadata("reqd_work_group_size"))
    GroupSize = mdconst::extract<ConstantInt>(MD->getOperand(0))->getZExtValue();
  VCap = Opts.MaxVRegs ? Opts.MaxVRegs : GroupSize ? (unsigned)std::min<uint64_t>(256, 2048 / GroupSize) : 256;
  if (Opts.MaxVPRegs) VPLimit = std::min(16u, std::max(4u, Opts.MaxVPRegs));
  if (Opts.MaxVSRegs && Opts.MaxVSRegs < 63) { VSUsed.resize(64, false); for (unsigned i = std::max(2u, Opts.MaxVSRegs); i < 64; i++) VSUsed[i] = true; }
  if (F.getInstructionCount() > 300) {   // a kernel that may run short of registers: reserve up front
    reserveSpillVS();
    Value *One = ConstantInt::get(Type::getInt64Ty(F.getContext()), 1);
    OneVS = allocInputVS().Idx; RegOf[One] = Reg{RC::VS, OneVS}; VSInputs.push_back({OneVS, One});
  }
  if (VCap < 256) {   // Spike: maxvl = 8 * (256 / (nvv + nvw)); the cap keeps a whole group in one stripmine
    LaneOffVV = alloc(RC::VV); HaveLaneOff = true;
    VSpillVS = allocInputVS().Idx; VStrideVS = allocInputVS().Idx; VSpillAddrVS = allocInputVS().Idx;
    emit("", "veidx", {LaneOffVV.str()});
    emit("", "vmul", {LaneOffVV.str(), LaneOffVV.str(), "vs" + std::to_string(VStrideVS)});
  }
  for (Instruction &I : instructions(F)) {
    if (!Needed.count(&I)) continue;
    if (isWorkItemId(&I)) {
      SpecialCalls.insert(&I);
      Reg Id = alloc(RC::VV);
      if (isGlobalId(&I)) {                       // global id = group offset + element index
        if (!VSOffset) VSOffset = allocInputVS().Idx;
        emit("", "veidx", {Id.str()});
        emit("", "vadd", {Id.str(), Id.str(), "vs" + std::to_string(VSOffset)});
      } else emit("", "veidx", {Id.str()});      // local id = element index
      RegOf[&I] = Id;
    } else if (isLocalSizeCall(&I)) {
      SpecialCalls.insert(&I);
      if (!VSLocalSize) VSLocalSize = allocInputVS().Idx;
      RegOf[&I] = Reg{RC::VS, VSLocalSize};
    } else if (isGroupIdCall(&I)) {
      SpecialCalls.insert(&I);
      if (!VSGroupId) VSGroupId = allocInputVS().Idx;
      RegOf[&I] = Reg{RC::VS, VSGroupId};
    } else if (isGlobalSizeCall(&I)) {
      SpecialCalls.insert(&I);
      if (!VSGlobalSize) VSGlobalSize = allocInputVS().Idx;
      RegOf[&I] = Reg{RC::VS, VSGlobalSize};
    }
  }
  // inputs (arguments, constants) get their vs registers before any temporary so they never collide
  for (auto &KV : GatherBase) if (isa<Argument>(KV.second) || isa<Constant>(KV.second)) (void)regOfValue(KV.second);
  for (auto &KV : GatherIndex) if (isa<Argument>(KV.second) || isa<Constant>(KV.second)) (void)regOfValue(KV.second);
  Pinning = true;
  for (Instruction &I : instructions(F)) {
    if (!Needed.count(&I) || SpecialCalls.count(&I)) continue;
    for (Value *Op : I.operands()) {
      if (isa<Function>(Op) || isa<BasicBlock>(Op) || isa<MetadataAsValue>(Op)) continue;
      if (auto *CB = dyn_cast<CallBase>(&I)) if (Op == CB->getCalledOperand()) continue;
      if (isa<Argument>(Op) || isa<Constant>(Op))
        (void)regOfValue(Op);
    }
  }
  Pinning = false;
  for (unsigned T : PendingTemps) VSUsed[T] = false; PendingTemps.clear();
  unsigned Pos = 0;
  for (size_t Idx = 0; Idx < Order.size(); Idx++) {
    Item &It = Order[Idx];
    bool Ok;
    if (It.K == Item::LoopBegin && CTLoopOf.count(It.L)) Ok = emitCTRegion(Idx, Pos);
    else Ok = It.K == Item::Block ? emitBlock(It.BB, Pos) : It.K == Item::LoopBegin ? beginLoop(It.L, Pos) : endLoop(It.L, Pos);
    if (!Ok) return false;
  }
  Out << "    vstop\n";
  // 32-bit registers are numbered after the 64-bit ones
  std::string Renamed; Renamed.reserve(Text.size());
  for (size_t i = 0; i < Text.size();) {
    if (Text.compare(i, 2, "vw") == 0 && i + 2 < Text.size() && isdigit((unsigned char)Text[i + 2])) {
      size_t j = i + 2; unsigned N = 0;
      while (j < Text.size() && isdigit((unsigned char)Text[j])) N = N * 10 + (Text[j++] - '0');
      Renamed += "vv" + std::to_string(NumVV + N); i = j;
    } else Renamed += Text[i++];
  }
  Text = Renamed;
  return true;
}

bool WTGen::emitInst(Instruction &I, unsigned Pos) {
  CurInst = &I; CurPos = Pos;
  if (getenv("HWCC_ANNOTATE")) { std::string S; raw_string_ostream OS(S); I.print(OS); Out << "    #" << Pos << ":" << S.substr(0, 110) << "\n"; }
  if (getenv("HWCC_TRACE")) { if (!TraceVS) TraceVS = allocInputVS().Idx; emit("", "vaddi", {"vs" + std::to_string(TraceVS), "vs0", std::to_string((int)(Pos % 4096) - 2048)}); }
  bool Ok = emitInstImpl(I, Pos);
  DefEnd[&I] = Text.size();
  return Ok;
}
bool WTGen::emitInstImpl(Instruction &I, unsigned Pos) {
  if (getenv("HWCC_DEBUG_VS")) { std::string S; raw_string_ostream OS(S); I.print(OS); DbgWhere = "inst " + std::to_string(Pos) + ":" + S.substr(0, 90); }
  CurOperands.clear(); for (Value *Op : I.operands()) CurOperands.insert(Op);
  if (auto It = GatherIndex.find(&I); It != GatherIndex.end()) { CurOperands.insert(It->second); CurOperands.insert(GatherBase[&I]); }
  auto R = [&](Value *V) { return regOfValue(V).str(); };
  // vector-register destinations and stores are masked by the block predicate; scalar (vs) ops are not
  const std::string &VP = CurPred;
  auto PV = [&](const std::string &Dst) -> std::string {
    if (!ArmPred.empty()) return ArmPred;
    return (Dst.rfind("vv", 0) == 0 || Dst.rfind("vw", 0) == 0 || Dst.rfind("vp", 0) == 0) ? VP : std::string(); };
  // Destination register: reuse the register of an operand that dies at this instruction (same
  // class); an instruction reads its sources before writing its destination, so this is safe for
  // single instructions, and the multi-instruction sequences below only ever read D after writing it.
  auto dest = [&](Instruction &I) {
    RC C = classOf(&I);
    ArmPred.clear();
    // An instruction whose only use is a select in the same block can be computed under the
    // select's condition straight into the select's register (saves a register and a move). Loads are
    // excluded: they emit under the block predicate, not ArmPred, so two loaded arms would both write
    // the select register unmasked and the second would clobber the first.
    if (I.hasOneUse() && !I.getType()->isIntegerTy(1) && !isa<LoadInst>(&I)) {
      if (auto *Sel = dyn_cast<SelectInst>(I.user_back())) {
        Value *Cond = Sel->getCondition();
        auto *CI = dyn_cast<Instruction>(Cond);
        bool CondReady = RegOf.count(Cond) && (!CI || CI->getParent() != I.getParent() || CI->comesBefore(&I));
        bool IsTrue = Sel->getTrueValue() == &I, IsFalse = Sel->getFalseValue() == &I;
        if (Sel->getParent() == I.getParent() && CondReady && (IsTrue != IsFalse) && classOf(Sel) == C) {
          Reg D;
          if (auto It = SelectReg.find(Sel); It != SelectReg.end()) D = It->second;
          else { D = alloc(C); SelectReg[Sel] = D; }
          RegOf[&I] = D; Alias.insert(&I);            // the register belongs to the select
          ArmPred = (IsTrue ? "" : "!") + regOfValue(Cond).str();
          return D.str();
        }
      }
    }
    if (auto It = PreferReg.find(&I); It != PreferReg.end() && It->second.Class == C &&
        PhiRealLast.lookup(PreferPhi[&I]) <= Pos) {   // == : this instruction reads the phi, then overwrites it
      Reg D = It->second; RegOf[&I] = D; Alias.insert(&I);   // register stays owned by the phi
      return D.str();
    }
    for (Value *Op : I.operands()) {
      auto It = RegOf.find(Op);
      if (It == RegOf.end() || It->second.Class != C || It->second.Idx == 0) continue;
      // a predicate select must not write into its own condition (the second move reads it)
      if (auto *Sel = dyn_cast<SelectInst>(&I)) if (Op == Sel->getCondition()) continue;
      if (isa<Argument>(Op) || isa<Constant>(Op) || Alias.count(Op)) continue;
      if (auto *OI = dyn_cast<Instruction>(Op)) if (SpecialCalls.count(OI)) continue;
      if (LastUse.lookup(Op) != Pos) continue;
      Reg D = It->second; RegOf[&I] = D; Transferred.insert(Op);
      return D.str();
    }
    Reg D = alloc(C); RegOf[&I] = D; return D.str();
  };
  Type *T = I.getType();

  if (auto *L = dyn_cast<LoadInst>(&I)) {
    AddrKind K = KindOf.lookup(L);
    bool IsFloat; std::string Suf = memSuffix(L->getType(), IsFloat);
    // an i1 load lands in a predicate register, but vlb into a vp register is illegal: load the byte into
    // a vector register and reduce it to the predicate (byte != 0).
    bool isPred = L->getType()->isIntegerTy(1);
    Reg pTmp{RC::VW, 0};
    std::string LD = dest(I);
    if (isPred) { pTmp = alloc(RC::VW); LD = pTmp.str(); Suf = "b"; }
    // a narrow integer load whose users are all zero-extensions is a zero-extending load
    else if (L->getType()->isIntegerTy() && L->getType()->getIntegerBitWidth() < 64 && !L->use_empty() &&
        all_of(L->users(), [](const User *U) { return isa<ZExtInst>(U); })) { Suf += "u"; UnsignedLoads.insert(L); }
    auto finish = [&]() {
      if (isPred) {
        std::string D = dest(I);   // one call: dest(I) allocates a fresh predicate each time it is invoked
        emit("", "vcmpeq", {D, LD, "vs0"});
        emit("", "vpop", {D, D, D, D, "0x55"});   // not: predicate = (byte != 0)
        VWUsed[pTmp.Idx] = false;
      }
      return true;
    };
    if (K == AddrKind::Stream) {
      Stream &St = Streams[StreamOfInst[L]];
      if (St.Unit) emit(VP, "vl" + Suf, {LD, "va" + std::to_string(St.VA)});
      else emit(VP, "vlst" + Suf, {LD, "va" + std::to_string(St.VA), "va" + std::to_string(St.StrideVA)});
      return finish();
    }
    if (K == AddrKind::Gather) {
      std::string Idx = R(GatherIndex[L]), Base = R(GatherBase[L]);
      // the base is uniform by construction, but a uniform value defined under divergent control flow
      // lives in a vector register (merged by predicates); vlx needs a vs base, so fold it into the index
      if (isVec(classOf(GatherBase[L]))) { Reg T = alloc(RC::VV); emit(VP, "vadd", {T.str(), Base, Idx}); emit(VP, "vlx" + Suf, {LD, "vs0", T.str()}); VVUsed[T.Idx] = false; return finish(); }
      emit(VP, "vlx" + Suf, {LD, Base, Idx}); return finish();
    }
    // uniform: scalar load from a vs address (or an indexed load off a zero base if the address
    // ended up in a vector register)
    std::string P = R(L->getPointerOperand());
    if (isVec(classOf(L->getPointerOperand()))) { emit(PV(LD), "vlx" + Suf, {LD, "vs0", P}); return finish(); }
    if (isPred) {   // a shared load cannot target a vector register: take the byte through a vs temp.
      // One temp per kernel, never freed: vs registers the control thread fills (vmcs, all sent before
      // the vf) are allocated lazily while the block is emitted, so a freed temp could be handed to a
      // later address and then be clobbered by this load at run time.
      VWUsed[pTmp.Idx] = false;
      if (!PredLoadVS) PredLoadVS = alloc(RC::VS).Idx;
      LD = "vs" + std::to_string(PredLoadVS);
      emit("", "vlsb", {LD, P}); return finish();
    }
    emit("", "vls" + Suf, {LD, P}); return finish();
  }
  if (auto *S = dyn_cast<StoreInst>(&I)) {
    AddrKind K = KindOf.lookup(S);
    Value *V = S->getValueOperand();
    bool IsFloat; std::string Suf = memSuffix(V->getType(), IsFloat);
    std::string Val = R(V);
    if (V->getType()->isIntegerTy(1)) {   // storing a predicate (e.g. a bufferized i1 comparison mask):
      Reg Tmp = alloc(RC::VW);            // materialize it as 0/1 bytes first -- vsb of a vp register is illegal
      Value *One = ConstantInt::get(Type::getInt64Ty(F.getContext()), 1);
      emit("", "vaddw", {Tmp.str(), "vs0", "vs0"});
      emit(R(V), "vaddw", {Tmp.str(), "vs0", R(One)});
      Val = Tmp.str(); VWUsed[Tmp.Idx] = false;
    }
    if (K == AddrKind::Stream) {
      Reg Bc{RC::VS, 0}; bool DidBc = false;
      if (classOf(V) == RC::VS) {          // vector store needs a vector source: broadcast
        Bc = alloc(isNarrow(V) ? RC::VW : RC::VV); DidBc = true;
        emit("", isNarrow(V) ? "vaddw" : "vadd", {Bc.str(), Val, "vs0"}); Val = Bc.str();
      }
      auto freeBc = [&]() { if (DidBc) (Bc.Class == RC::VV ? VVUsed : VWUsed)[Bc.Idx] = false; };
      Stream &St = Streams[StreamOfInst[S]];
      // The Chipyard Hwacha RTL deadlocks on a masked sub-word unit-stride store when most lanes
      // are inactive (a masked byte store with one active lane never completes). Work around it
      // by turning the store into load / select / unmasked store: every lane rewrites its own
      // byte, inactive lanes with the value they just read.
      if (St.Unit && !VP.empty() && DL.getTypeStoreSize(V->getType()) < 4 && Opts.SubwordRMW) {
        Reg Tmp = alloc(RC::VW);
        emit("", "vl" + Suf, {Tmp.str(), "va" + std::to_string(St.VA)});
        emit(VP, "vaddw", {Tmp.str(), Val, "vs0"});
        emit("", "vs" + Suf, {Tmp.str(), "va" + std::to_string(St.VA)});
        VWUsed[Tmp.Idx] = false; freeBc();
        return true;
      }
      if (St.Unit) emit(VP, "vs" + Suf, {Val, "va" + std::to_string(St.VA)});
      else emit(VP, "vsst" + Suf, {Val, "va" + std::to_string(St.VA), "va" + std::to_string(St.StrideVA)});
      freeBc();
      return true;
    }
    if (K == AddrKind::Gather) {
      if (classOf(V) == RC::VS) {          // scatter needs a vector source: broadcast
        Reg Tmp = alloc(isNarrow(V) ? RC::VW : RC::VV); emit("", isNarrow(V) ? "vaddw" : "vadd", {Tmp.str(), Val, "vs0"}); Val = Tmp.str();
        (Tmp.Class == RC::VV ? VVUsed : VWUsed)[Tmp.Idx] = false;
      }
      if (isVec(classOf(GatherBase[S]))) {   // vector base (see the load case): fold it into the index
        Reg T = alloc(RC::VV); emit(VP, "vadd", {T.str(), R(GatherBase[S]), R(GatherIndex[S])});
        emit(VP, "vsx" + Suf, {Val, "vs0", T.str()}); VVUsed[T.Idx] = false; return true;
      }
      emit(VP, "vsx" + Suf, {Val, R(GatherBase[S]), R(GatherIndex[S])}); return true;
    }
    if (isVec(classOf(S->getPointerOperand())) || isVec(classOf(V))) {
      std::string PR = R(S->getPointerOperand());
      if (!isVec(classOf(S->getPointerOperand()))) { Reg Tmp = alloc(RC::VV); emit("", "vadd", {Tmp.str(), PR, "vs0"}); PR = Tmp.str(); VVUsed[Tmp.Idx] = false; }
      if (!isVec(classOf(V))) { Reg Tmp = alloc(isNarrow(V) ? RC::VW : RC::VV); emit("", isNarrow(V) ? "vaddw" : "vadd", {Tmp.str(), Val, "vs0"}); Val = Tmp.str(); (Tmp.Class == RC::VV ? VVUsed : VWUsed)[Tmp.Idx] = false; }
      emit(VP, "vsx" + Suf, {Val, "vs0", PR}); return true;
    }
    emit("", "vss" + Suf, {R(S->getPointerOperand()), Val}); return true;   // scalar store: address first
  }
  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    std::string A = R(BO->getOperand(0)), B = R(BO->getOperand(1));
    std::string W = is32(T) ? "w" : "";
    std::string Op;
    switch (BO->getOpcode()) {
    case Instruction::Add:  Op = "vadd" + W; break;
    case Instruction::Sub:  Op = "vsub" + W; break;
    case Instruction::Mul:  Op = "vmul" + W; break;
    case Instruction::SDiv: Op = "vdiv" + W; break;
    case Instruction::UDiv: Op = "vdivu" + W; break;
    case Instruction::SRem: Op = "vrem" + W; break;
    case Instruction::URem: Op = "vremu" + W; break;
    case Instruction::Shl:  Op = "vsll" + W; break;
    case Instruction::LShr: Op = "vsrl" + W; break;
    case Instruction::AShr: Op = "vsra" + W; break;
    case Instruction::And:  Op = "vand"; break;
    case Instruction::Or:   Op = "vor"; break;
    case Instruction::Xor:  Op = "vxor"; break;
    case Instruction::FAdd: Op = "vfadd" + fpSuffix(T); break;
    case Instruction::FSub: Op = "vfsub" + fpSuffix(T); break;
    case Instruction::FMul: Op = "vfmul" + fpSuffix(T); break;
    case Instruction::FDiv: Op = "vfdiv" + fpSuffix(T); break;
    default: return fail("unsupported binary operator", &I);
    }
    if (T->isIntegerTy(1)) {  // predicate logic via vpop truth table over (a, b, c=a)
      unsigned Tab = BO->getOpcode() == Instruction::And ? 0x88 : BO->getOpcode() == Instruction::Or ? 0xEE : 0x66;
      emit("", "vpop", {dest(I), A, B, A, std::to_string(Tab)}); return true;
    }
    { std::string D = dest(I); emit(PV(D), Op, {D, A, B}); } return true;
  }
  if (auto *U = dyn_cast<UnaryOperator>(&I)) {
    if (U->getOpcode() == Instruction::FNeg) { std::string D = dest(I), S = R(U->getOperand(0)); emit(PV(D), "vfsgnjn" + fpSuffix(T), {D, S, S}); return true; }
    return fail("unsupported unary operator", &I);
  }
  if (auto *C = dyn_cast<CmpInst>(&I)) {
    Value *A = C->getOperand(0), *B = C->getOperand(1);
    bool Neg = false, Swap = false; std::string Op;
    if (auto *IC = dyn_cast<ICmpInst>(C)) {
      switch (IC->getPredicate()) {
      case CmpInst::ICMP_EQ:  Op = "vcmpeq"; break;
      case CmpInst::ICMP_NE:  Op = "vcmpeq"; Neg = true; break;
      case CmpInst::ICMP_SLT: Op = "vcmplt"; break;
      case CmpInst::ICMP_ULT: Op = "vcmpltu"; break;
      case CmpInst::ICMP_SGT: Op = "vcmplt"; Swap = true; break;
      case CmpInst::ICMP_UGT: Op = "vcmpltu"; Swap = true; break;
      case CmpInst::ICMP_SLE: Op = "vcmplt"; Swap = true; Neg = true; break;   // a<=b == !(b<a)
      case CmpInst::ICMP_ULE: Op = "vcmpltu"; Swap = true; Neg = true; break;
      case CmpInst::ICMP_SGE: Op = "vcmplt"; Neg = true; break;                // a>=b == !(a<b)
      case CmpInst::ICMP_UGE: Op = "vcmpltu"; Neg = true; break;
      default: return fail("unsupported icmp", &I);
      }
    } else {
      std::string S = fpSuffix(A->getType());
      switch (C->getPredicate()) {
      case CmpInst::FCMP_OEQ: Op = "vcmpfeq" + S; break;
      case CmpInst::FCMP_UNE: Op = "vcmpfeq" + S; Neg = true; break;
      case CmpInst::FCMP_OLT: Op = "vcmpflt" + S; break;
      case CmpInst::FCMP_OLE: Op = "vcmpfle" + S; break;
      case CmpInst::FCMP_OGT: Op = "vcmpflt" + S; Swap = true; break;
      case CmpInst::FCMP_OGE: Op = "vcmpfle" + S; Swap = true; break;
      case CmpInst::FCMP_UGE: Op = "vcmpflt" + S; Neg = true; break;          // !(a<b)
      case CmpInst::FCMP_UGT: Op = "vcmpfle" + S; Neg = true; break;          // !(a<=b)
      case CmpInst::FCMP_ULE: Op = "vcmpflt" + S; Swap = true; Neg = true; break;
      case CmpInst::FCMP_ULT: Op = "vcmpfle" + S; Swap = true; Neg = true; break;
      default: return fail("unsupported fcmp", &I);
      }
    }
    if (Swap) std::swap(A, B);
    std::string D = dest(I);
    std::string RA = R(A);
    Reg Bcast{RC::VS, 0}; bool DidBcast = false;
    if (classOf(A) == RC::VS && classOf(B) == RC::VS) {   // scalar compare would only write lane 0
      Bcast = alloc(isNarrow(A) ? RC::VW : RC::VV); DidBcast = true;
      emit("", isNarrow(A) ? "vaddw" : "vadd", {Bcast.str(), RA, "vs0"}); RA = Bcast.str();
    }
    emit("", Op, {D, RA, R(B)});
    if (DidBcast) (Bcast.Class == RC::VV ? VVUsed : VWUsed)[Bcast.Idx] = false;
    if (Neg) emit("", "vpop", {D, D, D, D, "0x55"});   // not
    return true;
  }
  if (auto *Sel = dyn_cast<SelectInst>(&I)) {
    std::string P = R(Sel->getCondition());
    std::string D;
    if (auto It = SelectReg.find(Sel); It != SelectReg.end()) { D = It->second.str(); RegOf[&I] = It->second; }
    else D = dest(I);
    std::string TR = R(Sel->getTrueValue()), FR = R(Sel->getFalseValue());
    if (T->isIntegerTy(1)) {                                 // predicate select: D = P ? T : F as one truth table
      emit("", "vpop", {D, P, TR, FR, "0xD8"});
      return true;
    }
    std::string Mv = isNarrow(&I) ? "vaddw" : "vadd";   // bit-preserving move for any payload
    if (TR != D) emit(P, Mv, {D, TR, "vs0"});
    if (FR != D) emit("!" + P, Mv, {D, FR, "vs0"});
    return true;
  }
  if (auto *CI = dyn_cast<CallInst>(&I)) {
    if (isBarrierCall(CI)) { UsesBarrier = true; emit("", "vfence", {}); return true; }
    if (StringRef Op; isWorkGroupReduce(CI, &Op)) {
      // lanes write their value (identity if masked off) to the scratch buffer; the control thread
      // reduces it after this segment and delivers the result in a vs register
      bool InRegion = inCTBody(&I);
      if (!InRegion && !LoopStack.empty()) return fail("work_group_reduce inside a vector-fetch loop (make the loop uniform or move the reduction out)", &I);
      Value *X = CI->getArgOperand(0);
      Constant *Id;
      if (T->isFloatingPointTy()) Id = Op == "add" ? ConstantFP::get(T, 0.0) : ConstantFP::getInfinity(T, /*Negative=*/Op == "max");
      else Id = Op == "add" ? ConstantInt::get(T, 0) : Op == "max" ? ConstantInt::get(T, APInt::getSignedMinValue(T->getIntegerBitWidth())) : ConstantInt::get(T, APInt::getSignedMaxValue(T->getIntegerBitWidth()));
      std::string Mv = isNarrow(X) ? "vaddw" : "vadd";
      Reg Tmp = alloc(isNarrow(X) ? RC::VW : RC::VV);
      emit("", Mv, {Tmp.str(), R(Id), "vs0"});
      emit(VP, Mv, {Tmp.str(), R(X), "vs0"});
      bool IsFloat; std::string Suf = memSuffix(T, IsFloat);
      emit("", "vs" + Suf, {Tmp.str(), "va" + std::to_string(ScratchVA)});
      (Tmp.Class == RC::VV ? VVUsed : VWUsed)[Tmp.Idx] = false;
      if (!ScratchVS) ScratchVS = allocInputVS().Idx;
      Reg Res = alloc(RC::VS); RegOf[&I] = Res;
      int RIdx = Reductions.size();
      std::string Label = F.getName().str() + "_wt_x" + std::to_string(RIdx);
      std::string Tree = F.getName().str() + "_wt_t" + std::to_string(RIdx);
      Reductions.push_back({Res.Idx, &I, Op.str(), T, Tree});
      // tree step: s[i] = op(s[i], s[i + m]) for i < n - m, issued by the control thread with vl = n - m
      {
        RC C = isNarrow(X) ? RC::VW : RC::VV;
        Reg A = alloc(C), Bv = alloc(C);
        std::string FS = fpSuffix(T);
        Out << "    vstop\n    .globl " << Tree << "\n" << Tree << ":\n";
        emit("", "vfence", {});
        emit("", "vl" + Suf, {A.str(), "va" + std::to_string(ScratchVA)});
        emit("", "vl" + Suf, {Bv.str(), "va" + std::to_string(TreeVA)});
        if (T->isFloatingPointTy()) emit("", (Op == "add" ? "vfadd" : Op == "max" ? "vfmax" : "vfmin") + FS, {A.str(), A.str(), Bv.str()});
        else if (Op == "add") emit("", Mv, {A.str(), A.str(), Bv.str()});
        else { Reg P = alloc(RC::VP); emit("", "vcmplt", {P.str(), Op == "max" ? A.str() : Bv.str(), Op == "max" ? Bv.str() : A.str()}); emit(P.str(), Mv, {A.str(), Bv.str(), "vs0"}); VPUsed[P.Idx] = false; }
        emit("", "vs" + Suf, {A.str(), "va" + std::to_string(ScratchVA)});
        Out << "    vstop\n";
        (C == RC::VV ? VVUsed : VWUsed)[A.Idx] = false; (C == RC::VV ? VVUsed : VWUsed)[Bv.Idx] = false;
      }
      Out << "    .globl " << Label << "\n" << Label << ":\n";
      emit("", "vfence", {});
      emit("", "vls" + Suf, {Res.str(), "vs" + std::to_string(ScratchVS)});   // slot 0 holds the result
      if (InRegion) CTLoops[CTLoopOf[outermost(I.getParent())]].BlockTail[I.getParent()].push_back({RIdx, Label});
      else Segments.push_back({Label, -1, RIdx});
      BlockSplit = true;
      return true;
    }
    if (Function *Callee = CI->getCalledFunction()) {
      StringRef N = Callee->getName();
      StringRef Body = N.starts_with("_Z") ? N.drop_front(2).drop_while([](char c) { return isdigit(c); }) : N;
      // OpenCL math builtins on float/double scalars: _Z4sqrtf, _Z4fabsf, _Z4fminff, _Z4fmaxff, native_/half_ variants
      if (T->isFloatingPointTy() && CI->arg_size() >= 1) {
        std::string S = fpSuffix(T);
        auto name1 = [&](StringRef Fn) { return Body.starts_with(Fn) && Body.drop_front(Fn.size()).size() <= 2; };
        if (name1("sqrt") || name1("native_sqrt") || name1("half_sqrt")) { std::string D = dest(I); emit(PV(D), "vfsqrt" + S, {D, R(CI->getArgOperand(0))}); return true; }
        if (name1("fabs")) { std::string D = dest(I), A = R(CI->getArgOperand(0)); emit(PV(D), "vfsgnjx" + S, {D, A, A}); return true; }
        if (name1("rsqrt") || name1("native_rsqrt") || name1("half_rsqrt")) {   // no rsqrt in the ISA: 1 / sqrt
          std::string D = dest(I), One = R(ConstantFP::get(T, 1.0));
          emit(PV(D), "vfsqrt" + S, {D, R(CI->getArgOperand(0))}); emit(PV(D), "vfdiv" + S, {D, One, D}); return true; }
        if (name1("fmin") && CI->arg_size() == 2) { std::string D = dest(I); emit(PV(D), "vfmin" + S, {D, R(CI->getArgOperand(0)), R(CI->getArgOperand(1))}); return true; }
        if (name1("fmax") && CI->arg_size() == 2) { std::string D = dest(I); emit(PV(D), "vfmax" + S, {D, R(CI->getArgOperand(0)), R(CI->getArgOperand(1))}); return true; }
      }
      // OpenCL 1.2 atomics: _Z10atomic_addPU8CLglobalVii etc. (atomic_/atom_ x add/sub/inc/dec/min/max/and/or/xor/xchg)
      static const std::pair<const char *, const char *> Table[] = {
        {"atomic_add", "vamoadd"}, {"atom_add", "vamoadd"}, {"atomic_sub", "vamoadd-"}, {"atom_sub", "vamoadd-"},
        {"atomic_inc", "vamoadd+1"}, {"atom_inc", "vamoadd+1"}, {"atomic_dec", "vamoadd-1"}, {"atom_dec", "vamoadd-1"},
        {"atomic_min", "vamomin"}, {"atom_min", "vamomin"}, {"atomic_max", "vamomax"}, {"atom_max", "vamomax"},
        {"atomic_and", "vamoand"}, {"atom_and", "vamoand"}, {"atomic_or", "vamoor"}, {"atom_or", "vamoor"},
        {"atomic_xor", "vamoxor"}, {"atom_xor", "vamoxor"}, {"atomic_xchg", "vamoswap"}, {"atom_xchg", "vamoswap"}};
      for (auto &[Name, Op] : Table) {
        if (!Body.starts_with(Name)) continue;
        std::string O = Op; bool Neg = false; Value *Src = nullptr;
        if (O.back() == '-') { O.pop_back(); Neg = true; Src = CI->getArgOperand(1); }
        else if (StringRef(O).ends_with("+1")) { O.resize(O.size() - 2); Src = ConstantInt::get(Type::getInt64Ty(F.getContext()), 1); }
        else if (StringRef(O).ends_with("-1")) { O.resize(O.size() - 2); Src = ConstantInt::get(Type::getInt64Ty(F.getContext()), -1); }
        else Src = CI->getArgOperand(1);
        bool Unsigned = Body.contains("PU8CLglobalVjj") || Body.contains("PU8CLglobalVmm");   // unsigned int / ulong
        if (Unsigned && (O == "vamomin" || O == "vamomax")) O += "u";
        O += T->isIntegerTy(64) ? ".d" : ".w";
        std::string SrcR = R(Src);
        if (Neg) { Reg Tmp = alloc(classOf(Src)); emit("", "vsub", {Tmp.str(), "vs0", SrcR}); SrcR = Tmp.str(); }
        std::string D = dest(I);
        emit(VP, O, {D, "0(" + R(CI->getArgOperand(0)) + ")", SrcR});
        return true;
      }
    }
    if (auto *II = dyn_cast<IntrinsicInst>(CI)) {
      std::string S = fpSuffix(T);
      switch (II->getIntrinsicID()) {
      case Intrinsic::fmuladd: case Intrinsic::fma:
        { std::string D = dest(I); emit(PV(D), "vfmadd" + S, {D, R(CI->getArgOperand(0)), R(CI->getArgOperand(1)), R(CI->getArgOperand(2))}); } return true;
      case Intrinsic::sqrt: { std::string D = dest(I); emit(PV(D), "vfsqrt" + S, {D, R(CI->getArgOperand(0))}); } return true;
      case Intrinsic::fabs: { std::string D = dest(I), A = R(CI->getArgOperand(0)); emit(PV(D), "vfsgnjx" + S, {D, A, A}); } return true;
      case Intrinsic::minnum: case Intrinsic::minimum: { std::string D = dest(I); emit(PV(D), "vfmin" + S, {D, R(CI->getArgOperand(0)), R(CI->getArgOperand(1))}); } return true;
      case Intrinsic::maxnum: case Intrinsic::maximum: { std::string D = dest(I); emit(PV(D), "vfmax" + S, {D, R(CI->getArgOperand(0)), R(CI->getArgOperand(1))}); } return true;
      case Intrinsic::smin: case Intrinsic::smax: case Intrinsic::umin: case Intrinsic::umax: {
        // no integer min/max in the worker-thread ISA: compare into a scratch predicate, then two predicated moves
        Value *A = CI->getArgOperand(0), *Bv = CI->getArgOperand(1);
        bool IsMin = II->getIntrinsicID() == Intrinsic::smin || II->getIntrinsicID() == Intrinsic::umin;
        bool Uns = II->getIntrinsicID() == Intrinsic::umin || II->getIntrinsicID() == Intrinsic::umax;
        std::string RA = R(A), RB = R(Bv);
        Reg P = alloc(RC::VP);
        Reg Bc{RC::VS, 0}; bool DidB = false;
        if (classOf(A) == RC::VS && classOf(Bv) == RC::VS) { Bc = alloc(isNarrow(A) ? RC::VW : RC::VV); DidB = true; emit("", isNarrow(A) ? "vaddw" : "vadd", {Bc.str(), RA, "vs0"}); RA = Bc.str(); }
        emit("", Uns ? "vcmpltu" : "vcmplt", {P.str(), RA, RB});          // P = a < b
        std::string D = dest(I); std::string Mv = isNarrow(&I) ? "vaddw" : "vadd";
        emit(P.str(), Mv, {D, IsMin ? RA : RB, "vs0"});
        emit("!" + P.str(), Mv, {D, IsMin ? RB : RA, "vs0"});
        VPUsed[P.Idx] = false; if (DidB) (Bc.Class == RC::VV ? VVUsed : VWUsed)[Bc.Idx] = false;
        return true;
      }
      case Intrinsic::abs: {   // |x| = x < 0 ? -x : x  (no integer abs in the worker-thread ISA)
        Value *A = CI->getArgOperand(0);
        std::string W = is32(T) ? "w" : "", RA = R(A);
        Reg Bc{RC::VS, 0}; bool DidB = false;
        if (classOf(A) == RC::VS) { Bc = alloc(isNarrow(A) ? RC::VW : RC::VV); DidB = true; emit("", isNarrow(A) ? "vaddw" : "vadd", {Bc.str(), RA, "vs0"}); RA = Bc.str(); }
        Reg Neg = alloc(isNarrow(&I) ? RC::VW : RC::VV);
        emit("", "vsub" + W, {Neg.str(), "vs0", RA});          // Neg = -x
        Reg P = alloc(RC::VP);
        emit("", "vcmplt", {P.str(), RA, "vs0"});               // P = x < 0
        std::string D = dest(I), Mv = isNarrow(&I) ? "vaddw" : "vadd";
        emit(P.str(), Mv, {D, Neg.str(), "vs0"});
        emit("!" + P.str(), Mv, {D, RA, "vs0"});
        VPUsed[P.Idx] = false; (Neg.Class == RC::VV ? VVUsed : VWUsed)[Neg.Idx] = false;
        if (DidB) (Bc.Class == RC::VV ? VVUsed : VWUsed)[Bc.Idx] = false;
        return true;
      }
      default: break;
      }
    }
    return fail("unsupported call", &I);
  }
  if (auto *Fr = dyn_cast<FreezeInst>(&I)) {   // freeze: no-op, share the operand's register
    aliasTo(I, Fr->getOperand(0));
    return true;
  }
  if (auto *Cast = dyn_cast<CastInst>(&I)) {
    Value *Src = Cast->getOperand(0);
    Type *ST = Src->getType();
    switch (Cast->getOpcode()) {
    case Instruction::Trunc:
      if (T->isIntegerTy(1)) {   // integer -> predicate: the low bit, i.e. (x & 1) != 0
        std::string D = dest(I), S = R(Src);
        Reg Tmp = alloc(isNarrow(Src) ? RC::VW : RC::VV);
        emit("", "vand", {Tmp.str(), S, R(ConstantInt::get(Type::getInt64Ty(F.getContext()), 1))});
        emit("", "vcmpeq", {D, Tmp.str(), "vs0"});
        emit("", "vpop", {D, D, D, D, "0x55"});   // not
        (Tmp.Class == RC::VV ? VVUsed : VWUsed)[Tmp.Idx] = false;
        return true;
      }
      [[fallthrough]];
    case Instruction::PtrToInt: case Instruction::IntToPtr:
    case Instruction::BitCast:
      // registers hold sign-extended 64-bit values; these are all no-ops
      aliasTo(I, Src);
      return true;
    case Instruction::ZExt: case Instruction::SExt: {
      unsigned Bits = ST->getIntegerBitWidth();
      if (Bits == 1) {   // predicate -> 0/1 (zext) or 0/-1 (sext): clear, then set under the predicate
        std::string D = dest(I), P = R(Src);
        Value *One = ConstantInt::get(Type::getInt64Ty(F.getContext()), Cast->getOpcode() == Instruction::ZExt ? 1 : -1);
        std::string Mv = isNarrow(&I) ? "vaddw" : "vadd";
        emit("", Mv, {D, "vs0", "vs0"});
        emit(P, Mv, {D, "vs0", R(One)});
        return true;
      }
      if (Cast->getOpcode() == Instruction::SExt) {
        // registers hold values sign-extended from 32 bits (w ops) or from the width of a narrow load;
        // anything narrower that is not a load (a trunc alias of a wider value, i8/i16 arithmetic done
        // in 32 bits) needs a real sign extension: lp_pool2d's `sext (trunc i64 to i2)`
        if (Bits >= 32 || isa<LoadInst>(Src)) { aliasTo(I, Src); return true; }
        std::string D = dest(I), S = R(Src);
        Value *K = ConstantInt::get(Type::getInt64Ty(F.getContext()), 64 - Bits);
        emit("", "vsll", {D, S, R(K)});
        emit("", "vsra", {D, D, R(K)});
        return true;
      }
      if (UnsignedLoads.count(Src)) {   // the load already zero-extended
        aliasTo(I, Src);
        return true;
      }
      std::string D = dest(I);
      // mask: value & ((1<<Bits)-1) via a shift pair with a vs constant
      Value *Sh = ConstantInt::get(Type::getInt64Ty(F.getContext()), 64 - Bits);
      std::string ShR = R(Sh);
      emit("", "vsll", {D, R(Src), ShR}); emit("", "vsrl", {D, D, ShR}); return true;
    }
    case Instruction::SIToFP: case Instruction::UIToFP:
      if (ST->isIntegerTy(1)) {   // predicate -> 0.0 / 1.0 (or -1.0 for sitofp): vfcvt cannot read a vp register
        std::string D = dest(I), P = R(Src);
        emit("", isNarrow(&I) ? "vaddw" : "vadd", {D, "vs0", "vs0"});   // all-zero bits = +0.0
        emit(P, "vfadd" + fpSuffix(T), {D, D, R(ConstantFP::get(T, Cast->getOpcode() == Instruction::UIToFP ? 1.0 : -1.0))});
        return true;
      }
      if (Cast->getOpcode() == Instruction::SIToFP) emit("", "vfcvt" + fpSuffix(T) + (is32(ST) ? ".w" : ".l"), {dest(I), R(Src)});
      else emit("", "vfcvt" + fpSuffix(T) + (is32(ST) ? ".wu" : ".lu"), {dest(I), R(Src)});
      return true;
    // C float->int conversion truncates: encode rtz explicitly (the assembler default is dyn = frm, normally RNE)
    case Instruction::FPToSI: emit("", "vfcvt" + std::string(is32(T) ? ".w" : ".l") + fpSuffix(ST), {dest(I), R(Src), "rtz"}); return true;
    case Instruction::FPToUI: emit("", "vfcvt" + std::string(is32(T) ? ".wu" : ".lu") + fpSuffix(ST), {dest(I), R(Src), "rtz"}); return true;
    case Instruction::FPExt: case Instruction::FPTrunc: emit("", "vfcvt" + fpSuffix(T) + fpSuffix(ST), {dest(I), R(Src)}); return true;
    default: return fail("unsupported cast", &I);
    }
  }
  if (auto *G = dyn_cast<GetElementPtrInst>(&I)) {
    // pointer arithmetic on registers: base + sum(idx * scale)
    std::string Cur = R(G->getPointerOperand());
    std::string D = dest(I);
    bool First = true;
    for (gep_type_iterator GTI = gep_type_begin(G), E = gep_type_end(G); GTI != E; ++GTI) {
      Value *Idx = GTI.getOperand();
      uint64_t Scale = GTI.isStruct() ? DL.getStructLayout(GTI.getStructType())->getElementOffset(cast<ConstantInt>(Idx)->getZExtValue())
                                      : GTI.getSequentialElementStride(DL).getFixedValue();
      if (auto *CI = dyn_cast<ConstantInt>(Idx)) {
        int64_t Off = GTI.isStruct() ? (int64_t)Scale : CI->getSExtValue() * (int64_t)Scale;
        if (!Off) continue;
        Value *OffC = ConstantInt::get(Type::getInt64Ty(F.getContext()), Off);
        emit("", "vadd", {D, Cur, R(OffC)}); Cur = D; First = false; continue;
      }
      Reg Tmp = alloc(isVec(classOf(Idx)) || isVec(classOf(&I)) ? RC::VV : RC::VS);
      if (Scale == 1) emit("", "vadd", {D, Cur, R(Idx)});
      else {
        Value *SC = ConstantInt::get(Type::getInt64Ty(F.getContext()), Scale);
        emit("", "vmul", {Tmp.str(), R(Idx), R(SC)});
        emit("", "vadd", {D, Cur, Tmp.str()});
      }
      (Tmp.Class == RC::VV ? VVUsed : VSUsed)[Tmp.Idx] = false;
      Cur = D; First = false;
    }
    if (First) emit("", "vadd", {D, Cur, "vs0"});
    return true;
  }
  return fail("unsupported instruction", &I);
}

// ---------------------------------------------------------------- control-thread codegen (IR)

// Clone a uniform expression (args/constants/instructions over them) from the kernel into the
// control-thread function, at the builder's insertion point.
static Value *cloneUniform(Value *V, ValueToValueMapTy &VMap, IRBuilder<> &B, raw_ostream &Err) {
  if (auto It = VMap.find(V); It != VMap.end()) return It->second;
  if (auto *GV = dyn_cast<GlobalVariable>(V)) {
    // __local buffers: one static buffer of the same shape in the control-thread module.
    // Work-groups run one after another, so a single buffer per kernel is enough.
    Module &CT = *B.GetInsertBlock()->getModule();
    // A declaration (e.g. hwacha_grid_size, set by the host launch) becomes a weak zero definition.
    // A constant with data (weights baked into the kernel by MLIR) is copied as is.
    GlobalVariable *C = CT.getGlobalVariable(GV->getName(), true);
    if (!C) {
      bool Data = GV->hasInitializer() && !isa<UndefValue>(GV->getInitializer());
      if (GV->isDeclaration() && GV->isConstant()) {
        // a weight stripped to the .incbin blob (external constant): keep it an external declaration so
        // it resolves against that blob at link time, rather than reserving a zero copy in .bss here.
        C = new GlobalVariable(CT, GV->getValueType(), true, GlobalValue::ExternalLinkage, nullptr, GV->getName());
      } else
        C = new GlobalVariable(CT, GV->getValueType(), Data && GV->isConstant(), GV->isDeclaration() ? GlobalValue::WeakAnyLinkage : GlobalValue::InternalLinkage,
                               Data ? GV->getInitializer() : Constant::getNullValue(GV->getValueType()), GV->getName());
      C->setAlignment(GV->getAlign().value_or(Align(8)));
    }
    VMap[V] = C; return C;
  }
  if (auto *CE = dyn_cast<ConstantExpr>(V)) {
    Instruction *I = CE->getAsInstruction();
    for (unsigned i = 0; i < I->getNumOperands(); i++) {
      Value *Op = cloneUniform(I->getOperand(i), VMap, B, Err);
      if (!Op) { I->deleteValue(); return nullptr; }
      I->setOperand(i, Op);
    }
    B.Insert(I); VMap[V] = I; return I;
  }
  if (isa<Constant>(V)) return V;
  auto *I = dyn_cast<Instruction>(V);
  if (!I || isa<LoadInst>(I) || isa<CallBase>(I) || isa<PHINode>(I)) { Err << "hwacha-cc: control-thread expression depends on " << *V << "\n"; return nullptr; }
  Instruction *C = I->clone();
  for (unsigned i = 0; i < C->getNumOperands(); i++) {
    Value *Op = cloneUniform(C->getOperand(i), VMap, B, Err);
    if (!Op) return nullptr;
    C->setOperand(i, Op);
  }
  B.Insert(C);
  VMap[V] = C;
  return C;
}

static void asmCall(IRBuilder<> &B, StringRef Asm, StringRef Cons, ArrayRef<Value *> Args) {
  SmallVector<Type *, 2> Tys; for (Value *A : Args) Tys.push_back(A->getType());
  FunctionType *FT = FunctionType::get(B.getVoidTy(), Tys, false);
  B.CreateCall(InlineAsm::get(FT, Asm, Cons, /*sideeffect=*/true), Args);
}

// Clone a uniform kernel value into the control thread. Instructions inside control-thread loop L
// go to the loop's block (Inside), everything else before the loop (Outside).
struct CTCloner {
  Function &K; WTGen &WT; ValueToValueMapTy &VMap; raw_ostream &Err; LoopInfo &LI;
  IRBuilder<> *Outside = nullptr; Loop *L = nullptr;
  DenseMap<const BasicBlock *, BasicBlock *> *CtBlock = nullptr;   // region block -> its control-thread clone
  Value *VL = nullptr, *Grp = nullptr; GlobalVariable *Scratch = nullptr;
  Value *clone(Value *V) {
    if (auto It = VMap.find(V); It != VMap.end()) return It->second;
    IRBuilder<> &B = *Outside;
    auto *I = dyn_cast<Instruction>(V);
    if (!I) return cloneUniform(V, VMap, B, Err);
    // instructions of the region go to their own block's clone (before its terminator, if any)
    bool InRegion = L && L->contains(I) && CtBlock;
    if (isLocalSizeCall(I)) { VMap[V] = VL; return VL; }
    if (isGroupIdCall(I)) { VMap[V] = Grp; return Grp; }
    if (auto *Phi = dyn_cast<PHINode>(I)) {
      if (Phi->getNumIncomingValues() == 1) { Value *C = clone(Phi->getIncomingValue(0)); VMap[V] = C; return C; }   // LCSSA
      Err << "hwacha-cc: control thread cannot evaluate " << *V << "\n"; return nullptr;
    }
    if (isWorkGroupReduce(I)) {   // result sits in scratch slot 0 after the tree: wait for the vector unit, read it
      IRBuilder<> &RB = B;
      asmCall(RB, "fence", "", {});
      Value *R = RB.CreateLoad(I->getType(), Scratch);
      VMap[V] = R; return R;
    }
    if (isa<CallBase>(I)) { Err << "hwacha-cc: control thread cannot evaluate " << *V << "\n"; return nullptr; }
    Instruction *C = I->clone();
    for (unsigned i = 0; i < C->getNumOperands(); i++) {
      Value *Op = clone(C->getOperand(i));
      if (!Op) { C->deleteValue(); return nullptr; }
      C->setOperand(i, Op);
    }
    C->dropUnknownNonDebugMetadata();
    // (LLVM 23: getTerminator() on a block under construction returns the last instruction, so test explicitly)
    if (InRegion) { BasicBlock *CB = (*CtBlock)[I->getParent()]; if (!CB->empty() && CB->back().isTerminator()) C->insertBefore(CB->back().getIterator()); else C->insertInto(CB, CB->end()); }
    else B.Insert(C);
    VMap[V] = C;
    return C;
  }
};

static Value *toI64Payload(IRBuilder<> &B, Value *CV, raw_ostream &Err) {
  Type *T = CV->getType(); Type *I64 = B.getInt64Ty();
  if (T->isFloatTy()) return B.CreateZExt(B.CreateBitCast(CV, B.getInt32Ty()), I64);
  if (T->isDoubleTy()) return B.CreateBitCast(CV, I64);
  if (T->isPointerTy()) return B.CreatePtrToInt(CV, I64);
  if (T->isIntegerTy()) return B.CreateSExtOrTrunc(CV, I64);
  if (T->isHalfTy()) return B.CreateZExt(B.CreateBitCast(CV, B.getInt16Ty()), I64);
  Err << "hwacha-cc: unsupported uniform input type\n"; return nullptr;
}

static bool generateCT(Function &K, WTGen &WT, Module &CT, raw_ostream &Err) {
  LLVMContext &Ctx = CT.getContext();
  Type *I64 = Type::getInt64Ty(Ctx);
  SmallVector<Type *, 8> Params{I64};
  for (Argument &A : K.args()) Params.push_back(A.getType());
  Function *F = Function::Create(FunctionType::get(Type::getVoidTy(Ctx), Params, false),
                                 GlobalValue::ExternalLinkage, K.getName() + "_ct", CT);
  F->setDSOLocal(true);
  ValueToValueMapTy VMap;
  auto AI = F->arg_begin(); AI->setName("n"); Value *N = &*AI++;
  for (Argument &A : K.args()) { AI->setName(A.getName()); VMap[&A] = &*AI++; }
  auto segSym = [&](const std::string &Name) {
    GlobalVariable *G = CT.getGlobalVariable(Name, true);
    if (!G) G = new GlobalVariable(CT, Type::getInt8Ty(Ctx), true, GlobalValue::ExternalLinkage, nullptr, Name);
    return G;
  };
  DominatorTree KDT(K); LoopInfo KLI(KDT);

  BasicBlock *Entry = BasicBlock::Create(Ctx, "entry", F);
  BasicBlock *Loop = BasicBlock::Create(Ctx, "stripmine", F);
  BasicBlock *Exit = BasicBlock::Create(Ctx, "done", F);
  IRBuilder<> B(Entry);

  // vsetcfg: VCFG(nvvd, nvvw, nvvh, nvp) | VRU bit
  uint64_t Cfg = (WT.NumVV & 0x1ff) | ((uint64_t)(WT.NumVP & 0x1f) << 9) | ((uint64_t)(WT.NumVW & 0x1ff) << 14) | (1ull << 63);
  asmCall(B, "vsetcfg $0", "r", {ConstantInt::get(I64, Cfg)});
  // uniform inputs -> vs registers, as 64-bit integer payloads
  if (WT.PoolVS) {   // the uniform pool: i64 payloads in the same format the vs registers hold
    SmallVector<Constant *, 64> Payloads; SmallVector<std::pair<unsigned, Value *>, 8> ArgSlots;
    for (Value *V : WT.PoolConsts) {
      if (auto *CF = dyn_cast<ConstantFP>(V)) Payloads.push_back(ConstantInt::get(I64, CF->getValueAPF().bitcastToAPInt().getZExtValue()));
      else if (auto *CI = dyn_cast<ConstantInt>(V)) Payloads.push_back(ConstantInt::get(I64, CI->getValue().sextOrTrunc(64).getZExtValue()));
      else { ArgSlots.push_back({(unsigned)Payloads.size(), V}); Payloads.push_back(ConstantInt::get(I64, 0)); }
    }
    auto *AT = ArrayType::get(I64, Payloads.size());
    auto *Pool = new GlobalVariable(CT, AT, ArgSlots.empty(), GlobalValue::PrivateLinkage, ConstantArray::get(AT, Payloads), K.getName() + "_cpool");
    Pool->setAlignment(Align(8));
    for (auto &[Idx, V] : ArgSlots) {   // arguments are stored at launch
      Value *CV = cloneUniform(V, VMap, B, Err);
      if (!CV) return false;
      CV = toI64Payload(B, CV, Err);
      if (!CV) return false;
      B.CreateStore(CV, B.CreateConstInBoundsGEP2_64(AT, Pool, 0, Idx));
    }
    asmCall(B, "vmcs vs" + std::to_string(WT.PoolVS) + ", $0", "r", {B.CreatePtrToInt(Pool, I64)});
  }
  if (WT.HaveLaneOff) {   // per-lane vector spill slots: lane * stride + 8 * slot
    unsigned NReg = WT.NumVV + WT.NumVW;
    uint64_t Lanes = NReg < 2 ? 2048 : 8 * (256 / NReg), Stride = 8 * std::max(1u, WT.NumVSlots);
    auto *AT = ArrayType::get(Type::getInt8Ty(Ctx), Lanes * Stride);
    auto *Sp = new GlobalVariable(CT, AT, false, GlobalValue::PrivateLinkage, ConstantAggregateZero::get(AT), K.getName() + "_vspill");
    Sp->setAlignment(Align(8));
    asmCall(B, "vmcs vs" + std::to_string(WT.VSpillVS) + ", $0", "r", {B.CreatePtrToInt(Sp, I64)});
    asmCall(B, "vmcs vs" + std::to_string(WT.VStrideVS) + ", $0", "r", {ConstantInt::get(I64, Stride)});
  }
  if (WT.SpillVS) {   // the spill area for uniform temporaries (one i64 slot per spilled value)
    auto *AT = ArrayType::get(I64, std::max(1u, WT.NumSpillSlots));
    auto *Sp = new GlobalVariable(CT, AT, false, GlobalValue::PrivateLinkage, ConstantAggregateZero::get(AT), K.getName() + "_spill");
    Sp->setAlignment(Align(8));
    asmCall(B, "vmcs vs" + std::to_string(WT.SpillVS) + ", $0", "r", {B.CreatePtrToInt(Sp, I64)});
  }
  for (auto &[VS, V] : WT.VSInputs) {
    Value *CV = cloneUniform(V, VMap, B, Err);
    if (!CV) return false;
    CV = toI64Payload(B, CV, Err);
    if (!CV) return false;
    asmCall(B, "vmcs vs" + std::to_string(VS) + ", $0", "r", {CV});
  }
  // stream bases that are fixed for the whole kernel, and stride registers
  SmallVector<Value *, 8> Bases;
  CTCloner EntryCloner{K, WT, VMap, Err, KLI}; EntryCloner.Outside = &B;
  for (auto &S : WT.Streams) {
    if (S.PerIter >= 0) { Bases.push_back(nullptr); continue; }
    Value *Bv = EntryCloner.clone(S.Base); if (!Bv) return false; Bases.push_back(Bv);
  }
  SmallVector<Value *, 8> StrideVals;
  for (auto &S : WT.Streams) {
    Value *SV = S.StrideV ? EntryCloner.clone(S.StrideV) : ConstantInt::get(I64, S.Stride);
    if (!SV) return false;
    StrideVals.push_back(SV);
    if (!S.Unit) asmCall(B, "vmca va" + std::to_string(S.StrideVA) + ", $0", "r", {SV});
  }
  if (!WT.Reductions.empty()) {   // scratch buffer for cross-lane reductions: one 64-bit slot per lane
    ArrayType *AT = ArrayType::get(I64, 2048);
    auto *Scratch = new GlobalVariable(CT, AT, false, GlobalValue::InternalLinkage, Constant::getNullValue(AT), K.getName() + "_scratch");
    Scratch->setAlignment(Align(64));
    asmCall(B, "vmca va" + std::to_string(WT.ScratchVA) + ", $0", "r", {Scratch});
    asmCall(B, "vmcs vs" + std::to_string(WT.ScratchVS) + ", $0", "r", {B.CreatePtrToInt(Scratch, I64)});
  }
  B.CreateCondBr(B.CreateICmpNE(N, ConstantInt::get(I64, 0)), Loop, Exit);

  B.SetInsertPoint(Loop);
  PHINode *Rem = B.CreatePHI(I64, 2, "remaining");
  PHINode *Off = B.CreatePHI(I64, 2, "offset");
  SmallVector<PHINode *, 8> BasePhis;
  for (Value *Bv : Bases) BasePhis.push_back(Bv ? B.CreatePHI(Bv->getType(), 2, "base") : nullptr);
  // vsetvl
  FunctionType *VLT = FunctionType::get(I64, {I64}, false);
  PHINode *Grp = B.CreatePHI(I64, 2, "group");
  // The work-group size caps the vector length so every group is exactly that size:
  // reqd_work_group_size if the kernel has one, else the host-settable global
  // `hwacha_group_size` (0 = let the hardware choose, i.e. one group per stripmine).
  Value *Req = Rem;
  Value *G;
  if (WT.GroupSize) G = ConstantInt::get(I64, WT.GroupSize);
  else {
    GlobalVariable *GS = CT.getGlobalVariable("hwacha_group_size", true);
    if (!GS) { GS = new GlobalVariable(CT, I64, false, GlobalValue::WeakAnyLinkage, ConstantInt::get(I64, 0), "hwacha_group_size"); GS->setAlignment(Align(8)); }
    G = B.CreateLoad(I64, GS, "gs");
    G = B.CreateSelect(B.CreateICmpEQ(G, ConstantInt::get(I64, 0)), Rem, G);
  }
  Req = B.CreateSelect(B.CreateICmpULT(Rem, G), Rem, G);
  Value *VL = B.CreateCall(InlineAsm::get(VLT, "vsetvl $0, $1", "=r,r", true), {Req}, "vl");
  // If the hardware cannot provide the requested work-group size (too many registers declared
  // for this kernel), flag it: kernels that rely on get_local_size() would silently miscompute.
  {
    GlobalVariable *Short = CT.getGlobalVariable("hwacha_vl_short", true);
    if (!Short) { Short = new GlobalVariable(CT, I64, false, GlobalValue::WeakAnyLinkage, ConstantInt::get(I64, 0), "hwacha_vl_short"); Short->setAlignment(Align(8)); }
    Value *Old = B.CreateLoad(I64, Short);
    Value *IsShort = B.CreateZExt(B.CreateICmpULT(VL, Req), I64);
    B.CreateStore(B.CreateOr(Old, IsShort), Short);
  }
  for (auto &S : WT.Streams) if (S.PerIter < 0) asmCall(B, "vmca va" + std::to_string(S.VA) + ", $0", "r", {BasePhis[S.VA]});
  if (WT.VSOffset) asmCall(B, "vmcs vs" + std::to_string(WT.VSOffset) + ", $0", "r", {Off});
  if (WT.VSLocalSize) asmCall(B, "vmcs vs" + std::to_string(WT.VSLocalSize) + ", $0", "r", {VL});
  if (WT.VSGroupId) asmCall(B, "vmcs vs" + std::to_string(WT.VSGroupId) + ", $0", "r", {Grp});
  if (WT.VSGlobalSize) asmCall(B, "vmcs vs" + std::to_string(WT.VSGlobalSize) + ", $0", "r", {N});

  // ---- vf segments; control-thread loops between them
  // Per-group values the kernel's uniform slice may reference are cloned fresh for every group
  // (they can depend on the offset / vl), so start from the entry-level map each time.
  ValueToValueMapTy GMap; for (auto KV : VMap) GMap[KV.first] = KV.second;
  CTCloner Cl{K, WT, GMap, Err, KLI}; Cl.VL = VL; Cl.Grp = Grp; Cl.Outside = &B;
  Cl.Scratch = WT.Reductions.empty() ? nullptr : CT.getGlobalVariable((K.getName() + "_scratch").str(), true);
  DenseMap<BasicBlock *, Value *> Reached;
  std::function<Value *(BasicBlock *)> reached = [&](BasicBlock *BB) -> Value * {
    if (auto It = Reached.find(BB); It != Reached.end()) return It->second;
    Value *R = nullptr;
    if (BB == &K.getEntryBlock()) R = B.getTrue();
    else {
      R = B.getFalse();
      for (BasicBlock *P : predecessors(BB)) {
        llvm::Loop *PL = KLI.getLoopFor(P);
        Value *C;
        if (PL && !PL->contains(BB)) { while (PL->getParentLoop()) PL = PL->getParentLoop(); C = reached(PL->getLoopPreheader()); }
        else {
          C = reached(P);
          auto *Br = cast<BranchInst>(P->getTerminator());
          if (Br->isConditional() && WT.isUniformValue(Br->getCondition())) {   // divergent: maybe taken
            Value *Cond = Cl.clone(Br->getCondition());
            if (!Cond) return nullptr;
            if (Br->getSuccessor(0) != BB) Cond = B.CreateNot(Cond);
            C = B.CreateAnd(C, Cond);
          }
        }
        if (!C) return nullptr;
        R = B.CreateOr(R, C);
      }
    }
    Reached[BB] = R; return R;
  };
  // A cross-lane reduction: wait for the lanes' stores, reduce the vl scratch slots in scalar code,
  // send the result to its vs register; the control thread keeps the value too (GMap).
  GlobalVariable *ScratchGV = WT.Reductions.empty() ? nullptr : CT.getGlobalVariable((K.getName() + "_scratch").str(), true);
  auto emitReduce = [&](int RIdx) -> bool {
    const WTGen::Reduction &R = WT.Reductions[RIdx];
    uint64_t ESz = K.getParent()->getDataLayout().getTypeStoreSize(R.Ty);
    FunctionType *VLT = FunctionType::get(I64, {I64}, false);
    BasicBlock *Pre = B.GetInsertBlock();
    BasicBlock *LoopB = BasicBlock::Create(Ctx, "tree_loop", F), *Done = BasicBlock::Create(Ctx, "tree_done", F);
    B.CreateCondBr(B.CreateICmpUGT(VL, ConstantInt::get(I64, 1)), LoopB, Done);
    B.SetInsertPoint(LoopB);
    PHINode *N = B.CreatePHI(I64, 2, "tn");
    Value *M = B.CreateLShr(B.CreateAdd(N, ConstantInt::get(I64, 1)), ConstantInt::get(I64, 1));   // m = ceil(n/2)
    B.CreateCall(InlineAsm::get(VLT, "vsetvl $0, $1", "=r,r", true), {B.CreateSub(N, M)});
    asmCall(B, "vmca va" + std::to_string(WT.TreeVA) + ", $0", "r", {B.CreatePtrAdd(ScratchGV, B.CreateMul(M, ConstantInt::get(I64, ESz)))});
    asmCall(B, "vf 0($0)", "r", {segSym(R.TreeLabel)});
    N->addIncoming(VL, Pre); N->addIncoming(M, LoopB);
    B.CreateCondBr(B.CreateICmpUGT(M, ConstantInt::get(I64, 1)), LoopB, Done);
    B.SetInsertPoint(Done);
    B.CreateCall(InlineAsm::get(VLT, "vsetvl $0, $1", "=r,r", true), {VL});   // back to the group's vector length
    return true;
  };
  // A control-thread region: the loop nest's CFG mirrored in scalar code, one vf per kernel block
  auto emitRegion = [&](const WTGen::CTLoop &C, int RIdx) -> bool {
    llvm::Loop *L = C.L; BasicBlock *H = L->getHeader(), *Pre = L->getLoopPreheader();
    BasicBlock *Exiting = L->getExitingBlock(), *Exit = L->getUniqueExitBlock();
    Value *Reach = reached(Pre);
    if (!Reach) return false;
    BasicBlock *Check = B.GetInsertBlock();
    bool MayBeSkipped = !(isa<ConstantInt>(Reach) && cast<ConstantInt>(Reach)->isOne());
    BasicBlock *After = BasicBlock::Create(Ctx, "ct_after", F);
    // CtBlock: entry clone of each kernel block (branch targets); CtEnd: where its code currently
    // ends (a reduction splices its scalar loop in, so a block can become several)
    DenseMap<const BasicBlock *, BasicBlock *> CtBlock, CtEnd;
    for (BasicBlock *BB : L->blocks()) CtEnd[BB] = CtBlock[BB] = BasicBlock::Create(Ctx, "ct_" + BB->getName(), F);
    Cl.L = L; Cl.CtBlock = &CtEnd;
    // uniform phis -> control-thread phis (incomings filled below)
    SmallVector<std::pair<PHINode *, PHINode *>, 8> Phis;
    for (BasicBlock *BB : L->blocks())
      for (PHINode &Phi : BB->phis()) {
        if (!WT.isUniformValue(&Phi)) continue;
        PHINode *P = PHINode::Create(Phi.getType(), Phi.getNumIncomingValues(), Phi.getName());
        P->insertInto(CtBlock[BB], CtBlock[BB]->begin());
        GMap[&Phi] = P; Phis.push_back({&Phi, P});
      }
    // edge blocks (phi moves the worker could not merge into the source block)
    DenseMap<std::pair<const BasicBlock *, const BasicBlock *>, BasicBlock *> EdgeBlk;
    auto edgeTarget = [&](BasicBlock *P, BasicBlock *S) -> BasicBlock * {
      BasicBlock *T = L->contains(S) ? CtBlock[S] : After;
      auto It = C.EdgeSeg.find({P, S});
      if (It == C.EdgeSeg.end()) return T;
      BasicBlock *&E = EdgeBlk[{P, S}];
      if (!E) {
        E = BasicBlock::Create(Ctx, "ct_edge", F);
        IRBuilder<> EB(E);
        asmCall(EB, "vf 0($0)", "r", {segSym(It->second)});
        EB.CreateBr(T);
      }
      return E;
    };
    // the block an edge P->S arrives from on the control thread: the edge block if there is one
    auto edgeSource = [&](BasicBlock *P, BasicBlock *S) -> BasicBlock * {
      edgeTarget(P, S);
      auto It = EdgeBlk.find({P, S});
      return It != EdgeBlk.end() ? It->second : CtEnd[P];
    };
    // per block: iteration inputs defined here, stream bases used here, then the block's vf.
    // A reduction splits the block: inputs defined after it are sent after the control thread has
    // reduced (they may depend on the result), before the next segment.
    for (BasicBlock *BB : L->blocks()) {
      auto TailIt = C.BlockTail.find(BB);
      std::vector<std::pair<int, std::string>> Tails; if (TailIt != C.BlockTail.end()) Tails = TailIt->second;
      auto bucketOf = [&](const Value *V) -> unsigned {   // number of reductions of BB preceding V
        auto *VI = dyn_cast<Instruction>(V);
        if (!VI || VI->getParent() != BB) return 0;
        unsigned n = 0;
        for (auto &[RI, Lbl] : Tails) if (cast<Instruction>(WT.Reductions[RI].V)->comesBefore(VI)) n++;
        return n;
      };
      auto emitBucket = [&](unsigned Bk) -> bool {
        for (auto &[VS, V] : C.IterInputs) {
          if (cast<Instruction>(V)->getParent() != BB || bucketOf(V) != Bk) continue;
          Value *CV = Cl.clone(V); if (!CV) return false;
          IRBuilder<> IB(CtEnd[BB]);
          CV = toI64Payload(IB, CV, Err); if (!CV) return false;
          asmCall(IB, "vmcs vs" + std::to_string(VS) + ", $0", "r", {CV});
        }
        for (auto &S : WT.Streams) {
          if (S.PerIter != RIdx || !S.Blocks.count(BB) || bucketOf(S.Base) != Bk) continue;
          Value *Bv = Cl.clone(S.Base); if (!Bv) return false;
          IRBuilder<> IB(CtEnd[BB]);
          // this group's lanes: streams indexed by the local id are already relative to the group
          Value *Adv = S.Local ? Bv : IB.CreatePtrAdd(Bv, IB.CreateMul(Off, StrideVals[S.VA]));
          asmCall(IB, "vmca va" + std::to_string(S.VA) + ", $0", "r", {Adv});
        }
        return true;
      };
      if (!emitBucket(0)) return false;
      { IRBuilder<> IB(CtEnd[BB]); asmCall(IB, "vf 0($0)", "r", {segSym(C.BlockSeg.lookup(BB))}); }
      for (unsigned k = 0; k < Tails.size(); k++) {
        // the reduction's scalar loop lives in its own blocks; splice them into this block's clone
        auto Saved = B.saveIP();
        B.SetInsertPoint(CtEnd[BB]);
        if (!emitReduce(Tails[k].first)) return false;
        CtEnd[BB] = B.GetInsertBlock();   // the block continues after the reduction loop
        B.restoreIP(Saved);
        if (!emitBucket(k + 1)) return false;
        IRBuilder<> IB(CtEnd[BB]); asmCall(IB, "vf 0($0)", "r", {segSym(Tails[k].second)});
      }
    }
    // terminators
    for (BasicBlock *BB : L->blocks()) {
      auto *Br = cast<BranchInst>(BB->getTerminator());
      Value *Cond = nullptr;
      if (Br->isConditional()) { Cond = Cl.clone(Br->getCondition()); if (!Cond) return false; }
      IRBuilder<> IB(CtEnd[BB]);
      if (Cond) IB.CreateCondBr(Cond, edgeTarget(BB, Br->getSuccessor(0)), edgeTarget(BB, Br->getSuccessor(1)));
      else IB.CreateBr(edgeTarget(BB, Br->getSuccessor(0)));
    }
    // phi incomings (values clone into their own blocks, before the terminators)
    for (auto &[KP, CP] : Phis)
      for (unsigned i = 0; i < KP->getNumIncomingValues(); i++) {
        BasicBlock *P = KP->getIncomingBlock(i);
        Value *CV = Cl.clone(KP->getIncomingValue(i)); if (!CV) return false;
        CP->addIncoming(CV, L->contains(P) ? edgeSource(P, KP->getParent()) : Check);
      }
    if (C.NeedFence) asmCall(B, "fence", "", {});   // drain earlier vector stores before the scalar loads
    if (MayBeSkipped) B.CreateCondBr(Reach, CtBlock[H], After); else B.CreateBr(CtBlock[H]);
    B.SetInsertPoint(After);
    // uniform LCSSA values of this region used later
    BasicBlock *ExitPred = edgeSource(Exiting, Exit);
    for (PHINode &Phi : Exit->phis()) {
      if (!WT.isUniformValue(&Phi)) continue;
      Value *In = Cl.clone(Phi.getIncomingValueForBlock(Exiting)); if (!In) return false;
      if (MayBeSkipped) {
        PHINode *P = B.CreatePHI(In->getType(), 2, Phi.getName());
        P->addIncoming(In, ExitPred); P->addIncoming(PoisonValue::get(In->getType()), Check);
        GMap[&Phi] = P;
      } else GMap[&Phi] = In;
    }
    Cl.L = nullptr; Cl.CtBlock = nullptr;
    return true;
  };
  for (const WTGen::Segment &Seg : WT.Segments) {
    if (Seg.CT >= 0 && !emitRegion(WT.CTLoops[Seg.CT], Seg.CT)) return false;
    if (Seg.Reduce >= 0 && !emitReduce(Seg.Reduce)) return false;
    asmCall(B, "vf 0($0)", "r", {segSym(Seg.Label)});
  }
  BasicBlock *Last = B.GetInsertBlock();
  Value *RemN = B.CreateSub(Rem, VL);
  Value *OffN = B.CreateAdd(Off, VL);
  Rem->addIncoming(N, Entry); Rem->addIncoming(RemN, Last);
  Off->addIncoming(ConstantInt::get(I64, 0), Entry); Off->addIncoming(OffN, Last);
  Grp->addIncoming(ConstantInt::get(I64, 0), Entry); Grp->addIncoming(B.CreateAdd(Grp, ConstantInt::get(I64, 1)), Last);
  for (unsigned i = 0; i < Bases.size(); i++) {
    if (!Bases[i]) continue;
    Value *Next = BasePhis[i];
    if (!WT.Streams[i].Local) {          // local-memory streams restart at the buffer for every group
      Value *Step = B.CreateMul(VL, StrideVals[i]);
      Next = B.CreatePtrAdd(BasePhis[i], Step);
    }
    BasePhis[i]->addIncoming(Bases[i], Entry); BasePhis[i]->addIncoming(Next, Last);
  }
  B.CreateCondBr(B.CreateICmpNE(RemN, ConstantInt::get(I64, 0)), Loop, Exit);

  B.SetInsertPoint(Exit);
  asmCall(B, "fence", "", {});
  B.CreateRetVoid();
  return true;
}

} // namespace

bool hwacha::generateKernel(Function &Kernel, KernelAnalysis &KA, Module &CTModule,
                            raw_ostream &WTOut, raw_ostream &Err, const CodeGenOptions &Opts) {
  WTGen WT(Kernel, KA, Err, Opts);
  if (!WT.run()) return false;
  if (Opts.Stats) {
    // Spike model: maxvl = 8*(256/nvv) capped by 8*(1024/nvp); hardware scales the same way
    unsigned NReg = WT.NumVV + WT.NumVW;   // Spike counts all widths alike; hardware packs two 32-bit values per entry
    unsigned MaxVLv = NReg < 2 ? 2048 : 8 * (256 / NReg);
    unsigned MaxVLp = WT.NumVP < 2 ? 8192 : 8 * (1024 / WT.NumVP);
    outs() << formatv("{0,-14} vv64={1,2} vv32={11,2} vs={2,3} vp={3,3}  maxvl={4,5}  insts={5,4}  masks={6,3} moves={7,3} jumps={8,3}  streams={9} inputs={10}\n",
                      Kernel.getName(), WT.NumVV, WT.NumVS, WT.NumVP, std::min(MaxVLv, MaxVLp),
                      WT.NumInsts, WT.NumMasks, WT.NumMoves, WT.NumJumps, WT.Streams.size(), WT.VSInputs.size(), WT.NumVW);
  }
  WTOut << "\n    .text\n    .align 3\n    .globl " << Kernel.getName() << "_wt\n"
        << Kernel.getName() << "_wt:\n" << WT.Text;
  return generateCT(Kernel, WT, CTModule, Err);
}

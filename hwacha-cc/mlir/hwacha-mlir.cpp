// hwacha-mlir: MLIR front end of hwacha-cc.
//   in.mlir (gpu dialect, or linalg/scf.parallel over memrefs) -> kernels as LLVM IR for hwacha-cc
//                                                              -> host as LLVM IR (gpu.launch_func lowered)
// Pipeline: [linalg -> parallel loops -> collapse to 1-D -> map -> gpu -> outline] -> lower-affine ->
// scf-to-cf -> {kernel: convert-gpu-to-nvvm; host: *-to-llvm, launch_func -> call <kernel>_ct}.
#include <algorithm>
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm::cl;

static opt<std::string> InputFile(Positional, desc("<input .mlir>"), Required);
static opt<std::string> OutputFile("o", desc("kernel LLVM IR output (.ll)"), init("-"));
static opt<std::string> HostFile("host", desc("host LLVM IR output (.ll); empty = do not emit the host"));
static opt<std::string> WeightsBin("weights-bin", desc("strip large constant globals from the host to this binary blob + a sibling .S stub (.incbin); empty = inline them as .word"));
static opt<std::string> MemRefConv("memref", desc("memref calling convention on both sides: bare (static shapes, default) or desc"), init("bare"));
static opt<bool> NoCollapse("no-collapse", desc("do not collapse multi-dimensional scf.parallel loops to 1-D"));
static opt<bool> CollapseAll("collapse-all", desc("collapse every dimension of a parallel loop into the lane dimension (old behaviour); default: innermost dimension = lanes, the others become control-thread loops"));
static opt<bool> NoConvLib("no-conv-lib", desc("do not lower linalg.conv_2d_nchw_fchw to the hwlib kernels (conv3x3 / conv3x3_s2 / conv1x1)"));
static opt<bool> FuseGenerics("fuse-generics", desc("collapse trailing parallel dims of elementwise generics into the lane dimension (larger vl, but delinearizes non-contiguous operands); default off: the innermost dim alone is the lane dimension"));
static opt<unsigned> UnrollSmall("unroll-small", desc("fully unroll kernel scf.for loops with at most this many iterations (0 = never)"), init(8));
static opt<bool> PrintMLIR("print-mlir", desc("print the MLIR after each stage to stderr"));

static const char *RISCV_TRIPLE = "riscv64-unknown-elf";
static const char *RISCV_DL = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128";

static bool runPipeline(ModuleOp m, StringRef pipeline) {
  PassManager pm(m.getContext());   // anchored on builtin.module: the text must not repeat the wrapper
  if (failed(parsePassPipeline(pipeline, pm))) { llvm::errs() << "hwacha-mlir: bad pipeline " << pipeline << "\n"; return false; }
  if (failed(pm.run(m))) { llvm::errs() << "hwacha-mlir: pipeline failed: " << pipeline << "\n"; return false; }
  if (PrintMLIR) { llvm::errs() << "// ---- after " << pipeline << "\n"; m.print(llvm::errs()); llvm::errs() << "\n"; }
  return true;
}

static bool hasGpuModule(ModuleOp m) {
  bool r = false; m.walk([&](gpu::GPUModuleOp) { r = true; }); return r;
}

// Every multi-dimensional scf.parallel becomes one-dimensional (Hwacha runs a 1-D NDRange).
static void collapseParallel(ModuleOp m) {
  SmallVector<scf::ParallelOp> loops;
  m.walk([&](scf::ParallelOp op) { if (op.getNumLoops() > 1 && !op->getParentOfType<scf::ParallelOp>()) loops.push_back(op); });
  IRRewriter rewriter(m.getContext());
  for (scf::ParallelOp op : loops) {
    std::vector<unsigned> all;
    for (unsigned i = 0; i < op.getNumLoops(); i++) all.push_back(i);
    rewriter.setInsertionPoint(op);
    collapseParallelLoops(rewriter, op, {all});
  }
}

// Nested scf.parallel loops (e.g. the row block of a tiled matmul) become sequential scf.for nests
// inside the work-item; only the outermost parallel loop is mapped to the NDRange.
static bool nestedParallelToFor(ModuleOp m) {
  SmallVector<scf::ParallelOp> nested;
  m.walk([&](scf::ParallelOp op) { if (op->getParentOfType<scf::ParallelOp>()) nested.push_back(op); });
  for (scf::ParallelOp op : llvm::reverse(nested)) {   // inner loops first
    if (op.getNumResults()) { op.emitError("nested parallel loop with reductions is not supported"); return false; }
    OpBuilder b(op);
    SmallVector<Value> ivs;
    scf::ForOp inner;
    for (unsigned i = 0; i < op.getNumLoops(); i++) {
      inner = scf::ForOp::create(b, op.getLoc(), op.getLowerBound()[i], op.getUpperBound()[i], op.getStep()[i]);
      ivs.push_back(inner.getInductionVar());
      b.setInsertionPointToStart(inner.getBody());
    }
    Block *src = op.getBody(), *dst = inner.getBody();
    for (Operation &o : llvm::make_early_inc_range(src->without_terminator())) o.moveBefore(dst->getTerminator());
    for (unsigned i = 0; i < ivs.size(); i++) src->getArgument(i).replaceAllUsesWith(ivs[i]);
    op.erase();
  }
  return true;
}

// ---------------------------------------------------------------- performance patterns for torch-mlir output
// 1. Elementwise generics: collapse the largest trailing group of parallel dimensions that every operand
//    keeps contiguous (identity maps: all of them; a per-channel broadcast operand: the (h, w) pair).
//    The collapsed dimension becomes the lane dimension with unit-stride addresses; the remaining
//    outer dimensions become control-thread loops. Without this, collapsing everything to 1-D
//    recovers (n, c, h, w) with vdiv/vrem and every access is a gather.
static void collapseGenerics(ModuleOp m) {
  SmallVector<linalg::GenericOp> gens;
  m.walk([&](linalg::GenericOp g) { gens.push_back(g); });
  IRRewriter rw(m.getContext());
  for (linalg::GenericOp g : gens) {
    unsigned n = g.getNumLoops();
    if (n < 2) continue;
    bool identity = true;
    for (Value v : g->getOperands())
      if (auto mt = dyn_cast<MemRefType>(v.getType())) if (!mt.getLayout().isIdentity()) identity = false;
    if (!identity) continue;   // strided subviews (pad / concat copies) are not collapsible
    SmallVector<AffineMap> maps = g.getIndexingMapsArray();
    SmallVector<utils::IteratorType> its = g.getIteratorTypesArray();
    SmallVector<int64_t> ranges = g.getStaticLoopRanges();
    for (unsigned k = n; k >= 2; --k) {
      ReassociationIndices grp;
      bool par = true;
      for (unsigned d = n - k; d < n; ++d) { grp.push_back(d); if (its[d] != utils::IteratorType::parallel) par = false; }
      if (!par) continue;
      // never fold a size-1 dimension into the group: collapsing it forces a linearize/delinearize
      // (with signed guards LLVM cannot remove) instead of a clean memref.collapse_shape. Leave those
      // (usually the leading N/C=1 dims) for laneInnermost.
      if (llvm::any_of(grp, [&](int64_t d) { return ranges[d] == 1; })) continue;
      if (!linalg::areDimSequencesPreserved(maps, {grp})) continue;
      rw.setInsertionPoint(g);
      FailureOr<linalg::CollapseResult> r = linalg::collapseOpIterationDims(cast<linalg::LinalgOp>(g.getOperation()), {grp}, rw);
      if (succeeded(r)) { rw.eraseOp(g); break; }
    }
  }
}

// 2. Multi-dimensional scf.parallel: the innermost dimension is the lane dimension, the others become
//    control-thread loops inside the kernel (legal: every dimension of an scf.parallel is parallel).
static void laneInnermost(ModuleOp m) {
  SmallVector<scf::ParallelOp> loops;
  m.walk([&](scf::ParallelOp op) { if (op.getNumLoops() > 1 && !op->getParentOfType<scf::ParallelOp>() && !op.getNumResults()) loops.push_back(op); });
  for (scf::ParallelOp op : loops) {
    unsigned n = op.getNumLoops();
    Location loc = op.getLoc();
    OpBuilder b(op);
    auto np = scf::ParallelOp::create(b, loc, ValueRange{op.getLowerBound()[n - 1]}, ValueRange{op.getUpperBound()[n - 1]}, ValueRange{op.getStep()[n - 1]});
    b.setInsertionPointToStart(np.getBody());
    SmallVector<Value> ivs(n);
    ivs[n - 1] = np.getInductionVars()[0];
    Block *dst = np.getBody();
    for (unsigned d = 0; d + 1 < n; d++) {
      scf::ForOp f = scf::ForOp::create(b, loc, op.getLowerBound()[d], op.getUpperBound()[d], op.getStep()[d]);
      ivs[d] = f.getInductionVar();
      b.setInsertionPointToStart(f.getBody());
      dst = f.getBody();
    }
    Block *src = op.getBody();
    for (Operation &o : llvm::make_early_inc_range(src->without_terminator())) o.moveBefore(dst->getTerminator());
    for (unsigned d = 0; d < n; d++) src->getArgument(d).replaceAllUsesWith(ivs[d]);
    op.erase();
  }
}

// 3. linalg.conv_2d_nchw_fchw (N = 1, 3x3 stride 1 or 2 on a zero-padded input, or 1x1 stride 1) -> calls
//    of the hand-written kernels of hwlib.cl. 3x3 kernels write zero-padded scratch planes with a guard,
//    which `unpad` copies back into the dense output. The zero-filled output torch-mlir prepares (linalg.fill)
//    is dropped: the kernels write every element.
static LLVM::LLVMFuncOp declareFn(ModuleOp m, StringRef name, ArrayRef<Type> args) {
  if (auto f = m.lookupSymbol<LLVM::LLVMFuncOp>(name)) return f;
  OpBuilder b(m.getContext()); b.setInsertionPointToStart(m.getBody());
  return LLVM::LLVMFuncOp::create(b, m.getLoc(), name, LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(m.getContext()), args));
}
// The op that last wrote `buf` before `before` in the same block, initializing every element either to
// zero (linalg.fill 0, or a copy of a zero buffer) or to a per-output-channel bias (linalg.broadcast of a
// 1-D memref along [0,2,3], i.e. b[c] into [1][C][H][W] -- exactly what the conv kernels add). On success
// returns the init op and, in `bias`, the bias memref (null for a zero init). nullptr on any other writer.
static Operation *initWriter(Value buf, Operation *before, Value &bias) {
  static bool DBG = getenv("HWDBG");
  for (Operation *op = before->getPrevNode(); op; op = op->getPrevNode()) {
    if (!llvm::is_contained(op->getOperands(), buf)) continue;
    if (auto f = dyn_cast<linalg::FillOp>(op)) {
      if (f.getDpsInits()[0] != buf) { if (DBG) llvm::errs() << "// fill: init != buf\n"; return nullptr; }
      auto cst = f.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      if (!cst) { if (DBG) llvm::errs() << "// fill: input not constant: " << f.getInputs()[0] << "\n"; return nullptr; }
      auto fa = dyn_cast<FloatAttr>(cst.getValue());
      if (!fa || !fa.getValue().isZero()) { if (DBG) llvm::errs() << "// fill: not zero\n"; return nullptr; }
      bias = Value(); return op;
    }
    if (auto bc = dyn_cast<linalg::BroadcastOp>(op)) {
      if (bc.getInit() != buf) return nullptr;
      auto src = dyn_cast<MemRefType>(bc.getInput().getType());
      ArrayRef<int64_t> dims = bc.getDimensions();
      if (!src || src.getRank() != 1 || dims.size() != 3 || dims[0] != 0 || dims[1] != 2 || dims[2] != 3) return nullptr;
      bias = bc.getInput(); return op;
    }
    if (auto c = dyn_cast<linalg::CopyOp>(op)) {
      if (c.getDpsInits()[0] != buf) return nullptr;
      Value b2; Operation *w = initWriter(c.getDpsInputs()[0], op, b2);
      if (PrintMLIR && !w) llvm::errs() << "//   copy source " << c.getDpsInputs()[0] << " has no recognizable init\n";
      return (w && !b2) ? (bias = Value(), op) : nullptr;
    }
    if (auto c = dyn_cast<memref::CopyOp>(op)) {
      if (c.getTarget() != buf) { if (buf == c.getSource()) continue; return nullptr; }
      Value b2; Operation *w = initWriter(c.getSource(), op, b2);
      return (w && !b2) ? (bias = Value(), op) : nullptr;
    }
    // any other linalg op that only READS buf (as an input) leaves the init intact: keep looking back
    if (auto lo = dyn_cast<linalg::LinalgOp>(op))
      if (llvm::is_contained(lo.getDpsInputs(), buf) && !llvm::is_contained(lo.getDpsInits(), buf)) continue;
    return nullptr;
  }
  return nullptr;
}
static bool lowerConvs(ModuleOp m) {
  MLIRContext *ctx = m.getContext();
  SmallVector<linalg::Conv2DNchwFchwOp> convs;
  m.walk([&](linalg::Conv2DNchwFchwOp c) { convs.push_back(c); });
  if (convs.empty()) return true;
  Type i64 = IntegerType::get(ctx, 64), i32 = IntegerType::get(ctx, 32), f32 = Float32Type::get(ctx);
  Type ptrTy = LLVM::LLVMPointerType::get(ctx);
  Type intsKK[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32, i32, i32};   // ..., n_in, plane, Wp, K, pad, oc
  LLVM::LLVMFuncOp ckk = declareFn(m, "convKxK_ct", intsKK), ckk1 = declareFn(m, "convKxK_1_ct", intsKK);
  Type ints11[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32, i32, i32, i32};   // n, x,w,b,y, n_in, plane, Wp, plane2, Wp2, K, oc
  LLVM::LLVMFuncOp cs2 = declareFn(m, "convKxK_s2_ct", ints11);
  Type ints8[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32};
  LLVM::LLVMFuncOp c1 = declareFn(m, "conv1x1_ct", ints8), c11 = declareFn(m, "conv1x1_1_ct", ints8);
  Type intsU[] = {i64, ptrTy, ptrTy, i32, i32, i32, i32, i32};   // ..., plane, Wp, H, W, pad
  LLVM::LLVMFuncOp unpad = declareFn(m, "unpad_ct", intsU);
  Type intsP[] = {i64, ptrTy, ptrTy, i32, i32, i32, i32, i32, i32, i32};   // ..., C, Hi, Wi, Ho, Wo, K, S
  LLVM::LLVMFuncOp poolmax = declareFn(m, "poolmax_ct", intsP);
  Type intsDW[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32, i32};         // ..., C, plane, Wp, K, pad
  LLVM::LLVMFuncOp dwkk = declareFn(m, "dwconvKxK_ct", intsDW);
  Type intsDW2[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32, i32, i32, i32};   // ..., C, plane, Wp, plane2, Wp2, K, pad
  LLVM::LLVMFuncOp dwkk2 = declareFn(m, "dwconvKxK_s2_ct", intsDW2);
  Type intsPE[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32, i32, i32, i32};   // ..., C, xplane, Wi, P, gw, plane, oc
  LLVM::LLVMFuncOp pe = declareFn(m, "patchembed_ct", intsPE), pe1 = declareFn(m, "patchembed_1_ct", intsPE);
  // linalg.depthwise_conv_2d_nchw_chw (N=1) -> dwconvKxK / dwconvKxK_s2 into a padded scratch + unpad
  SmallVector<linalg::DepthwiseConv2DNchwChwOp> dws;
  m.walk([&](linalg::DepthwiseConv2DNchwChwOp d) { dws.push_back(d); });
  for (linalg::DepthwiseConv2DNchwChwOp dw : dws) {
    auto xT = dyn_cast<MemRefType>(dw.getInputs()[0].getType()), wT = dyn_cast<MemRefType>(dw.getInputs()[1].getType()), yT = dyn_cast<MemRefType>(dw.getOutputs()[0].getType());
    if (!xT || !wT || !yT || !xT.hasStaticShape() || !yT.hasStaticShape() || !xT.getLayout().isIdentity() || !yT.getLayout().isIdentity()) continue;
    auto xs = xT.getShape(), ws = wT.getShape(), ys = yT.getShape();
    if (xs[0] != 1 || ws[1] != ws[2]) continue;
    auto sv = dw.getStrides().getValues<int64_t>(), dv = dw.getDilations().getValues<int64_t>();
    int64_t stride = sv[0], K = ws[1], C = xs[1], Hi = xs[2], Wi = xs[3], Ho = ys[2], Wo = ys[3];
    if (sv[1] != stride || dv[0] != 1 || dv[1] != 1 || K % 2 == 0) continue;
    int64_t pad;
    if (stride == 1 && Hi == Ho + K - 1 && Wi == Wo + K - 1) pad = (K - 1) / 2;
    else if (stride == 2 && Hi == 2 * Ho + K - 1 && Wi == 2 * Wo + K - 1) pad = K - 1;   // pre-padded input, like conv3x3_s2 (base offset -(K-1))
    else { if (PrintMLIR) llvm::errs() << "// conv-lib: unsupported depthwise\n"; continue; }
    Location loc = dw.getLoc(); OpBuilder b(dw);
    auto ptrOf = [&](Value mem){ Value idx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, b.getIndexType(), mem); Value ii = arith::IndexCastOp::create(b, loc, IntegerType::get(ctx,64), idx); return LLVM::IntToPtrOp::create(b, loc, LLVM::LLVMPointerType::get(ctx), ii); };
    auto i32c = [&](int64_t v){ return LLVM::ConstantOp::create(b, loc, IntegerType::get(ctx,32), b.getI32IntegerAttr(v)); };
    auto i64c = [&](int64_t v){ return LLVM::ConstantOp::create(b, loc, IntegerType::get(ctx,64), b.getI64IntegerAttr(v)); };
    Value xp = ptrOf(dw.getInputs()[0]), wp = ptrOf(dw.getInputs()[1]), yp = ptrOf(dw.getOutputs()[0]);
    Value z = memref::AllocOp::create(b, loc, MemRefType::get({std::max<int64_t>(C, 8)}, Float32Type::get(ctx))); linalg::FillOp::create(b, loc, ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0f))}, ValueRange{z}); Value zp = ptrOf(z);
    int64_t plane = Hi * Wi, Wp = Wi;
    if (stride == 1) {
      int64_t guard = 2 * Wp + 8;
      Value scratch = memref::AllocOp::create(b, loc, MemRefType::get({C * plane + 2 * guard}, Float32Type::get(ctx)));
      Value sp = LLVM::GEPOp::create(b, loc, LLVM::LLVMPointerType::get(ctx), Float32Type::get(ctx), ptrOf(scratch), ArrayRef<LLVM::GEPArg>{(int32_t)guard});
      LLVM::CallOp::create(b, loc, dwkk, ValueRange{i64c(plane), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(K), i32c(pad)});
      LLVM::CallOp::create(b, loc, unpad, ValueRange{i64c(C * Ho * Wo), sp, yp, i32c(plane), i32c(Wp), i32c(Ho), i32c(Wo), i32c(pad)});
    } else {
      int64_t Hp2 = Ho + 2, Wp2 = Wo + 2, plane2 = Hp2 * Wp2, guard = 2 * Wp2 + 8;
      Value scratch = memref::AllocOp::create(b, loc, MemRefType::get({C * plane2 + 2 * guard}, Float32Type::get(ctx)));
      Value sp = LLVM::GEPOp::create(b, loc, LLVM::LLVMPointerType::get(ctx), Float32Type::get(ctx), ptrOf(scratch), ArrayRef<LLVM::GEPArg>{(int32_t)guard});
      LLVM::CallOp::create(b, loc, dwkk2, ValueRange{i64c(plane2), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(plane2), i32c(Wp2), i32c(K), i32c(pad)});
      LLVM::CallOp::create(b, loc, unpad, ValueRange{i64c(C * Ho * Wo), sp, yp, i32c(plane2), i32c(Wp2), i32c(Ho), i32c(Wo), i32c(1)});
    }
    dw.erase();
  }
  // linalg.pooling_nchw_max (N=1, dense) -> poolmax_ct
  SmallVector<linalg::PoolingNchwMaxOp> pools;
  m.walk([&](linalg::PoolingNchwMaxOp p) { pools.push_back(p); });
  for (linalg::PoolingNchwMaxOp pool : pools) {
    auto xT = dyn_cast<MemRefType>(pool.getInputs()[0].getType()), yT = dyn_cast<MemRefType>(pool.getOutputs()[0].getType());
    auto wT = dyn_cast<MemRefType>(pool.getInputs()[1].getType());
    if (!xT || !yT || !wT || !xT.hasStaticShape() || !yT.hasStaticShape() || !xT.getLayout().isIdentity() || !yT.getLayout().isIdentity()) continue;
    auto xs = xT.getShape(), ys = yT.getShape(), ks = wT.getShape();
    if (xs[0] != 1 || ks[0] != ks[1]) continue;
    auto sv = pool.getStrides().getValues<int64_t>();
    if (sv[0] != sv[1]) continue;
    Location loc = pool.getLoc();
    OpBuilder b(pool);
    auto ptrOf = [&](Value mem) -> Value { Value idx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, b.getIndexType(), mem); Value ii = arith::IndexCastOp::create(b, loc, i64, idx); return LLVM::IntToPtrOp::create(b, loc, ptrTy, ii); };
    auto i32c = [&](int64_t v) { return LLVM::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v)); };
    LLVM::CallOp::create(b, loc, poolmax, ValueRange{LLVM::ConstantOp::create(b, loc, i64, b.getI64IntegerAttr(ys[1] * ys[2] * ys[3])),
      ptrOf(pool.getInputs()[0]), ptrOf(pool.getOutputs()[0]), i32c(xs[1]), i32c(xs[2]), i32c(xs[3]), i32c(ys[2]), i32c(ys[3]), i32c(ks[0]), i32c(sv[0])});
    pool.erase();
  }
  // spatial-sum reduction generics (global / adaptive avg-pool numerator): a linalg.generic with
  // (parallel, parallel, reduction, reduction) iterators, one dense input, output 1xCx1x1, body out += in.
  Type intsCS[] = {i64, ptrTy, ptrTy, i32, i32};   // n, x, y, C, HW
  LLVM::LLVMFuncOp chansum = declareFn(m, "chansum_ct", intsCS);
  SmallVector<linalg::GenericOp> reds;
  m.walk([&](linalg::GenericOp g) {
    auto its = g.getIteratorTypesArray();
    if (its.size() != 4 || its[0] != utils::IteratorType::parallel || its[1] != utils::IteratorType::parallel
        || its[2] != utils::IteratorType::reduction || its[3] != utils::IteratorType::reduction) return;
    if (g.getInputs().size() != 1 || g.getOutputs().size() != 1) return;
    Block &bb = g.getRegion().front();
    if (!llvm::hasSingleElement(bb.without_terminator())) return;
    auto add = dyn_cast<arith::AddFOp>(&bb.front());
    if (!add) return;
    reds.push_back(g);
  });
  for (linalg::GenericOp g : reds) {
    auto xT = dyn_cast<MemRefType>(g.getInputs()[0].getType()), yT = dyn_cast<MemRefType>(g.getOutputs()[0].getType());
    if (!xT || !yT || !xT.hasStaticShape() || !xT.getLayout().isIdentity() || !yT.getLayout().isIdentity()) continue;
    auto xs = xT.getShape();
    if (xs[0] != 1) continue;
    int64_t C = xs[1], HW = xs[2] * xs[3];
    Location loc = g.getLoc(); OpBuilder b(g);
    Value biasMem; Operation *init = initWriter(g.getOutputs()[0], g, biasMem);
    if (!init || biasMem) continue;   // output must be zero-initialized (sum starts at 0)
    auto ptrOf = [&](Value mem){ Value idx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, b.getIndexType(), mem); Value ii = arith::IndexCastOp::create(b, loc, IntegerType::get(ctx,64), idx); return LLVM::IntToPtrOp::create(b, loc, ptrTy, ii); };
    auto i32c = [&](int64_t v){ return LLVM::ConstantOp::create(b, loc, IntegerType::get(ctx,32), b.getI32IntegerAttr(v)); };
    LLVM::CallOp::create(b, loc, chansum, ValueRange{LLVM::ConstantOp::create(b, loc, IntegerType::get(ctx,64), b.getI64IntegerAttr(C)), ptrOf(g.getInputs()[0]), ptrOf(g.getOutputs()[0]), i32c(C), i32c(HW)});
    init->erase(); g.erase();
  }
  int lowered = 0;
  for (linalg::Conv2DNchwFchwOp conv : convs) {
    auto xT = dyn_cast<MemRefType>(conv.getInputs()[0].getType()), wT = dyn_cast<MemRefType>(conv.getInputs()[1].getType()), yT = dyn_cast<MemRefType>(conv.getOutputs()[0].getType());
    if (!xT || !wT || !yT || !xT.hasStaticShape() || !wT.hasStaticShape() || !yT.hasStaticShape()) continue;
    if (!xT.getLayout().isIdentity() || !wT.getLayout().isIdentity() || !yT.getLayout().isIdentity()) continue;
    auto xs = xT.getShape(), ws = wT.getShape(), ys = yT.getShape();
    if (xs[0] != 1 || ys[0] != 1) continue;
    int64_t C = xs[1], Hi = xs[2], Wi = xs[3], O = ws[0], kh = ws[2], kw = ws[3], Ho = ys[2], Wo = ys[3];
    auto sv = conv.getStrides().getValues<int64_t>(), dv = conv.getDilations().getValues<int64_t>();
    int64_t stride = sv[0];
    if (sv[1] != stride || dv[0] != 1 || dv[1] != 1 || ws[1] != C) continue;
    enum { KxKS1, KxKS2, K1S1, KEMBED } kind;
    int64_t pad = 0;
    if (kh == kw && kh % 2 == 1 && kh >= 3 && stride == 1 && Hi == Ho + kh - 1 && Wi == Wo + kw - 1) { kind = KxKS1; pad = (kh - 1) / 2; }
    else if (kh == kw && (kh % 2 == 1 || kh == 1) && stride == 2 && Hi == 2 * Ho + kh - 1 && Wi == 2 * Wo + kw - 1 && O % 4 == 0) kind = KxKS2;
    else if (kh == 1 && kw == 1 && stride == 1 && Hi == Ho && Wi == Wo) kind = K1S1;
    else if (kh == kw && kh == stride && kh >= 2 && Hi == Ho * kh && Wi == Wo * kw) kind = KEMBED;   // ViT/DiT patchify: kernel = stride, non-overlapping, no pad
    else { if (PrintMLIR) llvm::errs() << "// conv-lib: unsupported shape " << conv << "\n"; continue; }
    // The kernels accumulate onto the output torch-mlir has already initialized (a zero fill or a
    // per-channel bias broadcast), so the bias/zero is whatever is already in the output: pass a zero
    // bias, keep the init op, and the conv1x1 direct writes / unpad copies add to it.
    Location loc = conv.getLoc();
    OpBuilder b(conv);
    auto ptrOf = [&](Value mem) -> Value {
      Value idx = memref::ExtractAlignedPointerAsIndexOp::create(b, loc, b.getIndexType(), mem);
      Value i = arith::IndexCastOp::create(b, loc, i64, idx);
      return LLVM::IntToPtrOp::create(b, loc, ptrTy, i);
    };
    auto i32c = [&](int64_t v) -> Value { return LLVM::ConstantOp::create(b, loc, i32, b.getI32IntegerAttr(v)); };
    auto i64c = [&](int64_t v) -> Value { return LLVM::ConstantOp::create(b, loc, i64, b.getI64IntegerAttr(v)); };
    Value xp = ptrOf(conv.getInputs()[0]), wp = ptrOf(conv.getInputs()[1]), yp = ptrOf(conv.getOutputs()[0]);
    // zero bias buffer: the kernels start each accumulator at b[oc] and the real bias/init lives in the
    // output, so this must be all-zero AND at least O long (the 8-wide passes read b[oc0..oc0+7]).
    Value zeros = memref::AllocOp::create(b, loc, MemRefType::get({std::max<int64_t>(O, 8)}, f32));
    linalg::FillOp::create(b, loc, ValueRange{arith::ConstantOp::create(b, loc, b.getF32FloatAttr(0.0f))}, ValueRange{zeros});
    Value zp = ptrOf(zeros);
    if (kind == K1S1) {
      int64_t plane = Ho * Wo, oc = 0;
      for (; oc + 8 <= O; oc += 8) LLVM::CallOp::create(b, loc, c1, ValueRange{i64c(plane), xp, wp, zp, yp, i32c(C), i32c(plane), i32c(oc)});
      for (; oc < O; oc++) LLVM::CallOp::create(b, loc, c11, ValueRange{i64c(plane), xp, wp, zp, yp, i32c(C), i32c(plane), i32c(oc)});
    } else if (kind == KEMBED) {
      int64_t plane = Ho * Wo, xplane = Hi * Wi, oc = 0;   // patchify: P = kh, gw = Wo, output [O][Ho*Wo]
      for (; oc + 8 <= O; oc += 8) LLVM::CallOp::create(b, loc, pe, ValueRange{i64c(plane), xp, wp, zp, yp, i32c(C), i32c(xplane), i32c(Wi), i32c(kh), i32c(Wo), i32c(plane), i32c(oc)});
      for (; oc < O; oc++) LLVM::CallOp::create(b, loc, pe1, ValueRange{i64c(plane), xp, wp, zp, yp, i32c(C), i32c(xplane), i32c(Wi), i32c(kh), i32c(Wo), i32c(plane), i32c(oc)});
    } else {
      int64_t plane = Hi * Wi, Wp = Wi;
      int64_t Hp2 = Ho + 2, Wp2 = Wo + 2, plane2 = Hp2 * Wp2;
      int64_t oplane = kind == KxKS1 ? plane : plane2, oWp = kind == KxKS1 ? Wp : Wp2, oPad = kind == KxKS1 ? pad : 1;
      int64_t guard = 2 * oWp + 8;
      Value scratch = memref::AllocOp::create(b, loc, MemRefType::get({O * oplane + 2 * guard}, f32));
      Value sp = LLVM::GEPOp::create(b, loc, ptrTy, f32, ptrOf(scratch), ArrayRef<LLVM::GEPArg>{(int32_t)guard});
      if (kind == KxKS1) {
        int64_t oc = 0;
        for (; oc + 8 <= O; oc += 8) LLVM::CallOp::create(b, loc, ckk, ValueRange{i64c(plane), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(kh), i32c(pad), i32c(oc)});
        for (; oc < O; oc++) LLVM::CallOp::create(b, loc, ckk1, ValueRange{i64c(plane), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(kh), i32c(pad), i32c(oc)});
      } else {
        for (int64_t oc = 0; oc < O; oc += 4) LLVM::CallOp::create(b, loc, cs2, ValueRange{i64c(plane2), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(plane2), i32c(Wp2), i32c(kh), i32c(oc)});
      }
      LLVM::CallOp::create(b, loc, unpad, ValueRange{i64c(O * Ho * Wo), sp, yp, i32c(oplane), i32c(oWp), i32c(Ho), i32c(Wo), i32c(oPad)});
    }
    conv.erase();
    lowered++;
  }
  if (PrintMLIR) llvm::errs() << "// ---- conv-lib: " << lowered << " of " << convs.size() << " convolutions lowered to hwlib calls\n";
  return true;
}

// The outermost scf.parallel loops (1-D after collapsing) run one work-item per iteration (block_x,
// block size 1).
static void mapParallel(ModuleOp m) {
  MLIRContext *ctx = m.getContext();
  AffineMap id = AffineMap::getMultiDimIdentityMap(1, ctx);
  m.walk([&](scf::ParallelOp op) {
    SmallVector<gpu::ParallelLoopDimMappingAttr> maps;
    for (unsigned i = 0; i < op.getNumLoops(); i++)
      maps.push_back(gpu::ParallelLoopDimMappingAttr::get(ctx, i > 0 ? gpu::Processor::Sequential : gpu::Processor::BlockX, id, id));
    (void)gpu::setMappingAttr(op, maps);
  });
}

// Constants used inside a gpu.launch are cloned into it so that outlining does not turn loop bounds,
// steps and strides into kernel arguments (that would hide them from alias analysis and stream detection).
static void sinkConstants(ModuleOp m) {
  m.walk([&](gpu::LaunchOp launch) {
    Region &r = launch.getBody();
    SetVector<Value> used;
    getUsedValuesDefinedAbove(r, used);
    OpBuilder b(&r.front(), r.front().begin());
    for (Value v : used) {
      Operation *def = v.getDefiningOp();
      if (!def || def->getNumOperands() != 0 || def->getNumResults() != 1 || !isMemoryEffectFree(def)) continue;
      Operation *c = b.clone(*def);
      v.replaceUsesWithIf(c->getResult(0), [&](OpOperand &u) { return r.isAncestor(u.getOwner()->getParentRegion()); });
    }
  });
}

// Small constant-trip-count loops inside kernels (e.g. the 4-row block of a tiled matmul) are unrolled
// so their bodies become straight-line code that keeps accumulators in vector registers.
static void unrollSmall(ModuleOp m) {
  if (!UnrollSmall) return;
  SmallVector<scf::ForOp> loops;
  m.walk<WalkOrder::PostOrder>([&](scf::ForOp f) { if (f->getParentOfType<gpu::LaunchOp>() || f->getParentOfType<gpu::GPUFuncOp>()) loops.push_back(f); });
  for (scf::ForOp f : loops) {
    auto lb = getConstantIntValue(f.getLowerBound()), ub = getConstantIntValue(f.getUpperBound()), st = getConstantIntValue(f.getStep());
    if (!lb || !ub || !st || *st <= 0) continue;
    int64_t tc = (*ub - *lb + *st - 1) / *st;
    if (tc >= 1 && tc <= (int64_t)UnrollSmall) (void)loopUnrollFull(f);
  }
}

// gpu-kernel-outlining names every kernel <func>_kernel inside a gpu.module of a unique name
// (<func>_kernel, <func>_kernel_0, ...): once the modules are flattened the functions would collide,
// so every kernel takes its module's name (and the launches follow).
static void uniqueKernelNames(ModuleOp m) {
  m.walk([&](gpu::GPUModuleOp gm) {
    for (auto f : gm.getOps<gpu::GPUFuncOp>()) {
      if (f.getName() == gm.getName()) continue;
      StringRef old = f.getName();
      m.walk([&](gpu::LaunchFuncOp l) {
        if (l.getKernelModuleName() == gm.getNameAttr() && l.getKernelName().getValue() == old)
          l.setKernelAttr(SymbolRefAttr::get(gm.getNameAttr(), {FlatSymbolRefAttr::get(gm.getNameAttr())}));
      });
      f.setName(gm.getName());
    }
  });
}

static Value lookThrough(Value v) {
  while (auto c = v.getDefiningOp<UnrealizedConversionCastOp>()) { if (c.getInputs().size() != 1) break; v = c.getInputs()[0]; }
  return v;
}

// gpu.launch_func -> hwacha_group_size/hwacha_grid_size stores + call <kernel>_ct(n, args...)
static bool lowerLaunches(ModuleOp m) {
  MLIRContext *ctx = m.getContext();
  OpBuilder b(ctx);
  Type i64 = b.getI64Type(), i1 = b.getI1Type();
  SmallVector<gpu::LaunchFuncOp> launches;
  m.walk([&](gpu::LaunchFuncOp op) { launches.push_back(op); });
  if (launches.empty()) return true;
  bool bare = MemRefConv == "bare";
  auto global = [&](StringRef name) {
    if (auto g = m.lookupSymbol<LLVM::GlobalOp>(name)) return g;
    OpBuilder gb(ctx); gb.setInsertionPointToStart(m.getBody());
    return LLVM::GlobalOp::create(gb, m.getLoc(), i64, false, LLVM::Linkage::External, name, Attribute());
  };
  for (gpu::LaunchFuncOp op : launches) {
    if (op.getAsyncToken() || !op.getAsyncDependencies().empty()) { op.emitError("async launches are not supported"); return false; }
    Location loc = op.getLoc();
    b.setInsertionPoint(op);
    auto i64v = [&](Value v) -> Value {
      v = lookThrough(v);
      if (v.getType() != i64) { op.emitError("launch dimension is not i64 after conversion"); return nullptr; }
      return v;
    };
    Value gx = i64v(op.getGridSizeX()), gy = i64v(op.getGridSizeY()), gz = i64v(op.getGridSizeZ());
    Value bx = i64v(op.getBlockSizeX()), by = i64v(op.getBlockSizeY()), bz = i64v(op.getBlockSizeZ());
    if (!gx || !gy || !gz || !bx || !by || !bz) return false;
    Value blocks = LLVM::MulOp::create(b, loc, LLVM::MulOp::create(b, loc, gx, gy), gz);
    Value threads = LLVM::MulOp::create(b, loc, LLVM::MulOp::create(b, loc, bx, by), bz);
    Value n = LLVM::MulOp::create(b, loc, blocks, threads);
    // a block of one work-item means "no work-groups": let the hardware pick the vector length
    Value one = LLVM::ConstantOp::create(b, loc, i64, b.getI64IntegerAttr(1));
    Value zero = LLVM::ConstantOp::create(b, loc, i64, b.getI64IntegerAttr(0));
    Value isOne = LLVM::ICmpOp::create(b, loc, i1, LLVM::ICmpPredicate::eq, threads, one);
    Value gs = LLVM::SelectOp::create(b, loc, isOne, zero, threads);
    LLVM::StoreOp::create(b, loc, gs, LLVM::AddressOfOp::create(b, loc, global("hwacha_group_size")));
    LLVM::StoreOp::create(b, loc, blocks, LLVM::AddressOfOp::create(b, loc, global("hwacha_grid_size")));
    // kernel arguments: memrefs are LLVM descriptors here; bare -> aligned pointer, desc -> every field
    SmallVector<Value> args{n};
    for (Value a : op.getKernelOperands()) {
      Value v = lookThrough(a);
      if (auto st = dyn_cast<LLVM::LLVMStructType>(v.getType())) {
        if (bare) args.push_back(LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{1}));
        else {
          args.push_back(LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{0}));
          args.push_back(LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{1}));
          args.push_back(LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{2}));
          for (int f = 3; f <= 4; f++)
            if (st.getBody().size() > (size_t)f)
              if (auto arr = dyn_cast<LLVM::LLVMArrayType>(st.getBody()[f]))
                for (int64_t i = 0; i < (int64_t)arr.getNumElements(); i++)
                  args.push_back(LLVM::ExtractValueOp::create(b, loc, v, ArrayRef<int64_t>{f, i}));
        }
      } else if (isa<MemRefType, UnrankedMemRefType>(v.getType())) { op.emitError("memref operand was not converted"); return false; }
      else args.push_back(v);
    }
    std::string ct = (op.getKernelName().getValue() + "_ct").str();
    LLVM::LLVMFuncOp fn = m.lookupSymbol<LLVM::LLVMFuncOp>(ct);
    if (!fn) {
      SmallVector<Type> tys; for (Value v : args) tys.push_back(v.getType());
      OpBuilder fb(ctx); fb.setInsertionPointToStart(m.getBody());
      fn = LLVM::LLVMFuncOp::create(fb, loc, ct, LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(ctx), tys));
    }
    LLVM::CallOp::create(b, loc, fn, args);
    op.erase();
  }
  return true;
}

// Stream a constant's in-memory bytes (row-major) to `bin`, returning the count. Handles the forms
// torch-mlir weights take: flat float/int arrays (ConstantDataSequential), nested arrays like
// [M x [N x float]] (ConstantArray, recursed), and all-zero (ConstantAggregateZero). Returns -1 for
// anything else so the caller leaves that global inline.
static int64_t streamConstBytes(llvm::Constant *C, const llvm::DataLayout &DL, llvm::raw_ostream &bin) {
  if (auto *cds = llvm::dyn_cast<llvm::ConstantDataSequential>(C)) {
    llvm::StringRef b = cds->getRawDataValues(); bin.write(b.data(), b.size()); return b.size();
  }
  if (llvm::isa<llvm::ConstantAggregateZero>(C)) {
    int64_t sz = DL.getTypeAllocSize(C->getType());
    for (int64_t i = 0; i < sz; i++) bin << '\0';
    return sz;
  }
  if (auto *ca = llvm::dyn_cast<llvm::ConstantArray>(C)) {
    int64_t tot = 0;
    for (unsigned i = 0, e = ca->getNumOperands(); i < e; i++) {
      int64_t n = streamConstBytes(ca->getOperand(i), DL, bin);
      if (n < 0) return -1;
      tot += n;
    }
    return tot;
  }
  return -1;
}

// Shared state for weight stripping across the kernel and host modules: both reference the same weight
// globals (constants get sunk into the kernels too), so they are written to one blob + .S, deduplicated
// by symbol name, and made external in both .ll files.
struct WeightStrip {
  std::unique_ptr<llvm::raw_fd_ostream> bin, asmf;
  std::string binName;
  uint64_t off = 0, n = 0, bytes = 0;
  llvm::StringSet<> done;
};

static bool emit(ModuleOp m, StringRef path, StringRef name, WeightStrip *ws = nullptr) {
  llvm::LLVMContext lctx;
  std::unique_ptr<llvm::Module> lm = translateModuleToLLVMIR(m, lctx, name);
  if (!lm) { llvm::errs() << "hwacha-mlir: translation to LLVM IR failed (" << name << ")\n"; return false; }
  lm->setTargetTriple(llvm::Triple(RISCV_TRIPLE));
  lm->setDataLayout(RISCV_DL);
  std::error_code ec;
  if (ws) {
    const uint64_t THRESH = 256;   // leave small scalar constants inline
    const llvm::DataLayout &DL = lm->getDataLayout();
    for (llvm::GlobalVariable &G : lm->globals()) {
      if (!G.isConstant() || !G.hasInitializer()) continue;
      llvm::Constant *init = G.getInitializer();
      if (DL.getTypeAllocSize(init->getType()) < THRESH) continue;
      if (!llvm::isa<llvm::ConstantDataSequential, llvm::ConstantAggregateZero, llvm::ConstantArray>(init)) continue;
      if (!ws->done.insert(G.getName()).second) {   // already written from the other module: just externalize
        G.setInitializer(nullptr); G.setLinkage(llvm::GlobalValue::ExternalLinkage); G.setDSOLocal(false); continue;
      }
      uint64_t align = G.getAlign().value_or(llvm::Align(1)).value();
      if (align < 64) align = 64;
      int64_t wrote = streamConstBytes(init, DL, *ws->bin);
      if (wrote < 0) { llvm::errs() << "hwacha-mlir: cannot serialize " << G.getName() << ", leaving inline\n"; ws->done.erase(G.getName()); continue; }
      *ws->asmf << ".balign " << align << "\n.globl " << G.getName() << "\n" << G.getName() << ":\n"
                << ".incbin \"" << ws->binName << "\", " << ws->off << ", " << wrote << "\n";
      ws->off += wrote; ws->bytes += wrote; ws->n++;
      G.setInitializer(nullptr); G.setLinkage(llvm::GlobalValue::ExternalLinkage); G.setDSOLocal(false);
    }
  }
  llvm::raw_fd_ostream os(path, ec);
  if (ec) { llvm::errs() << "hwacha-mlir: " << path << ": " << ec.message() << "\n"; return false; }
  lm->print(os, nullptr);
  return true;
}

int main(int argc, char **argv) {
  ParseCommandLineOptions(argc, argv, "hwacha-mlir: MLIR gpu-dialect front end for hwacha-cc\n");
  if (MemRefConv != "bare" && MemRefConv != "desc") { llvm::errs() << "hwacha-mlir: --memref must be bare or desc\n"; return 1; }
  registerAllPasses();
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerAllToLLVMIRTranslations(registry);
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();
  OwningOpRef<ModuleOp> mod = parseSourceFile<ModuleOp>(InputFile, &ctx);
  if (!mod) return 1;
  ModuleOp m = *mod;
  std::string bareOpt = MemRefConv == "bare" ? "{use-bare-ptr-memref-call-conv=1}" : "";

  // bufferized torch-mlir output: memref.copy would lower to a runtime call (memrefCopy) -> make it a
  // linalg.copy kernel; cf.assert (shape checks) would lower to puts/abort -> drop it
  {
    SmallVector<memref::CopyOp> copies; m.walk([&](memref::CopyOp c) { copies.push_back(c); });
    for (memref::CopyOp c : copies) { OpBuilder b(c); linalg::CopyOp::create(b, c.getLoc(), ValueRange{c.getSource()}, ValueRange{c.getTarget()}); c.erase(); }
    SmallVector<cf::AssertOp> asserts; m.walk([&](cf::AssertOp a) { asserts.push_back(a); });
    for (cf::AssertOp a : asserts) a.erase();
  }
  // an embedded transform script (nested module with transform.with_named_sequence) is applied first
  bool hasTransform = false;
  for (Operation &op : *m.getBody()) if (op.hasAttr("transform.with_named_sequence")) hasTransform = true;
  if (hasTransform) {
    if (!runPipeline(m, "transform-interpreter")) return 1;
    for (Operation &op : llvm::make_early_inc_range(*m.getBody())) if (op.hasAttr("transform.with_named_sequence")) op.erase();
  }
  if (!hasGpuModule(m)) {
    if (!NoConvLib && !lowerConvs(m)) return 1;
    if (!runPipeline(m, "scf-forall-to-parallel,linalg-generalize-named-ops")) return 1;
    if (FuseGenerics) collapseGenerics(m);
    if (PrintMLIR) { llvm::errs() << "// ---- after collapseGenerics\n"; m.print(llvm::errs()); llvm::errs() << "\n"; }
    if (!runPipeline(m, "convert-linalg-to-parallel-loops,func.func(expand-strided-metadata,fold-memref-alias-ops,canonicalize)")) return 1;   // subviews from tiling folded into the accesses (bare pointers need identity layouts)
    if (!nestedParallelToFor(m)) return 1;
    if (CollapseAll) { if (!NoCollapse) collapseParallel(m); }
    else laneInnermost(m);
    mapParallel(m);
    if (!runPipeline(m, "func.func(convert-parallel-loops-to-gpu)")) return 1;
    sinkConstants(m);
    unrollSmall(m);   // before outlining: the loop bounds are still constants here
    if (!runPipeline(m, "gpu-kernel-outlining")) return 1;
    uniqueKernelNames(m);
  } else unrollSmall(m);
  if (!runPipeline(m, "lower-affine,convert-scf-to-cf")) return 1;

  // optional weight stripping shared across the kernel and host modules
  WeightStrip wsStore;
  WeightStrip *ws = nullptr;
  if (!WeightsBin.empty()) {
    std::error_code ecb, ecs;
    wsStore.bin = std::make_unique<llvm::raw_fd_ostream>(WeightsBin, ecb, llvm::sys::fs::OF_None);
    if (ecb) { llvm::errs() << "hwacha-mlir: " << WeightsBin << ": " << ecb.message() << "\n"; return 1; }
    std::string sPath = WeightsBin + ".S";
    wsStore.asmf = std::make_unique<llvm::raw_fd_ostream>(sPath, ecs, llvm::sys::fs::OF_Text);
    if (ecs) { llvm::errs() << "hwacha-mlir: " << sPath << ": " << ecs.message() << "\n"; return 1; }
    wsStore.binName = std::string(llvm::sys::path::filename(WeightsBin));
    *wsStore.asmf << "// generated by hwacha-mlir: constant globals stripped into " << wsStore.binName
                  << " (referenced by .incbin)\n.section .rodata\n";
    ws = &wsStore;
  }

  // kernels
  {
    OwningOpRef<ModuleOp> kc = m.clone();
    ModuleOp k = *kc;
    if (!runPipeline(k, "gpu.module(convert-gpu-to-nvvm" + bareOpt + "),reconcile-unrealized-casts")) return 1;
    SmallVector<Operation *> keep, drop;
    for (Operation &op : llvm::make_early_inc_range(*k.getBody())) {
      if (auto gm = dyn_cast<gpu::GPUModuleOp>(op)) {
        for (Operation &inner : llvm::make_early_inc_range(*gm.getBody()))
          if (!inner.hasTrait<OpTrait::IsTerminator>()) inner.moveBefore(gm);
        gm.erase();
      } else if (!isa<LLVM::GlobalOp, LLVM::LLVMFuncOp>(op)) op.erase();   // host-side leftovers
    }
    k->removeAttr("gpu.container_module");
    if (PrintMLIR) { llvm::errs() << "// ---- kernel module\n"; k.print(llvm::errs()); llvm::errs() << "\n"; }
    if (!emit(k, OutputFile, "hwacha-kernels", ws)) return 1;
  }
  // host: the kernels only need to exist for gpu.launch_func's symbol check, so gut their bodies
  // (workgroup memrefs etc. would not survive the host-side conversions)
  if (!HostFile.empty()) {
    m.walk([&](gpu::GPUFuncOp f) {
      Block &entry = f.getBody().front();
      for (Block &b : llvm::make_early_inc_range(llvm::drop_begin(f.getBody()))) b.dropAllDefinedValueUses(), b.erase();
      for (Operation &op : llvm::make_early_inc_range(entry)) op.dropAllUses(), op.erase();
      OpBuilder rb(&entry, entry.end());
      gpu::ReturnOp::create(rb, f.getLoc());
    });
    if (!runPipeline(m, "func.func(fold-memref-alias-ops,expand-strided-metadata),convert-math-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm" + bareOpt +
                        ",convert-index-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm")) return 1;
    if (!lowerLaunches(m)) return 1;
    for (Operation &op : llvm::make_early_inc_range(*m.getBody())) if (isa<gpu::GPUModuleOp>(op)) op.erase();
    m->removeAttr("gpu.container_module");
    if (!runPipeline(m, "reconcile-unrealized-casts")) return 1;
    if (!emit(m, HostFile, "hwacha-host", ws)) return 1;
  }
  if (ws && PrintMLIR) llvm::errs() << "// ---- weights: stripped " << ws->n << " globals (" << ws->bytes
                                    << " bytes) to " << WeightsBin << "\n";
  return 0;
}

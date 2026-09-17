// hwacha-mlir: MLIR front end of hwacha-cc.
//   in.mlir (gpu dialect, or linalg/scf.parallel over memrefs) -> kernels as LLVM IR for hwacha-cc
//                                                              -> host as LLVM IR (gpu.launch_func lowered)
// Pipeline: [linalg -> parallel loops -> collapse to 1-D -> map -> gpu -> outline] -> lower-affine ->
// scf-to-cf -> {kernel: convert-gpu-to-nvvm; host: *-to-llvm, launch_func -> call <kernel>_ct}.
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace llvm::cl;

static opt<std::string> InputFile(Positional, desc("<input .mlir>"), Required);
static opt<std::string> OutputFile("o", desc("kernel LLVM IR output (.ll)"), init("-"));
static opt<std::string> HostFile("host", desc("host LLVM IR output (.ll); empty = do not emit the host"));
static opt<std::string> MemRefConv("memref", desc("memref calling convention on both sides: bare (static shapes, default) or desc"), init("bare"));
static opt<bool> NoCollapse("no-collapse", desc("do not collapse multi-dimensional scf.parallel loops to 1-D"));
static opt<bool> CollapseAll("collapse-all", desc("collapse every dimension of a parallel loop into the lane dimension (old behaviour); default: innermost dimension = lanes, the others become control-thread loops"));
static opt<bool> NoConvLib("no-conv-lib", desc("do not lower linalg.conv_2d_nchw_fchw to the hwlib kernels (conv3x3 / conv3x3_s2 / conv1x1)"));
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
    for (unsigned k = n; k >= 2; --k) {
      ReassociationIndices grp;
      bool par = true;
      for (unsigned d = n - k; d < n; ++d) { grp.push_back(d); if (its[d] != utils::IteratorType::parallel) par = false; }
      if (!par) continue;
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
// The op that last wrote `buf` before `before` in the same block, if it made it all zeros: a linalg.fill
// of 0.0, or a linalg.copy whose source is itself zero-initialized. nullptr otherwise.
static Operation *zeroInitWriter(Value buf, Operation *before) {
  for (Operation *op = before->getPrevNode(); op; op = op->getPrevNode()) {
    if (!llvm::is_contained(op->getOperands(), buf)) continue;
    // reads of the buffer (copy source, linalg input) keep it zero
    if (auto lo = dyn_cast<linalg::LinalgOp>(op)) {
      bool writes = false;
      for (OpOperand &o : lo->getOpOperands()) if (o.get() == buf && lo.isDpsInit(&o)) writes = true;
      if (!writes) continue;
    }
    if (auto f = dyn_cast<linalg::FillOp>(op)) {
      if (f.getOutputs()[0] != buf) return nullptr;
      auto cst = f.getInputs()[0].getDefiningOp<arith::ConstantOp>();
      if (!cst) return nullptr;
      auto fa = dyn_cast<FloatAttr>(cst.getValue());
      return fa && fa.getValue().isZero() ? op : nullptr;
    }
    if (auto c = dyn_cast<linalg::CopyOp>(op)) {
      if (c.getOutputs()[0] != buf) return nullptr;
      return zeroInitWriter(c.getInputs()[0], op) ? op : nullptr;
    }
    return nullptr;   // any other use (read or write) in between: give up
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
  Type ints9[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32};
  LLVM::LLVMFuncOp c3 = declareFn(m, "conv3x3_ct", ints9), c31 = declareFn(m, "conv3x3_1_ct", ints9);
  Type ints11[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32, i32, i32, i32};
  LLVM::LLVMFuncOp c3s2 = declareFn(m, "conv3x3_s2_ct", ints11);
  Type ints8[] = {i64, ptrTy, ptrTy, ptrTy, ptrTy, i32, i32, i32};
  LLVM::LLVMFuncOp c1 = declareFn(m, "conv1x1_ct", ints8), c11 = declareFn(m, "conv1x1_1_ct", ints8);
  Type intsU[] = {i64, ptrTy, ptrTy, i32, i32, i32, i32};
  LLVM::LLVMFuncOp unpad = declareFn(m, "unpad_ct", intsU);
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
    enum { K3S1, K3S2, K1S1 } kind;
    if (kh == 3 && kw == 3 && stride == 1 && Hi == Ho + 2 && Wi == Wo + 2) kind = K3S1;
    else if (kh == 3 && kw == 3 && stride == 2 && Hi == 2 * Ho + 2 && Wi == 2 * Wo + 2 && O % 4 == 0) kind = K3S2;
    else if (kh == 1 && kw == 1 && stride == 1 && Hi == Ho && Wi == Wo) kind = K1S1;
    else { if (PrintMLIR) llvm::errs() << "// conv-lib: unsupported shape " << conv << "\n"; continue; }
    // the output must hold zeros: torch-mlir fills it (linalg.fill 0), or bufferization copied a shared
    // zero buffer into it (linalg.copy from a filled buffer). That last writer is dropped: the kernels
    // write every element.
    Operation *init = zeroInitWriter(conv.getOutputs()[0], conv);
    if (!init) { if (PrintMLIR) llvm::errs() << "// conv-lib: output not zero-initialized before " << conv << "\n"; continue; }
    Value zeroCst = isa<linalg::FillOp>(init) ? cast<linalg::FillOp>(init).getInputs()[0]
                                              : cast<linalg::FillOp>(zeroInitWriter(cast<linalg::CopyOp>(init).getInputs()[0], init)).getInputs()[0];
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
    // zero bias vector (the kernels add b[oc])
    Value zeros = memref::AllocOp::create(b, loc, MemRefType::get({512}, f32));
    linalg::FillOp::create(b, loc, ValueRange{zeroCst}, ValueRange{zeros});
    Value zp = ptrOf(zeros);
    if (kind == K1S1) {
      int64_t plane = Ho * Wo, oc = 0;
      for (; oc + 8 <= O; oc += 8) LLVM::CallOp::create(b, loc, c1, ValueRange{i64c(plane), xp, wp, zp, yp, i32c(C), i32c(plane), i32c(oc)});
      for (; oc < O; oc++) LLVM::CallOp::create(b, loc, c11, ValueRange{i64c(plane), xp, wp, zp, yp, i32c(C), i32c(plane), i32c(oc)});
    } else {
      int64_t plane = Hi * Wi, Wp = Wi;
      int64_t Hp2 = Ho + 2, Wp2 = Wo + 2, plane2 = Hp2 * Wp2;
      int64_t oplane = kind == K3S1 ? plane : plane2, oWp = kind == K3S1 ? Wp : Wp2;
      int64_t guard = 2 * oWp + 8;
      Value scratch = memref::AllocOp::create(b, loc, MemRefType::get({O * oplane + 2 * guard}, f32));
      Value sp = LLVM::GEPOp::create(b, loc, ptrTy, f32, ptrOf(scratch), ArrayRef<LLVM::GEPArg>{(int32_t)guard});
      if (kind == K3S1) {
        int64_t oc = 0;
        for (; oc + 8 <= O; oc += 8) LLVM::CallOp::create(b, loc, c3, ValueRange{i64c(plane), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(oc)});
        for (; oc < O; oc++) LLVM::CallOp::create(b, loc, c31, ValueRange{i64c(plane), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(oc)});
      } else {
        for (int64_t oc = 0; oc < O; oc += 4) LLVM::CallOp::create(b, loc, c3s2, ValueRange{i64c(plane2), xp, wp, zp, sp, i32c(C), i32c(plane), i32c(Wp), i32c(plane2), i32c(Wp2), i32c(oc)});
      }
      LLVM::CallOp::create(b, loc, unpad, ValueRange{i64c(O * Ho * Wo), sp, yp, i32c(oplane), i32c(oWp), i32c(Ho), i32c(Wo)});
    }
    init->erase();
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

static bool emit(ModuleOp m, StringRef path, StringRef name) {
  llvm::LLVMContext lctx;
  std::unique_ptr<llvm::Module> lm = translateModuleToLLVMIR(m, lctx, name);
  if (!lm) { llvm::errs() << "hwacha-mlir: translation to LLVM IR failed (" << name << ")\n"; return false; }
  lm->setTargetTriple(llvm::Triple(RISCV_TRIPLE));
  lm->setDataLayout(RISCV_DL);
  std::error_code ec;
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
    for (memref::CopyOp c : copies) { OpBuilder b(c); linalg::CopyOp::create(b, c.getLoc(), c.getSource(), c.getTarget()); c.erase(); }
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
    collapseGenerics(m);
    if (PrintMLIR) { llvm::errs() << "// ---- after collapseGenerics\n"; m.print(llvm::errs()); llvm::errs() << "\n"; }
    if (!runPipeline(m, "convert-linalg-to-parallel-loops,func.func(fold-memref-alias-ops)")) return 1;   // subviews from tiling folded into the accesses (bare pointers need identity layouts)
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
    if (!emit(k, OutputFile, "hwacha-kernels")) return 1;
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
    if (!emit(m, HostFile, "hwacha-host")) return 1;
  }
  return 0;
}

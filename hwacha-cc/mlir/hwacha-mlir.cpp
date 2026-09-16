// hwacha-mlir: MLIR front end of hwacha-cc.
//   in.mlir (gpu dialect, or linalg/scf.parallel over memrefs) -> kernels as LLVM IR for hwacha-cc
//                                                              -> host as LLVM IR (gpu.launch_func lowered)
// Pipeline: [linalg -> parallel loops -> collapse to 1-D -> map -> gpu -> outline] -> lower-affine ->
// scf-to-cf -> {kernel: convert-gpu-to-nvvm; host: *-to-llvm, launch_func -> call <kernel>_ct}.
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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
    if (!runPipeline(m, "scf-forall-to-parallel,convert-linalg-to-parallel-loops,func.func(fold-memref-alias-ops)")) return 1;   // subviews from tiling folded into the accesses (bare pointers need identity layouts)
    if (!nestedParallelToFor(m)) return 1;
    if (!NoCollapse) collapseParallel(m);
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

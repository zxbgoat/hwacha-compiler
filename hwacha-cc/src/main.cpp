// hwacha-cc: OpenCL (via clang LLVM IR) -> Hwacha vector-fetch code.
//   hwacha-cc kernels.ll -o kernels.s        emit assembly (worker threads + control threads)
//   hwacha-cc kernels.ll --analyze           print the analysis report only
#include "Analysis.h"
#include "CodeGen.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/LineIterator.h"

using namespace llvm;
static cl::opt<std::string> InputFile(cl::Positional, cl::desc("<input .ll/.bc>"), cl::Required);
static cl::opt<std::string> OutputFile("o", cl::desc("output assembly file"), cl::value_desc("file"));
static cl::opt<bool> AnalyzeOnly("analyze", cl::desc("print analysis report and exit"));
static cl::opt<bool> KeepTemps("keep", cl::desc("keep the control-thread .ll/.s temporaries"));
static cl::opt<bool> Stats("kstats", cl::desc("print per-kernel register and instruction statistics"));
static cl::opt<bool> NoV32("no-v32", cl::desc("ablation: all vector registers 64-bit"));
static cl::opt<bool> NoSkip("no-skip", cl::desc("ablation: no consensual jumps around inactive blocks"));
static cl::opt<bool> NoCoalesce("no-coalesce", cl::desc("ablation: no phi coalescing"));
static cl::opt<bool> ScalarFP("scalar-fp", cl::desc("allow uniform floating-point ops in vs registers (Spike only)"));
static cl::opt<bool> VerboseOpt("verbose", cl::desc("print code generation diagnostics"));
static cl::opt<bool> NoCTLoops("no-ct-loops", cl::desc("do not run uniform loops on the control thread (ablation)"));
static cl::opt<bool> SubwordRMW("subword-rmw", cl::desc("lower masked sub-word stores to load/select/store (workaround for the unpatched Hwacha RTL store-credit bug)"));
static cl::opt<bool> GPUBlock1("gpu-block1", cl::desc("GPU-dialect input: treat every kernel as launched with block size 1 (block id = work-item id)"));
static cl::opt<bool> GPUNoOpt("gpu-no-opt", cl::desc("GPU-dialect input: skip the -O2 pipeline after adaptation"));
static cl::opt<bool> AssumeNoAlias("assume-noalias", cl::desc("treat every pointer kernel argument as restrict (lets accumulators stay in registers across control-thread loops)"));
static cl::opt<bool> FPContract("fp-contract", cl::desc("fuse fmul+fadd into vfmadd (like -ffp-contract=fast; changes rounding)"));
static cl::opt<std::string> HostFile("host", cl::desc("host LLVM IR (from hwacha-mlir --host) to optimize, compile with llc and append to the output"));
static cl::opt<std::string> LLCPath("llc", cl::desc("path to llc"), cl::init(LLC_DEFAULT_PATH));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "hwacha-cc\n");
  LLVMContext Ctx; SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFile, Err, Ctx);
  if (!M) { Err.print(argv[0], errs()); return 1; }
  if (AssumeNoAlias)
    for (Function &F : *M)
      if (F.getCallingConv() == CallingConv::PTX_Kernel || hwacha::isKernel(F))
        for (Argument &A : F.args()) if (A.getType()->isPointerTy()) A.addAttr(Attribute::NoAlias);
  bool FromGPU = false;
  if (!hwacha::adaptGPUModule(*M, GPUBlock1, GPUNoOpt, errs(), FromGPU)) return 1;
  if (FromGPU && KeepTemps) { std::error_code EC; raw_fd_ostream O((OutputFile.empty() ? std::string("out") : OutputFile.substr(0, OutputFile.rfind('.'))) + ".gpu.ll", EC); if (!EC) M->print(O, nullptr); }

  auto CT = std::make_unique<Module>("hwacha-ct", Ctx);
  CT->setTargetTriple(M->getTargetTriple());
  CT->setDataLayout(M->getDataLayout());
  std::string WTText; raw_string_ostream WT(WTText);
  hwacha::CodeGenOptions Opts; Opts.Stats = Stats; Opts.NoV32 = NoV32; Opts.NoSkip = NoSkip; Opts.NoCoalesce = NoCoalesce; Opts.ScalarFP = ScalarFP; Opts.SubwordRMW = SubwordRMW; Opts.NoCTLoops = NoCTLoops; Opts.Verbose = VerboseOpt;
  int n = 0;
  for (Function &F : *M) {
    if (!hwacha::isKernel(F)) continue;
    hwacha::expandAbsI(F); hwacha::expandLogExpM1Pow(F); hwacha::expandTanhf(F); hwacha::expandFloorf(F); hwacha::expandErff(F); hwacha::expandLogf(F);
    hwacha::expandExpf(F);
    if (FPContract) hwacha::contractFMA(F);
    hwacha::prepareKernel(F);
    hwacha::KernelAnalysis KA(F);
    if (AnalyzeOnly) { KA.print(outs()); n++; continue; }
    if (!hwacha::generateKernel(F, KA, *CT, WT, errs(), Opts)) return 1;
    n++;
  }
  if (!n) { errs() << "no kernels found\n"; return 1; }
  if (AnalyzeOnly) return 0;
  if (verifyModule(*CT, &errs())) { if (KeepTemps) CT->print(errs(), nullptr); return 1; }
  if (!HostFile.empty()) {   // host code from hwacha-mlir: same -O2, linked into the control-thread module
    std::unique_ptr<Module> HM = parseIRFile(HostFile, Err, Ctx);
    if (!HM) { Err.print(argv[0], errs()); return 1; }
    for (Function &F : *HM) if (!F.isDeclaration()) { hwacha::expandAbsI(F); hwacha::expandLogExpM1Pow(F); hwacha::expandTanhf(F); hwacha::expandFloorf(F); hwacha::expandErff(F); hwacha::expandLogf(F); hwacha::expandExpf(F); }
    hwacha::optimizeModule(*HM);
    for (StringRef G : {"hwacha_group_size", "hwacha_grid_size"})   // written by the lowered launches
      if (!CT->getGlobalVariable(G, true)) {
        Type *I64 = Type::getInt64Ty(Ctx);
        auto *GV = new GlobalVariable(*CT, I64, false, GlobalValue::WeakAnyLinkage, ConstantInt::get(I64, 0), G);
        GV->setAlignment(Align(8));
      }
    if (Linker::linkModules(*CT, std::move(HM))) { errs() << "hwacha-cc: cannot link " << HostFile << "\n"; return 1; }
  }

  std::string Base = OutputFile.empty() ? "out" : OutputFile.substr(0, OutputFile.rfind('.'));
  std::string CTll = Base + ".ct.ll", CTs = Base + ".ct.s";
  { std::error_code EC; raw_fd_ostream O(CTll, EC); if (EC) { errs() << EC.message() << "\n"; return 1; } CT->print(O, nullptr); }
  auto runLLC = [&](const std::string &In, const std::string &Out) {
    std::string Msg;
    int RC = sys::ExecuteAndWait(LLCPath, {LLCPath, "-O2", "-no-integrated-as", "--code-model=medium", "-mtriple=riscv64-unknown-elf", "-mattr=+m,+a,+f,+d", "-target-abi", "lp64d", In, "-o", Out}, std::nullopt, {}, 0, 0, &Msg);
    if (RC) errs() << "llc failed: " << Msg << "\n";
    return RC == 0;
  };
  if (!runLLC(CTll, CTs)) return 1;
  auto CTBuf = MemoryBuffer::getFile(CTs);
  if (!CTBuf) { errs() << "cannot read " << CTs << "\n"; return 1; }
  std::error_code EC;
  raw_fd_ostream O(OutputFile.empty() ? "-" : OutputFile.getValue(), EC);
  if (EC) { errs() << EC.message() << "\n"; return 1; }
  O << "# generated by hwacha-cc from " << InputFile << "\n";
  O << "# ---- worker threads (vector-fetch blocks) ----\n" << WTText;
  O << "\n# ---- control threads (from llc) ----\n";
  // binutils 2.29 does not understand these directives / new mnemonics
  auto append = [&](MemoryBuffer &Buf) {
    for (line_iterator L(Buf, false); !L.is_at_end(); ++L) {
      StringRef S = L->trim();
      if (S.starts_with(".attribute") || S.starts_with(".option")) continue;
      std::string Line = L->str();
      size_t p;
      while ((p = Line.find("fmv.x.w")) != std::string::npos) Line.replace(p, 7, "fmv.x.s");
      while ((p = Line.find("fmv.w.x")) != std::string::npos) Line.replace(p, 7, "fmv.s.x");
      O << Line << "\n";
    }
  };
  append(**CTBuf);
  if (!KeepTemps) { sys::fs::remove(CTll); sys::fs::remove(CTs); }
  return 0;
}

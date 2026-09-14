#!/bin/bash
# MLIR (gpu dialect, or linalg/scf.parallel on memrefs) -> LLVM IR of the kernels, ready for hwacha-cc.
#   usage: mlir-to-ll.sh in.mlir out.ll [extra mlir-opt passes inserted before the NVVM lowering]
#   env:   MEMREF_CONV=bare|desc  kernel memref calling convention (bare needs static shapes; default bare)
#          COLLAPSE=0,1           collapse these scf.parallel dims into one before mapping (linalg input)
#          MLIR_BIN               directory holding mlir-opt / mlir-translate (default ~/miniforge3/bin)
set -e
MLIR_BIN=${MLIR_BIN:-$HOME/miniforge3/bin}
in=$1; out=$2; shift 2
front=()
if ! grep -q "gpu.module" "$in"; then
  front=(--convert-linalg-to-parallel-loops)
  [ -n "$COLLAPSE" ] && front+=("--test-scf-parallel-loop-collapsing=collapsed-indices-0=$COLLAPSE")
  front+=(--gpu-map-parallel-loops --convert-parallel-loops-to-gpu --gpu-kernel-outlining)
fi
nvvm=--convert-gpu-to-nvvm
[ "${MEMREF_CONV:-bare}" = bare ] && nvvm="--convert-gpu-to-nvvm=use-bare-ptr-memref-call-conv=1"
"$MLIR_BIN/mlir-opt" "$in" "${front[@]}" "$@" --lower-affine --convert-scf-to-cf "$nvvm" --reconcile-unrealized-casts -o "$out.nvvm.mlir"
python3 "$(dirname "$0")/mlir-extract-kernels.py" "$out.nvvm.mlir" "$out.kern.mlir"
"$MLIR_BIN/mlir-translate" --mlir-to-llvmir "$out.kern.mlir" -o "$out"

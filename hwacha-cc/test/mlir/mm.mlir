func.func @matmul(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64x64xf32>) {
  linalg.matmul ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>) outs(%c : memref<64x64xf32>)
  return
}

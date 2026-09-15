// linalg.matmul with an embedded transform script: rows tiled by 4 and iterators interchanged to
// (j, k, i') so that lanes run over columns, k is a control-thread loop and the 4-row block is
// unrolled into 4 accumulators (register blocking); the row-tile loop stays on the host (16 launches).
func.func @matmul_blk(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64x64xf32>) {
  linalg.matmul ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>) outs(%c : memref<64x64xf32>)
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gen = transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %gen tile_sizes [4, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %inter = transform.structured.interchange %tiled iterator_interchange = [1, 2, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

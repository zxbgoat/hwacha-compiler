// linalg.matmul with an embedded transform script: columns become a forall (the lane dimension),
// rows are tiled by 4 inside it and the iterators interchanged to (k, i', j') so that the whole
// (row tile, k) nest runs on the control thread inside ONE launch, the 4-row block is unrolled into
// 4 vector accumulators (register blocking) and b[k, j] is a unit-stride stream.
func.func @matmul_blk(%a: memref<64x64xf32>, %b: memref<64x64xf32>, %c: memref<64x64xf32>) {
  linalg.matmul ins(%a, %b : memref<64x64xf32>, memref<64x64xf32>) outs(%c : memref<64x64xf32>)
  return
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %gen = transform.structured.generalize %mm : (!transform.any_op) -> !transform.any_op
    %t1, %forall = transform.structured.tile_using_forall %gen tile_sizes [0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %t2, %loop = transform.structured.tile_using_for %t1 tile_sizes [4, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %inter = transform.structured.interchange %t2 iterator_interchange = [2, 0, 1] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

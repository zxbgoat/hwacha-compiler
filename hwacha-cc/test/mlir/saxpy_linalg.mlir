func.func @saxpy(%a: f32, %x: memref<?xf32>, %y: memref<?xf32>) {
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
                  iterator_types = ["parallel"]}
    ins(%x : memref<?xf32>) outs(%y : memref<?xf32>) {
  ^bb0(%xv: f32, %yv: f32):
    %m = arith.mulf %a, %xv : f32
    %r = arith.addf %m, %yv : f32
    linalg.yield %r : f32
  }
  return
}

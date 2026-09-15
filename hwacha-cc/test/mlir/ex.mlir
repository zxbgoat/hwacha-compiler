// math ops: p = exp(fma(sqrt(|v|), v, |v|) * 0.1)  (exp is expanded by hwacha-cc into vector arithmetic)
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @ex(%p: memref<4096xf32>) kernel {
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bd = gpu.block_dim x
      %t = arith.muli %bid, %bd : index
      %g = arith.addi %t, %tid : index
      %v = memref.load %p[%g] : memref<4096xf32>
      %a = math.absf %v : f32
      %s = math.sqrt %a : f32
      %f = math.fma %s, %v, %a : f32
      %c = arith.constant 0.1 : f32
      %m = arith.mulf %f, %c : f32
      %e = math.exp %m : f32
      memref.store %e, %p[%g] : memref<4096xf32>
      gpu.return
    }
  }
  func.func @run_ex(%p: memref<4096xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    gpu.launch_func @kernels::@ex blocks in (%c32, %c1, %c1) threads in (%c128, %c1, %c1) args(%p : memref<4096xf32>)
    return
  }
}

module attributes {gpu.container_module} {
  gpu.module @k {
    gpu.func @ex(%p: memref<4096xf32>) kernel {
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bd = gpu.block_dim x
      %t = arith.muli %bid, %bd : index
      %g = arith.addi %t, %tid : index
      %v = memref.load %p[%g] : memref<4096xf32>
      %e = math.absf %v : f32
      %s = math.sqrt %e : f32
      %f = math.fma %s, %v, %e : f32
      memref.store %f, %p[%g] : memref<4096xf32>
      gpu.return
    }
  }
}

module attributes {gpu.container_module} {
  gpu.module @k {
    gpu.func @wg(%p: memref<4096xf32>) workgroup(%s: memref<256xf32, #gpu.address_space<workgroup>>) kernel {
      %tid = gpu.thread_id x
      %bid = gpu.block_id x
      %bd = gpu.block_dim x
      %t = arith.muli %bid, %bd : index
      %g = arith.addi %t, %tid : index
      %v = memref.load %p[%g] : memref<4096xf32>
      memref.store %v, %s[%tid] : memref<256xf32, #gpu.address_space<workgroup>>
      gpu.barrier
      %c0 = arith.constant 0 : index
      %w = memref.load %s[%c0] : memref<256xf32, #gpu.address_space<workgroup>>
      memref.store %w, %p[%g] : memref<4096xf32>
      gpu.return
    }
  }
}

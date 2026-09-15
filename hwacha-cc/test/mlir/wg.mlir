// workgroup memory + gpu.barrier: p[g] = p[first element of the block] (block size 256)
module attributes {gpu.container_module} {
  gpu.module @kernels {
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
  func.func @run_wg(%p: memref<4096xf32>) {
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    gpu.launch_func @kernels::@wg blocks in (%c16, %c1, %c1) threads in (%c256, %c1, %c1) args(%p : memref<4096xf32>)
    return
  }
}

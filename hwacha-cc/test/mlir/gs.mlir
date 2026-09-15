// grid-stride loop (the other CUDA idiom): every work-item handles i = gid, gid+stride, ... with
// stride = gridDim.x*blockDim.x; launched with far fewer work-items than elements.
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @gs(%x: memref<4096xf32>, %y: memref<4096xf32>, %n: index) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %gdim = gpu.grid_dim x
      %t = arith.muli %bid, %bdim : index
      %gid = arith.addi %t, %tid : index
      %stride = arith.muli %gdim, %bdim : index
      %two = arith.constant 2.0 : f32
      scf.for %i = %gid to %n step %stride {
        %v = memref.load %x[%i] : memref<4096xf32>
        %r = arith.mulf %v, %two : f32
        memref.store %r, %y[%i] : memref<4096xf32>
      }
      gpu.return
    }
  }
  func.func @run_gs(%x: memref<4096xf32>, %y: memref<4096xf32>, %n: index) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    gpu.launch_func @kernels::@gs blocks in (%c4, %c1, %c1) threads in (%c64, %c1, %c1)
      args(%x : memref<4096xf32>, %y : memref<4096xf32>, %n : index)
    return
  }
}

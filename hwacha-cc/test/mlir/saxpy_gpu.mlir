// saxpy written directly as a gpu.func (the CUDA idiom): y[gid] = a*x[gid] + y[gid], gid = blockIdx.x*blockDim.x + threadIdx.x
module attributes {gpu.container_module} {
  gpu.module @kernels {
    gpu.func @saxpy(%a: f32, %x: memref<4096xf32>, %y: memref<4096xf32>, %n: i64) kernel {
      %bid = gpu.block_id x
      %bdim = gpu.block_dim x
      %tid = gpu.thread_id x
      %t = arith.muli %bid, %bdim : index
      %gid = arith.addi %t, %tid : index
      %gi = arith.index_cast %gid : index to i64
      %in = arith.cmpi ult, %gi, %n : i64
      scf.if %in {
        %xv = memref.load %x[%gid] : memref<4096xf32>
        %yv = memref.load %y[%gid] : memref<4096xf32>
        %m = arith.mulf %a, %xv : f32
        %r = arith.addf %m, %yv : f32
        memref.store %r, %y[%gid] : memref<4096xf32>
      }
      gpu.return
    }
  }
}

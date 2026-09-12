// pathfinder prologue pieces: group-id based uniform base, __local pointer args, barrier
__kernel void k1(__global int *gpuSrc, __global int *out, int cols, int border, __local int *prev) {
  int BLOCK_SIZE = get_local_size(0); int bx = get_group_id(0); int tx = get_local_id(0);
  int small = BLOCK_SIZE - 2; int blkX = small * bx - border; int xidx = blkX + tx;
  if (xidx >= 0 && xidx <= cols - 1) prev[tx] = gpuSrc[xidx];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (xidx >= 0 && xidx <= cols - 1) out[xidx] = prev[tx] + 1;
}
__kernel void k2(__global int *gpuSrc, __global int *out, int cols, int border, __local int *prev) {
  int BLOCK_SIZE = get_local_size(0); int bx = get_group_id(0); int tx = get_local_id(0);
  int small = BLOCK_SIZE - 2; int blkX = small * bx - border; int xidx = blkX + tx;
  if (xidx >= 0 && xidx <= cols - 1) out[xidx] = gpuSrc[xidx] + 1;      // no local memory at all
}
__kernel void k3(__global int *gpuSrc, __global int *out, int cols) {
  int gid = get_global_id(0); int bx = get_group_id(0); int tx = get_local_id(0);
  if (gid < cols) out[gid] = gpuSrc[gid] + bx * 1000 + tx;             // group id / local id as data
}
// k4: pathfinder's loop skeleton — result/prev locals, computed flag, barrier, break — minus the arithmetic
__kernel void k4(int iteration, __global int *gpuWall, __global int *gpuSrc, __global int *gpuResults, int cols, int startStep,
                 int border, __local int *prev, __local int *result) {
  int BLOCK_SIZE = get_local_size(0); int bx = get_group_id(0); int tx = get_local_id(0);
  int small = BLOCK_SIZE - iteration * 2; int blkX = small * bx - border; int blkXmax = blkX + BLOCK_SIZE - 1; int xidx = blkX + tx;
  int validXmin = (blkX < 0) ? -blkX : 0; int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
  bool isValid = tx >= validXmin && tx <= validXmax;
  if (xidx >= 0 && xidx <= cols-1) prev[tx] = gpuSrc[xidx];
  barrier(CLK_LOCAL_MEM_FENCE);
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (tx >= i+1 && tx <= BLOCK_SIZE-i-2 && isValid) { computed = true; result[tx] = prev[tx] + gpuWall[cols*(startStep+i)+xidx]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (i == iteration-1) break;
    if (computed) prev[tx] = result[tx];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (computed) gpuResults[xidx] = result[tx];
}
// k5: k4's prologue, dumping the intermediate quantities for every lane
__kernel void k5(int iteration, __global int *gpuSrc, __global int *dbg, int cols, int border, __local int *prev) {
  int BLOCK_SIZE = get_local_size(0); int bx = get_group_id(0); int tx = get_local_id(0);
  int small = BLOCK_SIZE - iteration * 2; int blkX = small * bx - border; int blkXmax = blkX + BLOCK_SIZE - 1; int xidx = blkX + tx;
  int validXmin = (blkX < 0) ? -blkX : 0; int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
  bool isValid = tx >= validXmin && tx <= validXmax;
  bool computed = false;
  if (tx >= 1 && tx <= BLOCK_SIZE-2 && isValid) computed = true;
  if (xidx >= 0 && xidx <= cols-1) { dbg[xidx*6+0] = validXmin; dbg[xidx*6+1] = validXmax; dbg[xidx*6+2] = isValid; dbg[xidx*6+3] = computed; dbg[xidx*6+4] = blkX; dbg[xidx*6+5] = small; }
}
// k6: boolean probes
__kernel void k6(__global int *dbg, int vmin_in, int vmax_in) {
  int tx = get_local_id(0); int gid = get_global_id(0);
  int vmin = vmin_in, vmax = vmax_in;
  bool b1 = tx >= vmin; bool b2 = tx <= vmax; bool b3 = b1 && b2;
  dbg[gid*4+0] = b1; dbg[gid*4+1] = b2; dbg[gid*4+2] = b3; dbg[gid*4+3] = (tx > 3) ? 7 : 9;
}
// k7: k5 with the pieces of isValid written out separately
__kernel void k7(int iteration, __global int *dbg, int cols, int border) {
  int BLOCK_SIZE = get_local_size(0); int bx = get_group_id(0); int tx = get_local_id(0);
  int small = BLOCK_SIZE - iteration * 2; int blkX = small * bx - border; int blkXmax = blkX + BLOCK_SIZE - 1; int xidx = blkX + tx;
  int validXmin = (blkX < 0) ? -blkX : 0; int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;
  bool b1 = tx >= validXmin; bool b2 = tx <= validXmax; bool isValid = b1 && b2;
  int gid = get_global_id(0);
  dbg[gid*4+0] = b1; dbg[gid*4+1] = b2; dbg[gid*4+2] = isValid; dbg[gid*4+3] = validXmin * 1000 + validXmax;
}

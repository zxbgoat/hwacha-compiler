// work-group reduction with local memory and barriers; the work-group is one stripmine chunk
__kernel __attribute__((reqd_work_group_size(256, 1, 1)))
void group_sum(__global const float *x, __global float *sums) {
  __local float tile[256];
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  tile[lid] = x[gid];
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = 128; s > 0; s >>= 1) {
    if (lid < s) tile[lid] += tile[lid + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0) sums[get_group_id(0)] = tile[0];
}
// neighbour exchange through local memory (no reqd size: group = whatever vl the hardware picks)
__kernel void shift_left(__global const float *x, __global float *y) {
  __local float buf[2048];
  int lid = get_local_id(0), gid = get_global_id(0), ls = get_local_size(0);
  buf[lid] = x[gid];
  barrier(CLK_LOCAL_MEM_FENCE);
  y[gid] = buf[(lid + 1) % ls];
}

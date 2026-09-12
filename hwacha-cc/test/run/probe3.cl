__kernel void st_ld(__global int *a, __global const int *idx, __global int *out) { int i = get_global_id(0); a[idx[i]] = i; out[i] = a[i] + 1; }          // scatter then unit load, same array
__kernel void ld_st(__global int *a, __global const int *idx, __global int *out) { int i = get_global_id(0); int v = a[i]; a[idx[i]] = v + 1; out[i] = v; } // unit load then scatter
__kernel void st_ld_loop(__global int *a, __global const int *idx, __global int *out, int iters) { int i = get_global_id(0); int s = 0;
  for (int k = 0; k < iters; k++) { a[idx[i]] = i + k; s += a[i]; } out[i] = s; }                                                                           // bfs-like: scatter/load in a loop
__kernel void bst_bld(__global char *m, __global const int *idx, __global int *out) { int i = get_global_id(0); m[idx[i]] = 1; out[i] = m[i]; }              // byte scatter then byte load

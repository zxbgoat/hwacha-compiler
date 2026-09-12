__kernel void sparse_st(__global int *out, __global const int *sel) { int i = get_global_id(0); if (sel[i]) out[i] = 5; }                                 // vsw, one active lane
__kernel void sparse_sc(__global int *out, __global const int *sel, __global const int *idx) { int i = get_global_id(0); if (sel[i]) out[idx[i]] = 5; }   // vsxw sparse
__kernel void sparse_bs(__global char *out, __global const int *sel) { int i = get_global_id(0); if (sel[i]) out[i] = 1; }                                // vsb sparse
__kernel void sparse_ld(__global int *out, __global const int *sel, __global const int *a) { int i = get_global_id(0); if (sel[i]) out[i] = a[i] + 1; }  // vlw + vsw sparse

__kernel void sparse_sxb(__global char *out, __global const int *sel, __global const int *idx) { int i = get_global_id(0); if (sel[i]) out[idx[i]] = 1; }   // byte scatter, 1 lane
__kernel void sparse_sh(__global short *out, __global const int *sel) { int i = get_global_id(0); if (sel[i]) out[i] = 1; }                                // halfword unit store, 1 lane
__kernel void sparse_bs3(__global char *out, __global const int *sel) { int i = get_global_id(0); if (sel[i]) out[i] = 1; }                               // byte unit store, lane 3 active
__kernel void half_bs(__global char *out) { int i = get_global_id(0); if (i & 1) out[i] = 1; }                                                            // byte unit store, every other lane

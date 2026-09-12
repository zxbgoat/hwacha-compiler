__kernel void bscatter(__global const int *idx, __global char *out) { int i = get_global_id(0); out[idx[i]] = (char)(i & 0x7f); }
__kernel void bgather(__global const int *idx, __global const char *tab, __global char *out) { int i = get_global_id(0); out[i] = tab[idx[i]]; }
__kernel void wscatter(__global const int *idx, __global int *out) { int i = get_global_id(0); out[idx[i]] = i; }

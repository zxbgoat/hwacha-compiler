__kernel void u8copy(__global const char *a, __global char *b) { int i = get_global_id(0); b[i] = a[i]; }              // vlb + vsb
__kernel void u8load(__global const char *a, __global int *b)  { int i = get_global_id(0); b[i] = a[i]; }              // vlb + vsw
__kernel void u8store(__global const int *a, __global char *b) { int i = get_global_id(0); b[i] = (char)a[i]; }        // vlw + vsb
__kernel void u16copy(__global const short *a, __global short *b) { int i = get_global_id(0); b[i] = a[i]; }           // vlh + vsh
__kernel void u8gath(__global const char *a, __global const int *idx, __global int *b) { int i = get_global_id(0); b[i] = a[idx[i]]; }  // vlxb

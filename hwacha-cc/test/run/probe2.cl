__kernel void uni_store(__global int *flag, __global char *bflag, __global int *out) { int i = get_global_id(0); *flag = 7; *bflag = 3; out[i] = i; }   // vssw + vssb
__kernel void mul64(__global const long *a, __global const long *b, __global long *c) { int i = get_global_id(0); c[i] = a[i] * b[i]; }              // vmul 64-bit
__kernel void dblfma(__global const double *x, __global double *y, double a) { int i = get_global_id(0); y[i] = a * x[i] + y[i]; }                    // vfmadd.d
__kernel void atom(__global const int *bin, __global int *hist) { int i = get_global_id(0); atomic_add(&hist[bin[i]], 1); }                          // vamoadd.w
__kernel void mul32(__global const int *a, __global const int *b, __global int *c) { int i = get_global_id(0); c[i] = a[i] * b[i]; }                  // vmulw

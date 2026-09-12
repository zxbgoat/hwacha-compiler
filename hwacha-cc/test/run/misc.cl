#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void hist(__global const uchar *pix, __global int *hist) {
  int i = get_global_id(0);
  atomic_add(&hist[pix[i]], 1);                     // uchar load (zext) + atomic add with divergent address
}
__kernel void widen(__global const short *a, __global const ushort *b, __global long *out) {
  int i = get_global_id(0);
  out[i] = (long)a[i] * (long)b[i];                 // sext and zext narrow loads, 64-bit multiply
}
__kernel void dbl(__global const double *x, __global double *y, double a) {
  int i = get_global_id(0);
  y[i] = a * x[i] + y[i];                           // double precision
}
__kernel void bytes(__global const uchar *a, __global uchar *b) {
  int i = get_global_id(0);
  b[i] = (uchar)(a[i] + 100);                       // byte store
}

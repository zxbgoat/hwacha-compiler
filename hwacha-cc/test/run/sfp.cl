// scalar (vs-destination) floating-point ops: compiled with --scalar-fp they go to Rocket's FPU via the RoCC port
__kernel void sfp_mul(__global const float *x, __global float *y, float a, float b) {
  int i = get_global_id(0);
  float k = a * b + 1.5f;          // uniform: vfmul.s / vfadd.s with vs destinations under --scalar-fp
  y[i] = x[i] * k;
}
__kernel void sfp_cvt(__global const int *x, __global float *y, int n) {
  int i = get_global_id(0);
  float fn = (float)n * 0.5f;      // uniform int->float conversion and multiply
  y[i] = (float)x[i] + fn;
}
__kernel void sfp_dbl(__global const double *x, __global double *y, double a) {
  int i = get_global_id(0);
  double k = a * a - 1.0;          // double precision scalar ops
  y[i] = x[i] + k;
}

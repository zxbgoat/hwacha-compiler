// loop-carried stream bases and runtime strides
__kernel void window(int m, __global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = 0;
  for (int k = 0; k < m; k++) s += x[i + k];        // base advances inside the vf block
  out[i] = s;
}
__kernel void strided(int ld, __global const float *a, __global float *out) {
  int i = get_global_id(0);
  out[i] = a[i * ld];                                // runtime (uniform) stride
}
__kernel void stencil(__global const float *x, __global float *out) {
  int i = get_global_id(0) + 1;
  out[i] = 0.25f * x[i - 1] + 0.5f * x[i] + 0.25f * x[i + 1];   // constant offsets: still plain streams
}
__kernel void matvec(int n, __global const float *A, __global const float *v, __global float *y) {
  int i = get_global_id(0);
  float s = 0;
  for (int j = 0; j < n; j++) s += A[i * n + j] * v[j];        // row stride n (runtime), v[j] uniform
  y[i] = s;
}

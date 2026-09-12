// kernels for cycle measurements on RTL: compiled vs hand-written vs scalar
__kernel void saxpy(float a, __global const float *x, __global float *y) {
  int i = get_global_id(0);
  y[i] = a * x[i] + y[i];
}
__kernel void clamp_scale(float a, __global const float *x, __global float *y) {
  int i = get_global_id(0);
  float v = x[i];
  if (v > 0.0f) y[i] = a * v; else y[i] = 0.0f;
}
__kernel void divloop(__global const int *n, __global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = 0;
  for (int k = 0; k < n[i]; k++) s += x[k];
  out[i] = s;
}
__kernel void stencil(__global const float *x, __global float *out) {
  int i = get_global_id(0) + 1;
  out[i] = 0.25f * x[i - 1] + 0.5f * x[i] + 0.25f * x[i + 1];
}
__kernel void gather(__global const int *idx, __global const float *tab, __global float *out) {
  int i = get_global_id(0);
  out[i] = tab[idx[i]];
}

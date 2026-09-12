// control-flow kernels for phase 2b
__kernel void cond_store(__global const int *x, __global int *y) {
  int i = get_global_id(0);
  if (x[i] > 0) y[i] = x[i] * 2;                       // divergent if with a store
}
__kernel void nested_if(__global const float *x, __global float *y) {
  int i = get_global_id(0);
  float v = x[i], r;
  if (v > 0.0f) { if (v > 10.0f) r = 10.0f; else r = v; }
  else          { r = -v; y[i + 0] = r; }               // else arm also stores
  y[i] = r * 2.0f;
}
__kernel void divloop(__global const int *n, __global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = 0;
  for (int k = 0; k < n[i]; k++) s += x[k];             // divergent trip count
  out[i] = s;
}
__kernel void uloop(int m, __global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = 0, v = x[i];
  for (int k = 0; k < m; k++) s = s * 0.5f + v;         // uniform trip count
  out[i] = s;
}
__kernel void divloop_vv(__global const int *n, __global int *out) {
  int i = get_global_id(0);
  int acc = i;
  for (int k = 0; k < n[i]; k++) acc = acc * 3 + k;     // loop-carried divergent value
  out[i] = acc;
}

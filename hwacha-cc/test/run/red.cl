#include "hwacha_builtins.h"
// per-group sum / max / int sum, written by lane 0 of the group
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void rsum(__global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = work_group_reduce_add(x[i]);
  if (get_local_id(0) == 0) out[get_group_id(0)] = s;
}
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void rmax(__global const float *x, __global float *out) {
  int i = get_global_id(0);
  float m = work_group_reduce_max(x[i]);
  if (get_local_id(0) == 0) out[get_group_id(0)] = m;
}
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void risum(__global const int *x, __global int *out) {
  int i = get_global_id(0);
  int s = work_group_reduce_add(x[i]);
  if (get_local_id(0) == 0) out[get_group_id(0)] = s;
}
// the result feeds vector code: y = x / sum(group); only lanes with x > 0 contribute (masked lanes give identity)
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void rnorm(__global const float *x, __global float *y) {
  int i = get_global_id(0);
  float v = x[i];
  float s = 0.0f;
  if (v > 0.0f) s = work_group_reduce_add(v);
  s = work_group_reduce_max(s);      // broadcast the sum to every lane (masked lanes got nothing)
  y[i] = v / s;
}
// layernorm-style: one row per group, two reductions, then a uniform value used by vector code
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void rln(__global const float *x, __global float *y, int C) {
  int i = get_global_id(0);
  float v = x[i];
  float mean = work_group_reduce_add(v) / C;
  float d = v - mean;
  float var = work_group_reduce_add(d * d) / C;
  y[i] = d * rsqrt(var + 1e-5f);
}
// reduction inside a control-thread region: per group, K rows of a matrix, row sums
__kernel __attribute__((reqd_work_group_size(64, 1, 1))) void rrows(__global const float *a, __global float *out, int K, int n) {
  int i = get_global_id(0);
  int g = get_group_id(0);
  for (int k = 0; k < K; k++) {
    float s = work_group_reduce_add(a[k * n + i]);
    out[g * K + k] = s;              // uniform value to a uniform address: a scalar store, no divergence
  }
}

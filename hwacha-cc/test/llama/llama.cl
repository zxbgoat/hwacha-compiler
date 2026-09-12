// llama2.c forward pass as OpenCL kernels for hwacha-cc: one work-item per output element, loops inside.
#include "hwacha_math.h"
// xout[i] = sum_j w[i*n+j] * x[j]   (each lane walks its own row: strided stream over w, uniform x)
__kernel void matmul(__global float *xout, __global const float *x, __global const float *w, int n) {
  int i = get_global_id(0);
  float val = 0.0f;
  for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
  xout[i] = val;
}
// o[j] = w[j] * (ss * x[j]); ss = 1/sqrt(mean(x^2)+eps) is a 64-element reduction done by the host
__kernel void rmsnorm(__global float *o, __global const float *x, __global const float *w, float ss) {
  int j = get_global_id(0);
  o[j] = w[j] * (ss * x[j]);
}
// RoPE on q (and on k for i < kv_dim) with host-precomputed cos/sin tables [pos][head_dim/2]
__kernel void rope(__global float *q, __global float *k, __global const float *fcr, __global const float *fci,
                   int pos, int head_size, int kv_dim) {
  int i = 2 * get_global_id(0);
  int head_dim = i % head_size;
  int ti = pos * (head_size / 2) + head_dim / 2;
  float c = fcr[ti], s = fci[ti];
  float q0 = q[i], q1 = q[i + 1];
  q[i] = q0 * c - q1 * s;
  q[i + 1] = q0 * s + q1 * c;
  if (i < kv_dim) {
    float k0 = k[i], k1 = k[i + 1];
    k[i] = k0 * c - k1 * s;
    k[i + 1] = k0 * s + k1 * c;
  }
}
// att[h][t] = q_h . k_t / sqrt(hs) for all (h, t<=pos) pairs flattened into one NDRange
__kernel void att_score(__global float *att, __global const float *q, __global const float *kc,
                        int pos, int head_size, int kv_dim, int kv_mul, int seq, float sqrt_hs) {
  int gid = get_global_id(0);
  int np = pos + 1;
  int h = gid / np;
  int t = gid - h * np;
  __global const float *qh = q + h * head_size;
  __global const float *kh = kc + t * kv_dim + (h / kv_mul) * head_size;
  float score = 0.0f;
  for (int i = 0; i < head_size; i++) score += qh[i] * kh[i];
  att[h * seq + t] = score / sqrt_hs;
}
// softmax over t for each head (one lane per head)
__kernel void att_softmax(__global float *att, int pos, int seq) {
  __global float *a = att + get_global_id(0) * seq;
  float mx = a[0];
  for (int t = 1; t <= pos; t++) mx = fmax(mx, a[t]);
  float sum = 0.0f;
  for (int t = 0; t <= pos; t++) { float e = hw_expf(a[t] - mx); a[t] = e; sum += e; }
  for (int t = 0; t <= pos; t++) a[t] /= sum;
}
// xb[h*hs+i] = sum_t att[h][t] * v_t[h][i]
__kernel void att_value(__global float *xb, __global const float *att, __global const float *vc,
                        int pos, int head_size, int kv_dim, int kv_mul, int seq) {
  int gid = get_global_id(0);
  int h = gid / head_size;
  int hi = gid - h * head_size;
  __global const float *a = att + h * seq;
  __global const float *v = vc + (h / kv_mul) * head_size + hi;
  float acc = 0.0f;
  for (int t = 0; t <= pos; t++) acc += a[t] * v[t * kv_dim];
  xb[gid] = acc;
}
__kernel void residual(__global float *x, __global const float *y) { int i = get_global_id(0); x[i] += y[i]; }
// hb[i] = silu(hb[i]) * hb2[i]
__kernel void silu_mul(__global float *hb, __global const float *hb2) {
  int i = get_global_id(0);
  float v = hb[i];
  v *= 1.0f / (1.0f + hw_expf(-v));
  hb[i] = v * hb2[i];
}
// same product with host-transposed weights wt[n][d]: per j the lanes read one contiguous row
__kernel void matmul_t(__global float *xout, __global const float *x, __global const float *wt, int n, int d) {
  int i = get_global_id(0);
  float val = 0.0f;
  for (int j = 0; j < n; j++) val += wt[j * d + i] * x[j];
  xout[i] = val;
}

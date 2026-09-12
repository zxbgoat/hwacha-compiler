// GPT-2 forward pass (llm.c train_gpt2.c) as hwacha-cc kernels. B = 1; weights for the matmuls are
// host-transposed to [in][out] so each j step of a row launch reads one contiguous row (unit stride).
#include "../llama/hwacha_math.h"
static inline float hw_tanhf(float y) { return 1.0f - 2.0f / (hw_expf(2.0f * y) + 1.0f); }

// out[t][c] = wte[tok[t]][c] + wpe[t][c]           (gid over T*C)
__kernel void encoder(__global float *out, __global const int *tok, __global const float *wte, __global const float *wpe, int C) {
  int gid = get_global_id(0);
  int t = gid / C, c = gid - t * C;
  out[gid] = wte[tok[t] * C + c] + wpe[gid];
}
// per row t: mean and 1/std over C                 (gid over T)
__kernel void ln_stats(__global float *mean, __global float *rstd, __global const float *inp, int C) {
  int t = get_global_id(0);
  __global const float *x = inp + t * C;
  float m = 0.0f;
  for (int i = 0; i < C; i++) m += x[i];
  m = m / C;
  float v = 0.0f;
  for (int i = 0; i < C; i++) { float xs = x[i] - m; v += xs * xs; }
  v = v / C;
  mean[t] = m;
  rstd[t] = 1.0f / sqrt(v + 1e-5f);
}
// out[t][i] = rstd[t]*(inp[t][i]-mean[t])*w[i] + b[i]   (gid over T*C)
__kernel void ln_apply(__global float *out, __global const float *inp, __global const float *mean, __global const float *rstd,
                       __global const float *w, __global const float *b, int C) {
  int gid = get_global_id(0);
  int t = gid / C, i = gid - t * C;
  float n = rstd[t] * (inp[gid] - mean[t]);
  out[gid] = n * w[i] + b[i];
}
// one row of out = bias + inp_row . W^T, with WT[i][o]   (gid over OC)
__kernel void matmul_t_row(__global float *out, __global const float *inp, __global const float *WT, __global const float *bias, int C, int OC) {
  int o = get_global_id(0);
  float val = bias[o];
  for (int i = 0; i < C; i++) val += inp[i] * WT[i * OC + o];
  out[o] = val;
}
__kernel void matmul_t_row_nb(__global float *out, __global const float *inp, __global const float *WT, int C, int OC) {
  int o = get_global_id(0);
  float val = 0.0f;
  for (int i = 0; i < C; i++) val += inp[i] * WT[i * OC + o];
  out[o] = val;
}
// preatt[h][t][t2] = scale * q_t . k_t2 for t2 <= t, else 0     (gid over NH*T*T)
__kernel void att_score(__global float *preatt, __global const float *qkv, int T, int C, int NH, float scale) {
  int gid = get_global_id(0);
  int hs = C / NH, C3 = 3 * C;
  int h = gid / (T * T), r = gid - h * T * T;
  int t = r / T, t2 = r - t * T;
  float val = 0.0f;
  if (t2 <= t) {
    __global const float *q = qkv + t * C3 + h * hs;
    __global const float *k = qkv + t2 * C3 + C + h * hs;
    for (int i = 0; i < hs; i++) val += q[i] * k[i];
    val *= scale;
  }
  preatt[gid] = val;
}
// causal softmax of one (h,t) row over t2                        (gid over NH*T)
__kernel void att_softmax(__global float *att, __global const float *preatt, int T) {
  int gid = get_global_id(0);
  int t = gid % T;
  __global const float *p = preatt + gid * T;
  __global float *a = att + gid * T;
  float maxval = -10000.0f;
  for (int t2 = 0; t2 < T; t2++) { float v = t2 <= t ? p[t2] : -1.0e30f; maxval = fmax(maxval, v); }
  float expsum = 0.0f;
  for (int t2 = 0; t2 < T; t2++) { float e = t2 <= t ? hw_expf(p[t2] - maxval) : 0.0f; expsum += e; a[t2] = e; }
  float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
  for (int t2 = 0; t2 < T; t2++) a[t2] *= inv;
}
// out[t][h*hs+i] = sum_t2 att[h][t][t2] * v_t2[h][i]   (att is 0 beyond t)   (gid over T*C)
__kernel void att_value(__global float *out, __global const float *att, __global const float *qkv, int T, int C, int NH) {
  int gid = get_global_id(0);
  int hs = C / NH, C3 = 3 * C;
  int t = gid / C, c = gid - t * C;
  int h = c / hs, i = c - h * hs;
  __global const float *a = att + h * T * T + t * T;
  __global const float *v = qkv + 2 * C + h * hs + i;
  float acc = 0.0f;
  for (int t2 = 0; t2 < T; t2++) acc += a[t2] * v[t2 * C3];
  out[gid] = acc;
}
__kernel void gelu(__global float *out, __global const float *inp, float gelu_scale) {
  int i = get_global_id(0);
  float x = inp[i];
  float cube = 0.044715f * x * x * x;
  out[i] = 0.5f * x * (1.0f + hw_tanhf(gelu_scale * (x + cube)));
}
__kernel void residual(__global float *out, __global const float *a, __global const float *b) {
  int i = get_global_id(0);
  out[i] = a[i] + b[i];
}
// softmax of one logits row over V                                (gid over T)
__kernel void softmax_row(__global float *probs, __global const float *logits, int V) {
  int t = get_global_id(0);
  __global const float *l = logits + t * V;
  __global float *p = probs + t * V;
  float maxval = -10000.0f;
  for (int i = 0; i < V; i++) maxval = fmax(maxval, l[i]);
  float sum = 0.0f;
  for (int i = 0; i < V; i++) { float e = hw_expf(l[i] - maxval); p[i] = e; sum += e; }
  for (int i = 0; i < V; i++) p[i] /= sum;
}

// ---- the same three row-wise stages written with cross-lane reductions (work-group = one row) ----
#include "../run/hwacha_builtins.h"
// layernorm in one kernel: 64 lanes = one row (C must equal the work-group size)
__kernel __attribute__((reqd_work_group_size(64, 1, 1)))
void layernorm(__global float *out, __global const float *inp, __global const float *w, __global const float *b, int C) {
  int gid = get_global_id(0), i = get_local_id(0);
  float x = inp[gid];
  float m = work_group_reduce_add(x) / C;
  float d = x - m;
  float v = work_group_reduce_add(d * d) / C;
  float s = 1.0f / sqrt(v + 1e-5f);
  out[gid] = (s * d) * w[i] + b[i];
}
// causal softmax: 16 lanes = one (h,t) row over t2
__kernel __attribute__((reqd_work_group_size(16, 1, 1)))
void att_softmax_g(__global float *att, __global const float *preatt, int T) {
  int gid = get_global_id(0), t2 = get_local_id(0);
  int t = get_group_id(0) % T;
  float p = t2 <= t ? preatt[gid] : -1.0e30f;
  float maxval = fmax(-10000.0f, work_group_reduce_max(p));
  float e = t2 <= t ? hw_expf(p - maxval) : 0.0f;
  float expsum = work_group_reduce_add(e);
  float inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
  att[gid] = e * inv;
}
// vocabulary softmax: 128 lanes = one row, each lane owns V/128 consecutive logits
__kernel __attribute__((reqd_work_group_size(128, 1, 1)))
void softmax_g(__global float *probs, __global const float *logits, int V) {
  int row = get_group_id(0), lane = get_local_id(0);
  int per = V / 128;
  __global const float *l = logits + row * V + lane * per;
  __global float *p = probs + row * V + lane * per;
  float mx = -10000.0f;
  for (int i = 0; i < per; i++) mx = fmax(mx, l[i]);
  float maxval = work_group_reduce_max(mx);
  float sum = 0.0f;
  for (int i = 0; i < per; i++) { float e = hw_expf(l[i] - maxval); p[i] = e; sum += e; }
  float total = work_group_reduce_add(sum);
  for (int i = 0; i < per; i++) p[i] /= total;
}

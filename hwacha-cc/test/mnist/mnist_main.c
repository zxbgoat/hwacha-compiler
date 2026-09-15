// Three MNIST diffusion models on Hwacha, sharing the kernels of mnist.cl:
//   -DMODEL_BOT66     bot66/MNISTDiffusion: ShuffleNet-style UNet (depthwise convs), 1000 steps, cosine schedule, clipped sampler
//   -DMODEL_TEAPEARCE TeaPearce/Conditional_Diffusion_MNIST: ContextUnet n_feat 128, classifier-free guidance, 400 steps
//   -DMODEL_AESTUANS  aestuans/mnist-diffusion: ResNet UNet with concatenated time/context embeddings, 500 steps
// Every op exists twice: a scalar version on the same zero-padded planes (the reference) and the Hwacha kernel;
// the network code is written once and dispatches on `use_hw`. The first forward of each build is checked
// against a PyTorch output exported by export.py (<model>_check.bin), the first CHECK_STEPS sampling steps
// compare Hwacha against the scalar path, and an x86 build (-DX86, scalar only, same RNG) gives the reference image.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifndef STEPS
#define STEPS 100000
#endif
#ifndef CHECK_STEPS
#define CHECK_STEPS 1
#endif
#ifndef CLASS
#define CLASS 7
#endif
#ifndef GUIDE_W
#define GUIDE_W 0.0f   // classifier-free guidance weight (teapearce / aestuans): > 0 doubles the forwards
#endif
#define OCB 8
#if defined(MODEL_BOT66)
#define MODEL_NAME "bot66"
#elif defined(MODEL_TEAPEARCE)
#define MODEL_NAME "teapearce"
#elif defined(MODEL_AESTUANS)
#define MODEL_NAME "aestuans"
#else
#error "define one of MODEL_BOT66 / MODEL_TEAPEARCE / MODEL_AESTUANS"
#endif

#ifdef X86
static unsigned char *model_bin, *check_bin;
static long rdcycle(void) { return 0; }
static int use_hw = 0;
#else
asm(".section .rodata\n.balign 8\n.globl model_bin\nmodel_bin:\n.incbin \"" MODEL_NAME ".bin\"\n"
    ".balign 8\n.globl check_bin\ncheck_bin:\n.incbin \"" MODEL_NAME "_check.bin\"\n.previous");
extern const unsigned char model_bin[], check_bin[];
static long rdcycle(void) { long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
static int use_hw = 1;
float __math_oflowf(unsigned sign) { return sign ? -1.0f / 0.0f : 1.0f / 0.0f; }
float __math_uflowf(unsigned sign) { return sign ? -0.0f : 0.0f; }
int *__errno(void) { static int e; return &e; }
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  uintptr_t tval; asm volatile("csrr %0, mtval" : "=r"(tval));
  printf("TRAP cause %lx epc %lx tval %lx ra %lx sp %lx\n", cause, epc, tval, regs[1], regs[2]);
  exit(1);
}
#endif
static const char *fmtf(float v) {   // the bare-metal printf has no %g
  static char buf[16][32]; static int k;
  char *b = buf[k++ & 15], *q = b;
  long m = (long)(fabsf(v) * 1e6f + 0.5f), ip = m / 1000000, fp = m % 1000000;
  if (v < 0) *q++ = '-';
  char tmp[24]; int n = 0;
  do { tmp[n++] = '0' + ip % 10; ip /= 10; } while (ip);
  while (n) *q++ = tmp[--n];
  *q++ = '.';
  for (long d = 100000; d; d /= 10) *q++ = '0' + (fp / d) % 10;
  *q = 0;
  return b;
}

// ------------------------------------------------------------------ deterministic noise
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) { rng = rng * 6364136223846793005ull + 1442695040888963407ull; return (float)((rng >> 40) & 0xFFFFFF) / 16777216.0f; }
static float randn(void) { float u1 = frand(), u2 = frand(); if (u1 < 1e-7f) u1 = 1e-7f; return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2); }

// ------------------------------------------------------------------ activations on padded planes
typedef struct { float *d; int C, H, W, Hp, Wp, plane; } Act;
#define ARENA (1 << 24)
static float arena[ARENA]; static long atop;
static Act new_act(int C, int H, int W) {
  Act a; a.C = C; a.H = H; a.W = W; a.Hp = H + 2; a.Wp = W + 2; a.plane = a.Hp * a.Wp;
  int guard = 2 * a.Wp + 8;
  atop = (atop + 15) & ~15L;
  a.d = arena + atop + guard;
  atop += (long)C * a.plane + 2 * guard;
  if (atop > ARENA) { printf("arena overflow\n"); exit(1); }
  return a;
}
static Act chunk(Act x, int c0, int C) { Act a = x; a.d = x.d + (long)c0 * x.plane; a.C = C; return a; }
#define AT(a, c, i, j) ((a).d[(long)(c) * (a).plane + ((i) + 1) * (a).Wp + (j) + 1])   // interior (i, j) of channel c
static float zeros[512], ones[512];

// ------------------------------------------------------------------ kernels
#ifndef X86
void conv3x3_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int Wp, int oc0);
void conv3x3_1_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int Wp, int oc);
void conv3x3_s2_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int Wp, int plane2, int Wp2, int oc0);
void conv1x1_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int oc0);
void conv1x1_1_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int oc);
void conv1x1_s2_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int Wp, int plane2, int Wp2, int oc0);
void dwconv3x3_ct(long n, const float *x, const float *w, const float *b, float *y, int n_ch, int plane, int Wp);
void dwconv3x3_s2_ct(long n, const float *x, const float *w, const float *b, float *y, int n_ch, int plane, int Wp, int plane2, int Wp2);
void bn_act_ct(long n, float *x, int plane, int Wp, int H, int W, const float *scale, const float *shift, int act);
void bn_add_act_ct(long n, float *x, int plane, int Wp, int H, int W, const float *scale, const float *shift, const float *res, int act);
void chan_sum_ct(long n, const float *x, int plane, int Wp, int H, int W, float *out);
void chan_sqdiff_ct(long n, const float *x, int plane, int Wp, int H, int W, const float *mean, int cpg, float *out);
void gn_relu_ct(long n, float *x, int plane, int Wp, int H, int W, const float *mean, const float *rstd, int cpg, const float *gamma, const float *beta);
void maxpool2_ct(long n, const float *x, int plane, int Wp, int H, int W, float *y, int plane2, int Wp2);
void avgpool_gelu_ct(long n, const float *x, int plane, int Wp, int K, float *v);
void up_gemv_ct(long n, const float *v, const float *w, const float *b, float *y, int n_in, int n_out, int plane2, int Wp2, int K);
void convT2_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int n_out, int plane, int Wp, int W, int plane2, int Wp2, int oc0);
void upsample2_ct(long n, const float *x, int plane, int Wp, int H, int W, float *y, int plane2, int Wp2);
void fc_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int act);
void shuffle2_ct(long n, const float *src, float *dst, int C, int plane);
void chan_fill_ct(long n, float *dst, int plane, int Wp, int H, int W, const float *vals);
void vcopy_ct(long n, const float *src, float *dst);
void vzero_ct(long n, float *dst);
#endif

// ------------------------------------------------------------------ scalar versions (the reference)
static float sc_act(float x, int act) {
  if (act == 1) return x / (1.0f + expf(-x));
  if (act == 2) return 0.5f * x * (1.0f + erff(x * 0.70710678f));
  if (act == 3) return x > 0 ? x : 0;
  return x;
}
static void zero_border(Act y) {
  for (int c = 0; c < y.C; c++) {
    float *p = y.d + (long)c * y.plane;
    for (int j = 0; j < y.Wp; j++) { p[j] = 0; p[(y.Hp - 1) * y.Wp + j] = 0; }
    for (int i = 0; i < y.Hp; i++) { p[i * y.Wp] = 0; p[i * y.Wp + y.Wp - 1] = 0; }
  }
}
// generic conv: k in {1, 3} (pad k/2), stride 1 or 2, groups 1 (dense) or C (depthwise)
static void sc_conv(Act x, const float *w, const float *b, Act y, int k, int stride, int depthwise) {
  int pad = k / 2;
  for (int oc = 0; oc < y.C; oc++)
    for (int i = 0; i < y.H; i++)
      for (int j = 0; j < y.W; j++) {
        float s = b ? b[oc] : 0;
        int ic0 = depthwise ? oc : 0, ic1 = depthwise ? oc + 1 : x.C;
        for (int ic = ic0; ic < ic1; ic++) {
          const float *kk = depthwise ? w + oc * 9 : w + ((long)oc * x.C + ic) * k * k;
          for (int di = 0; di < k; di++)
            for (int dj = 0; dj < k; dj++) {
              int I = i * stride + di - pad, J = j * stride + dj - pad;
              if (I < 0 || I >= x.H || J < 0 || J >= x.W) continue;
              s += kk[di * k + dj] * AT(x, ic, I, J);
            }
        }
        AT(y, oc, i, j) = s;
      }
  zero_border(y);
}
static void sc_affine(Act y, const float *scale, const float *shift, const float *res, int act) {
  for (int c = 0; c < y.C; c++)
    for (int i = 0; i < y.H; i++)
      for (int j = 0; j < y.W; j++) {
        float v = AT(y, c, i, j) * scale[c] + shift[c];
        if (res) v += res[(long)c * y.plane + (i + 1) * y.Wp + j + 1];
        AT(y, c, i, j) = sc_act(v, act);
      }
  zero_border(y);
}
static void sc_groupnorm_relu(Act y, int G, const float *gamma, const float *beta) {
  int cpg = y.C / G;
  for (int g = 0; g < G; g++) {
    double s = 0, s2 = 0; long n = (long)cpg * y.H * y.W;
    for (int c = g * cpg; c < (g + 1) * cpg; c++) for (int i = 0; i < y.H; i++) for (int j = 0; j < y.W; j++) s += AT(y, c, i, j);
    float mean = s / n;
    for (int c = g * cpg; c < (g + 1) * cpg; c++) for (int i = 0; i < y.H; i++) for (int j = 0; j < y.W; j++) { float d = AT(y, c, i, j) - mean; s2 += d * d; }
    float rstd = 1.0f / sqrtf(s2 / n + 1e-5f);
    for (int c = g * cpg; c < (g + 1) * cpg; c++) for (int i = 0; i < y.H; i++) for (int j = 0; j < y.W; j++) {
      float v = (AT(y, c, i, j) - mean) * rstd * gamma[c] + beta[c]; AT(y, c, i, j) = v > 0 ? v : 0;
    }
  }
  zero_border(y);
}
static void sc_maxpool2(Act x, Act y) {
  for (int c = 0; c < y.C; c++) for (int i = 0; i < y.H; i++) for (int j = 0; j < y.W; j++) {
    float m = AT(x, c, 2 * i, 2 * j);
    for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) { float v = AT(x, c, 2 * i + a, 2 * j + b); if (v > m) m = v; }
    AT(y, c, i, j) = m;
  }
  zero_border(y);
}
static void sc_avgpool_gelu(Act x, int K, float *v) {
  for (int c = 0; c < x.C; c++) { float s = 0; for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) s += AT(x, c, i, j); v[c] = sc_act(s / (K * K), 2); }
}
static void sc_up_gemv(const float *v, int n_in, const float *w, const float *b, Act y, int K) {
  for (int oc = 0; oc < y.C; oc++) for (int i = 0; i < K; i++) for (int j = 0; j < K; j++) {
    float a = b[oc];
    for (int ic = 0; ic < n_in; ic++) a += v[ic] * w[((long)ic * y.C + oc) * K * K + i * K + j];
    AT(y, oc, i, j) = a;
  }
  zero_border(y);
}
static void sc_convT2(Act x, const float *w, const float *b, Act y) {
  for (int oc = 0; oc < y.C; oc++) for (int i = 0; i < y.H; i++) for (int j = 0; j < y.W; j++) AT(y, oc, i, j) = b[oc];
  for (int ic = 0; ic < x.C; ic++) for (int i = 0; i < x.H; i++) for (int j = 0; j < x.W; j++) {
    float v = AT(x, ic, i, j);
    for (int oc = 0; oc < y.C; oc++) { const float *k = w + ((long)ic * y.C + oc) * 4; AT(y, oc, 2 * i, 2 * j) += k[0] * v; AT(y, oc, 2 * i, 2 * j + 1) += k[1] * v; AT(y, oc, 2 * i + 1, 2 * j) += k[2] * v; AT(y, oc, 2 * i + 1, 2 * j + 1) += k[3] * v; }
  }
  zero_border(y);
}
static void sc_upsample2(Act x, Act y) {   // bilinear, align_corners = False
  for (int c = 0; c < y.C; c++) for (int I = 0; I < y.H; I++) for (int J = 0; J < y.W; J++) {
    float si = (I + 0.5f) / 2 - 0.5f, sj = (J + 0.5f) / 2 - 0.5f;
    if (si < 0) si = 0; if (sj < 0) sj = 0;
    int i0 = (int)si, j0 = (int)sj; float li = si - i0, lj = sj - j0;
    int i1 = i0 + 1 < x.H ? i0 + 1 : x.H - 1, j1 = j0 + 1 < x.W ? j0 + 1 : x.W - 1;
    AT(y, c, I, J) = (AT(x, c, i0, j0) * (1 - lj) + AT(x, c, i0, j1) * lj) * (1 - li) + (AT(x, c, i1, j0) * (1 - lj) + AT(x, c, i1, j1) * lj) * li;
  }
  zero_border(y);
}
static void sc_fc(const float *x, int n_in, const float *w, const float *b, float *y, int n_out, int act) {
  for (int o = 0; o < n_out; o++) { float a = b[o]; for (int i = 0; i < n_in; i++) a += x[i] * w[(long)o * n_in + i]; y[o] = sc_act(a, act); }
}

// ------------------------------------------------------------------ ops (dispatch)
static void op_conv(Act x, const float *w, const float *b, Act y, int k, int stride, int depthwise) {
  if (!b) b = zeros;
#ifndef X86
  if (use_hw) {
    if (depthwise) {
      if (stride == 1) dwconv3x3_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp);
      else dwconv3x3_s2_ct(y.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, y.plane, y.Wp);
    } else if (k == 3 && stride == 1) {
      int oc = 0;
      for (; oc + OCB <= y.C; oc += OCB) conv3x3_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, oc);
      for (; oc < y.C; oc++) conv3x3_1_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, oc);
    } else if (k == 3) {
      for (int oc = 0; oc < y.C; oc += 4) conv3x3_s2_ct(y.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, y.plane, y.Wp, oc);
    } else if (stride == 1) {
      int oc = 0;
      for (; oc + OCB <= y.C; oc += OCB) conv1x1_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, oc);
      for (; oc < y.C; oc++) conv1x1_1_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, oc);
    } else {
      for (int oc = 0; oc < y.C; oc += 4) conv1x1_s2_ct(y.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, y.plane, y.Wp, oc);
    }
    return;
  }
#endif
  sc_conv(x, w, b, y, k, stride, depthwise);
}
static void op_affine(Act y, const float *scale, const float *shift, const float *res, int act) {
#ifndef X86
  if (use_hw) {
    if (res) bn_add_act_ct((long)y.C * y.plane, y.d, y.plane, y.Wp, y.H, y.W, scale, shift, res, act);
    else bn_act_ct((long)y.C * y.plane, y.d, y.plane, y.Wp, y.H, y.W, scale, shift, act);
    return;
  }
#endif
  sc_affine(y, scale, shift, res, act);
}
static float stat_a[512], stat_b[512], stat_m[512], stat_r[512];
static void op_groupnorm_relu(Act y, int G, const float *gamma, const float *beta) {
#ifndef X86
  if (use_hw) {
    int cpg = y.C / G; long n = (long)cpg * y.H * y.W;
    chan_sum_ct(y.C, y.d, y.plane, y.Wp, y.H, y.W, stat_a);
    for (int g = 0; g < G; g++) { float s = 0; for (int c = 0; c < cpg; c++) s += stat_a[g * cpg + c]; stat_m[g] = s / n; }
    chan_sqdiff_ct(y.C, y.d, y.plane, y.Wp, y.H, y.W, stat_m, cpg, stat_b);
    for (int g = 0; g < G; g++) { float s = 0; for (int c = 0; c < cpg; c++) s += stat_b[g * cpg + c]; stat_r[g] = 1.0f / sqrtf(s / n + 1e-5f); }
    gn_relu_ct((long)y.C * y.plane, y.d, y.plane, y.Wp, y.H, y.W, stat_m, stat_r, cpg, gamma, beta);
    return;
  }
#endif
  sc_groupnorm_relu(y, G, gamma, beta);
}
static Act op_maxpool2(Act x) {
  Act y = new_act(x.C, x.H / 2, x.W / 2);
#ifndef X86
  if (use_hw) { maxpool2_ct((long)y.C * y.plane, x.d, x.plane, x.Wp, x.H, x.W, y.d, y.plane, y.Wp); return y; }
#endif
  sc_maxpool2(x, y); return y;
}
static void op_avgpool_gelu(Act x, int K, float *v) {
#ifndef X86
  if (use_hw) { avgpool_gelu_ct(x.C, x.d, x.plane, x.Wp, K, v); return; }
#endif
  sc_avgpool_gelu(x, K, v);
}
static Act op_up_gemv(const float *v, int n_in, const float *w, const float *b, int n_out, int K) {
  Act y = new_act(n_out, K, K);
#ifndef X86
  if (use_hw) { up_gemv_ct((long)y.C * y.plane, v, w, b, y.d, n_in, n_out, y.plane, y.Wp, K); return y; }
#endif
  sc_up_gemv(v, n_in, w, b, y, K); return y;
}
static Act op_convT2(Act x, const float *w, const float *b, int n_out) {
  Act y = new_act(n_out, 2 * x.H, 2 * x.W);
#ifndef X86
  if (use_hw) {
    vzero_ct((long)y.C * y.plane, y.d);
    for (int oc = 0; oc < n_out; oc += 4) convT2_ct((long)x.H * x.W, x.d, w, b, y.d, x.C, n_out, x.plane, x.Wp, x.W, y.plane, y.Wp, oc);
    return y;
  }
#endif
  sc_convT2(x, w, b, y); return y;
}
static Act op_upsample2(Act x) {
  Act y = new_act(x.C, 2 * x.H, 2 * x.W);
#ifndef X86
  if (use_hw) { upsample2_ct((long)y.C * y.plane, x.d, x.plane, x.Wp, x.H, x.W, y.d, y.plane, y.Wp); return y; }
#endif
  sc_upsample2(x, y); return y;
}
static void op_fc(const float *x, int n_in, const float *w, const float *b, float *y, int n_out, int act) {
#ifndef X86
  if (use_hw) { fc_ct(n_out, x, w, b, y, n_in, act); return; }
#endif
  sc_fc(x, n_in, w, b, y, n_out, act);
}
static void op_concat_into(Act dst, Act src) {
#ifndef X86
  if (use_hw) { vcopy_ct((long)src.C * src.plane, src.d, dst.d); return; }
#endif
  memcpy(dst.d, src.d, (long)src.C * src.plane * sizeof(float));
}
static Act op_concat(Act a, Act b) {
  Act c = new_act(a.C + b.C, a.H, a.W);
#ifndef X86
  if (use_hw) { vcopy_ct((long)a.C * a.plane, a.d, c.d); vcopy_ct((long)b.C * b.plane, b.d, c.d + (long)a.C * a.plane); return c; }
#endif
  memcpy(c.d, a.d, (long)a.C * a.plane * sizeof(float)); memcpy(c.d + (long)a.C * a.plane, b.d, (long)b.C * b.plane * sizeof(float)); return c;
}
static Act op_shuffle2(Act x) {
  Act y = new_act(x.C, x.H, x.W);
#ifndef X86
  if (use_hw) { shuffle2_ct((long)x.C * x.plane, x.d, y.d, x.C, x.plane); return y; }
#endif
  for (int c = 0; c < x.C; c++) memcpy(y.d + (long)c * y.plane, x.d + (long)((c & 1) * (x.C / 2) + c / 2) * x.plane, x.plane * sizeof(float));
  return y;
}
static Act op_chan_fill(int C, int H, int W, const float *vals) {
  Act y = new_act(C, H, W);
#ifndef X86
  if (use_hw) { chan_fill_ct((long)C * y.plane, y.d, y.plane, y.Wp, H, W, vals); return y; }
#endif
  for (int c = 0; c < C; c++) for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) AT(y, c, i, j) = vals[c];
  zero_border(y); return y;
}
static Act pad_input(const float *x, int C, int H, int W) {
  Act a = new_act(C, H, W);
  memset(a.d, 0, (long)C * a.plane * sizeof(float));
  for (int c = 0; c < C; c++) for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) AT(a, c, i, j) = x[(long)c * H * W + i * W + j];
  return a;
}
static void unpad_output(Act a, float *out) {
  for (int c = 0; c < a.C; c++) for (int i = 0; i < a.H; i++) for (int j = 0; j < a.W; j++) out[(long)c * a.H * a.W + i * a.W + j] = AT(a, c, i, j);
}

// ------------------------------------------------------------------ weights
typedef struct { const float *w, *b; } Conv;
typedef struct { const float *g, *bt, *mean, *var; float *scale, *shift; } BN;   // eval-mode batch norm folded to an affine map
typedef struct { Conv c; BN bn; } CBN;
typedef struct { const float *w1, *b1, *w2, *b2; } MLP;
static const float *wp;   // load cursor
static float bnpool[1 << 16]; static long bntop;
static Conv load_conv(int n_out, int n_in, int k, int bias) { Conv c; c.w = wp; wp += (long)n_out * n_in * k * k; c.b = bias ? wp : NULL; if (bias) wp += n_out; return c; }
static Conv load_convT(int n_in, int n_out, int k) { Conv c; c.w = wp; wp += (long)n_in * n_out * k * k; c.b = wp; wp += n_out; return c; }
static Conv load_dw(int C) { Conv c; c.w = wp; wp += (long)C * 9; c.b = wp; wp += C; return c; }
static BN load_bn(int C) {
  BN b; b.g = wp; wp += C; b.bt = wp; wp += C; b.mean = wp; wp += C; b.var = wp; wp += C;
  b.scale = bnpool + bntop; bntop += C; b.shift = bnpool + bntop; bntop += C;
  for (int c = 0; c < C; c++) { b.scale[c] = b.g[c] / sqrtf(b.var[c] + 1e-5f); b.shift[c] = b.bt[c] - b.mean[c] * b.scale[c]; }
  return b;
}
static CBN load_cbn(int n_out, int n_in, int k) { CBN r; r.c = load_conv(n_out, n_in, k, 1); r.bn = load_bn(n_out); return r; }
static MLP load_mlp(int n_in, int hid, int n_out) { MLP m; m.w1 = wp; wp += (long)hid * n_in; m.b1 = wp; wp += hid; m.w2 = wp; wp += (long)n_out * hid; m.b2 = wp; wp += n_out; return m; }
static const float *load_vec(long n) { const float *p = wp; wp += n; return p; }
static const int *hdr; static int nhdr;

// ================================================================== bot66/MNISTDiffusion
#ifdef MODEL_BOT66
#define IMG_C 1
typedef struct { Conv dw1; BN bn1; CBN c1; CBN c2; Conv dw2; BN bn2; CBN c3; } Bottleneck;   // also the downsample unit
typedef struct { Bottleneck b[4]; MLP mlp; Bottleneck down; } EncBlock;
typedef struct { Bottleneck b[4]; MLP mlp; Bottleneck c1; } DecBlock;
static struct { const float *betas; CBN init; const float *temb; EncBlock enc[2]; Bottleneck mid[3]; DecBlock dec[2]; Conv final; } W;
static int T_STEPS, TEMB_DIM, BASE;
static Bottleneck load_bottleneck(int in, int out, int down) {
  Bottleneck b; int h = down ? in : in / 2, o = out / 2;
  b.dw1 = load_dw(h); b.bn1 = load_bn(h); b.c1 = load_cbn(o, h, 1);
  b.c2 = load_cbn(down ? o : h, h, 1); b.dw2 = load_dw(down ? o : h); b.bn2 = load_bn(down ? o : h); b.c3 = load_cbn(o, down ? o : h, 1);
  return b;
}
static void load_model(void) {
  hdr = (const int *)model_bin; nhdr = 6;
  T_STEPS = hdr[1]; TEMB_DIM = hdr[2]; BASE = hdr[3];
  int dims[3] = {BASE, BASE * hdr[4], BASE * hdr[5]};
  wp = (const float *)(model_bin + nhdr * 4);
  W.betas = load_vec(T_STEPS);
  W.init = load_cbn(BASE, IMG_C, 3); W.temb = load_vec((long)T_STEPS * TEMB_DIM);
  for (int e = 0; e < 2; e++) {
    int in = dims[e], out = dims[e + 1];
    for (int i = 0; i < 3; i++) W.enc[e].b[i] = load_bottleneck(in, in, 0);
    W.enc[e].b[3] = load_bottleneck(in, out / 2, 0);
    W.enc[e].mlp = load_mlp(TEMB_DIM, out, out / 2);
    W.enc[e].down = load_bottleneck(out / 2, out, 1);
  }
  W.mid[0] = load_bottleneck(dims[2], dims[2], 0); W.mid[1] = load_bottleneck(dims[2], dims[2], 0); W.mid[2] = load_bottleneck(dims[2], dims[2] / 2, 0);
  for (int d = 0; d < 2; d++) {
    int in = dims[2 - d], out = dims[1 - d];
    for (int i = 0; i < 3; i++) W.dec[d].b[i] = load_bottleneck(in, in, 0);
    W.dec[d].b[3] = load_bottleneck(in, in / 2, 0);
    W.dec[d].mlp = load_mlp(TEMB_DIM, in, in / 2);
    W.dec[d].c1 = load_bottleneck(in / 2, out / 2, 0);
  }
  W.final = load_conv(IMG_C, dims[0] / 2, 1, 1);
}
static Act run_cbn(Act x, const CBN *c, int n_out, int k, int stride, int act) {
  Act y = new_act(n_out, x.H / stride, x.W / stride);
  op_conv(x, c->c.w, c->c.b, y, k, stride, 0);
  op_affine(y, c->bn.scale, c->bn.shift, NULL, act);
  return y;
}
static Act run_dwbn(Act x, const Conv *dw, const BN *bn, int stride) {
  Act y = new_act(x.C, x.H / stride, x.W / stride);
  op_conv(x, dw->w, dw->b, y, 3, stride, 1);
  op_affine(y, bn->scale, bn->shift, NULL, 0);
  return y;
}
static Act run_bottleneck(Act x, const Bottleneck *b, int out) {
  Act x1 = chunk(x, 0, x.C / 2), x2 = chunk(x, x.C / 2, x.C / 2);
  Act y1 = run_cbn(run_dwbn(x1, &b->dw1, &b->bn1, 1), &b->c1, out / 2, 1, 1, 1);
  Act t = run_cbn(x2, &b->c2, x.C / 2, 1, 1, 1);
  Act y2 = run_cbn(run_dwbn(t, &b->dw2, &b->bn2, 1), &b->c3, out / 2, 1, 1, 1);
  return op_shuffle2(op_concat(y1, y2));
}
static Act run_downsample(Act x, const Bottleneck *b, int out) {
  Act y1 = run_cbn(run_dwbn(x, &b->dw1, &b->bn1, 2), &b->c1, out / 2, 1, 1, 1);
  Act t = run_cbn(x, &b->c2, out / 2, 1, 1, 1);
  Act y2 = run_cbn(run_dwbn(t, &b->dw2, &b->bn2, 2), &b->c3, out / 2, 1, 1, 1);
  return op_shuffle2(op_concat(y1, y2));
}
static float mlp_h[512], mlp_o[512];
static void run_timemlp(Act x, const MLP *m, const float *temb, int hid) {   // x = silu(x + mlp(t))
  op_fc(temb, TEMB_DIM, m->w1, m->b1, mlp_h, hid, 1);
  op_fc(mlp_h, hid, m->w2, m->b2, mlp_o, x.C, 0);
  op_affine(x, ones, mlp_o, NULL, 1);
}
// eps = Unet(x, t)
static void forward(const float *x, int t, const float *cvec, float *eps) {
  (void)cvec;
  atop = 0;
  int dims[3] = {BASE, BASE * hdr[4], BASE * hdr[5]};
  Act a = run_cbn(pad_input(x, IMG_C, 28, 28), &W.init, BASE, 3, 1, 1);
#ifdef DEBUG_STAGES
  printf("init %s %s %s %s\n", fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
  { Act b0 = a; Act x1 = chunk(b0, 0, b0.C / 2), x2 = chunk(b0, b0.C / 2, b0.C / 2);
    Act y1 = run_cbn(run_dwbn(x1, &W.enc[0].b[0].dw1, &W.enc[0].b[0].bn1, 1), &W.enc[0].b[0].c1, BASE / 2, 1, 1, 1);
    printf("b0.branch1 %s %s %s %s\n", fmtf(AT(y1,0,0,0)), fmtf(AT(y1,0,0,1)), fmtf(AT(y1,0,0,2)), fmtf(AT(y1,0,0,3)));
    Act tt = run_cbn(x2, &W.enc[0].b[0].c2, b0.C / 2, 1, 1, 1);
    Act y2 = run_cbn(run_dwbn(tt, &W.enc[0].b[0].dw2, &W.enc[0].b[0].bn2, 1), &W.enc[0].b[0].c3, BASE / 2, 1, 1, 1);
    printf("b0.branch2 %s %s %s %s\n", fmtf(AT(y2,0,0,0)), fmtf(AT(y2,0,0,1)), fmtf(AT(y2,0,0,2)), fmtf(AT(y2,0,0,3))); }
#endif
  const float *temb = W.temb + (long)t * TEMB_DIM;
  Act sc[2];
  for (int e = 0; e < 2; e++) {
    int in = dims[e], out = dims[e + 1];
    for (int i = 0; i < 3; i++) a = run_bottleneck(a, &W.enc[e].b[i], in);
    a = run_bottleneck(a, &W.enc[e].b[3], out / 2);
#ifdef DEBUG_STAGES
    printf("enc%d.conv0 %s %s %s %s\n", e, fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
#endif
    sc[e] = a;                                   // the shortcut is conv0's output before the time MLP ...
    Act am = new_act(a.C, a.H, a.W);             // ... which is applied in place: work on a copy
    op_concat_into(am, a);
    a = am;
    run_timemlp(a, &W.enc[e].mlp, temb, out);
#ifdef DEBUG_STAGES
    printf("enc%d.mlp %s %s %s %s\n", e, fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
#endif
    a = run_downsample(a, &W.enc[e].down, out);
#ifdef DEBUG_STAGES
    printf("enc%d.down %s %s %s %s row1 %s %s %s %s\n", e, fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)), fmtf(AT(a,0,1,0)), fmtf(AT(a,0,1,1)), fmtf(AT(a,0,1,2)), fmtf(AT(a,0,1,3)));
#endif
  }
  a = run_bottleneck(a, &W.mid[0], dims[2]); a = run_bottleneck(a, &W.mid[1], dims[2]); a = run_bottleneck(a, &W.mid[2], dims[2] / 2);
#ifdef DEBUG_STAGES
  printf("mid %s %s %s %s\n", fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
  printf("mid row1 %s %s %s %s row2 %s %s %s %s\n", fmtf(AT(a,0,1,0)), fmtf(AT(a,0,1,1)), fmtf(AT(a,0,1,2)), fmtf(AT(a,0,1,3)), fmtf(AT(a,0,2,0)), fmtf(AT(a,0,2,1)), fmtf(AT(a,0,2,2)), fmtf(AT(a,0,2,3)));
#endif
  for (int d = 0; d < 2; d++) {
    int in = dims[2 - d], out = dims[1 - d];
    Act up = op_upsample2(a);
#ifdef DEBUG_STAGES
    printf("up %s %s %s %s / %s %s %s %s\n", fmtf(AT(up,0,0,0)), fmtf(AT(up,0,0,1)), fmtf(AT(up,0,0,2)), fmtf(AT(up,0,0,3)), fmtf(AT(up,0,1,0)), fmtf(AT(up,0,1,1)), fmtf(AT(up,0,1,2)), fmtf(AT(up,0,1,3)));
#endif
    a = op_concat(up, sc[1 - d]);
    for (int i = 0; i < 3; i++) a = run_bottleneck(a, &W.dec[d].b[i], in);
    a = run_bottleneck(a, &W.dec[d].b[3], in / 2);
#ifdef DEBUG_STAGES
    printf("dec%d.conv0 %s %s %s %s\n", d, fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
#endif
    run_timemlp(a, &W.dec[d].mlp, temb, in);
#ifdef DEBUG_STAGES
    printf("dec%d.mlp %s %s %s %s\n", d, fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
#endif
    a = run_bottleneck(a, &W.dec[d].c1, out / 2);
#ifdef DEBUG_STAGES
    printf("dec%d.conv1 %s %s %s %s\n", d, fmtf(AT(a,0,0,0)), fmtf(AT(a,0,0,1)), fmtf(AT(a,0,0,2)), fmtf(AT(a,0,0,3)));
#endif
  }
  Act o = new_act(IMG_C, 28, 28);
  op_conv(a, W.final.w, W.final.b, o, 1, 1, 0);
  unpad_output(o, eps);
}
#endif

// ================================================================== TeaPearce/Conditional_Diffusion_MNIST
#ifdef MODEL_TEAPEARCE
#define IMG_C 1
typedef struct { CBN c1, c2; } ResBlock;
typedef struct { const float *g, *b; } GN;
typedef struct { Conv up; ResBlock r1, r2; } UpBlock;
static struct { ResBlock init, down1, down2; MLP temb1, temb2, cemb1, cemb2; Conv up0; GN gn0; UpBlock up1, up2; Conv out1; GN gn1; Conv out2; } W;
static int N_T, NF, NCLS;
static ResBlock load_res(int in, int out) { ResBlock r; r.c1 = load_cbn(out, in, 3); r.c2 = load_cbn(out, out, 3); return r; }
static GN load_gn(int C) { GN g; g.g = load_vec(C); g.b = load_vec(C); return g; }
static UpBlock load_up(int in, int out) { UpBlock u; u.up = load_convT(in, out, 2); u.r1 = load_res(out, out); u.r2 = load_res(out, out); return u; }
static void load_model(void) {
  hdr = (const int *)model_bin; nhdr = 4;
  N_T = hdr[1]; NF = hdr[2]; NCLS = hdr[3];
  wp = (const float *)(model_bin + nhdr * 4);
  W.init = load_res(IMG_C, NF); W.down1 = load_res(NF, NF); W.down2 = load_res(NF, 2 * NF);
  W.temb1 = load_mlp(1, 2 * NF, 2 * NF); W.temb2 = load_mlp(1, NF, NF); W.cemb1 = load_mlp(NCLS, 2 * NF, 2 * NF); W.cemb2 = load_mlp(NCLS, NF, NF);
  W.up0 = load_convT(2 * NF, 2 * NF, 7); W.gn0 = load_gn(2 * NF);
  W.up1 = load_up(4 * NF, NF); W.up2 = load_up(2 * NF, NF);
  W.out1 = load_conv(NF, 2 * NF, 3, 1); W.gn1 = load_gn(NF); W.out2 = load_conv(IMG_C, NF, 3, 1);
}
static Act run_cbg(Act x, const CBN *c, int n_out) {
  Act y = new_act(n_out, x.H, x.W);
  op_conv(x, c->c.w, c->c.b, y, 3, 1, 0);
  op_affine(y, c->bn.scale, c->bn.shift, NULL, 2);
  return y;
}
static float inv1414[512];
static Act run_res(Act x, const ResBlock *r, int out, int is_res) {
  Act x1 = run_cbg(x, &r->c1, out);
  Act x2 = run_cbg(x1, &r->c2, out);
  if (!is_res) return x2;
  op_affine(x2, inv1414, zeros, NULL, 0);          // x2 / 1.414 ...
  op_affine(x1, inv1414, zeros, x2.d, 0);          // ... + x1 / 1.414 (channels differ: 1 -> NF, so the residual is x1)
  return x1;
}
static float emb_h[512], temb[512], cemb[512], tvec[1], cvec_z[16];
static void run_embfc(const MLP *m, const float *x, int n_in, int n_out, float *out) {
  op_fc(x, n_in, m->w1, m->b1, emb_h, n_out, 2);
  op_fc(emb_h, n_out, m->w2, m->b2, out, n_out, 0);
}
static float hvec[512];
static void forward(const float *x, int ti, const float *cvec, float *eps) {
  atop = 0;
  for (int c = 0; c < 512; c++) inv1414[c] = 1.0f / 1.414f;
  Act x0 = run_res(pad_input(x, IMG_C, 28, 28), &W.init, NF, 1);
  Act d1 = op_maxpool2(run_res(x0, &W.down1, NF, 0));
  Act d2 = op_maxpool2(run_res(d1, &W.down2, 2 * NF, 0));
  op_avgpool_gelu(d2, 7, hvec);
  tvec[0] = (float)ti / N_T;
  run_embfc(&W.cemb1, cvec, NCLS, 2 * NF, cemb); run_embfc(&W.temb1, tvec, 1, 2 * NF, temb);
  Act u0 = op_up_gemv(hvec, 2 * NF, W.up0.w, W.up0.b, 2 * NF, 7);
  op_groupnorm_relu(u0, 8, W.gn0.g, W.gn0.b);
  op_affine(u0, cemb, temb, NULL, 0);
  Act u1 = op_convT2(op_concat(u0, d2), W.up1.up.w, W.up1.up.b, NF);
  u1 = run_res(run_res(u1, &W.up1.r1, NF, 0), &W.up1.r2, NF, 0);
  run_embfc(&W.cemb2, cvec, NCLS, NF, cemb); run_embfc(&W.temb2, tvec, 1, NF, temb);
  op_affine(u1, cemb, temb, NULL, 0);
  Act u2 = op_convT2(op_concat(u1, d1), W.up2.up.w, W.up2.up.b, NF);
  u2 = run_res(run_res(u2, &W.up2.r1, NF, 0), &W.up2.r2, NF, 0);
  Act o1 = new_act(NF, 28, 28);
  op_conv(op_concat(u2, x0), W.out1.w, W.out1.b, o1, 3, 1, 0);
  op_groupnorm_relu(o1, 8, W.gn1.g, W.gn1.b);
  Act o = new_act(IMG_C, 28, 28);
  op_conv(o1, W.out2.w, W.out2.b, o, 3, 1, 0);
  unpad_output(o, eps);
}
#endif

// ================================================================== aestuans/mnist-diffusion
#ifdef MODEL_AESTUANS
#define IMG_C 1
typedef struct { Conv c1; BN bn1; Conv c2; BN bn2; int has_sc; Conv sc; BN bnsc; int stride; } ResBlock;
static struct { ResBlock e1, e2, e3; MLP tmlp, cmlp; ResBlock u1; Conv up1; ResBlock u2; Conv up2; ResBlock fin; Conv final; } W;
static int N_STEPS, F0, F1, F2, EMB, NCLS;
static ResBlock load_res(int in, int out, int stride) {
  ResBlock r; r.stride = stride;
  r.c1 = load_conv(out, in, 3, 0); r.bn1 = load_bn(out); r.c2 = load_conv(out, out, 3, 0); r.bn2 = load_bn(out);
  r.has_sc = stride != 1 || in != out;
  if (r.has_sc) { r.sc = load_conv(out, in, 1, 0); r.bnsc = load_bn(out); }
  return r;
}
static void load_model(void) {
  hdr = (const int *)model_bin; nhdr = 7;
  N_STEPS = hdr[1]; F0 = hdr[2]; F1 = hdr[3]; F2 = hdr[4]; EMB = hdr[5]; NCLS = hdr[6];
  wp = (const float *)(model_bin + nhdr * 4);
  W.e1 = load_res(IMG_C, F0, 1); W.e2 = load_res(F0, F1, 2); W.e3 = load_res(F1, F2, 2);
  W.tmlp = load_mlp(1, EMB, EMB); W.cmlp = load_mlp(NCLS, EMB, EMB);
  W.u1 = load_res(F2 + 2 * EMB, F1, 1); W.up1 = load_convT(F1, F1, 2); W.u2 = load_res(F1, F0, 1); W.up2 = load_convT(F0, F0, 2);
  W.fin = load_res(F0 + F0, F0, 1); W.final = load_conv(1, F0, 1, 1);
}
static Act run_res(Act x, const ResBlock *r, int out) {
  Act y = new_act(out, x.H / r->stride, x.W / r->stride);
  op_conv(x, r->c1.w, NULL, y, 3, r->stride, 0);
  op_affine(y, r->bn1.scale, r->bn1.shift, NULL, 3);
  Act z = new_act(out, y.H, y.W);
  op_conv(y, r->c2.w, NULL, z, 3, 1, 0);
  const float *res = x.d;
  if (r->has_sc) {
    Act s = new_act(out, y.H, y.W);
    op_conv(x, r->sc.w, NULL, s, 1, r->stride, 0);
    op_affine(s, r->bnsc.scale, r->bnsc.shift, NULL, 0);
    res = s.d;
  }
  op_affine(z, r->bn2.scale, r->bn2.shift, res, 3);
  return z;
}
static float emb_h[64], temb[64], cemb[64], tvec[1], embv[64];
static void forward(const float *x, int t, const float *cvec, float *eps) {
  atop = 0;
  Act a = run_res(pad_input(x, IMG_C, 28, 28), &W.e1, F0);
  Act skip = a;
  a = run_res(run_res(a, &W.e2, F1), &W.e3, F2);
  tvec[0] = (float)t / N_STEPS;
  op_fc(tvec, 1, W.tmlp.w1, W.tmlp.b1, emb_h, EMB, 3); op_fc(emb_h, EMB, W.tmlp.w2, W.tmlp.b2, embv, EMB, 0);
  op_fc(cvec, NCLS, W.cmlp.w1, W.cmlp.b1, emb_h, EMB, 3); op_fc(emb_h, EMB, W.cmlp.w2, W.cmlp.b2, embv + EMB, EMB, 0);
  a = op_concat(a, op_chan_fill(2 * EMB, a.H, a.W, embv));
  a = op_convT2(run_res(a, &W.u1, F1), W.up1.w, W.up1.b, F1);
  a = op_convT2(run_res(a, &W.u2, F0), W.up2.w, W.up2.b, F0);
  a = run_res(op_concat(a, skip), &W.fin, F0);
  Act o = new_act(IMG_C, 28, 28);
  op_conv(a, W.final.w, W.final.b, o, 1, 1, 0);
  unpad_output(o, eps);
}
#endif

// ------------------------------------------------------------------ driver
#define NPIX (IMG_C * 28 * 28)
static float img[NPIX], eps[NPIX], eps2[NPIX], eps_ref[NPIX], eps_pt[NPIX * 2];
static float cvec[16], cvec_zero[16];
static int compare(const char *what, const float *a, const float *b, int n, float tol) {
  float md = 0, mx = 0;
  for (int k = 0; k < n; k++) { float d = fabsf(a[k] - b[k]); if (d > md) md = d; if (fabsf(b[k]) > mx) mx = fabsf(b[k]); }
  int ok = md <= tol * mx;
  printf("%s: max |diff| = %s, max |ref| = %s, %s\n", what, fmtf(md), fmtf(mx), ok ? "ok" : "FAIL");
  return !ok;
}
static void print_image(const float *x, float lo, float hi) {
  printf("image 28x28 (hex, %s..%s -> 00..ff):\n", fmtf(lo), fmtf(hi));
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) { float v = (x[i * 28 + j] - lo) / (hi - lo); if (v < 0) v = 0; if (v > 1) v = 1; printf("%02x", (int)(v * 255)); }
    printf("\n");
  }
  for (int i = 0; i < 28; i++) {   // ASCII view
    for (int j = 0; j < 28; j++) { float v = (x[i * 28 + j] - lo) / (hi - lo); printf("%c", v < 0.2f ? ' ' : v < 0.45f ? '.' : v < 0.7f ? '+' : '#'); }
    printf("\n");
  }
}
// eps for (img, t, class) with optional guidance
static void predict(int t, float *out) {
  forward(img, t, cvec, out);
  if (GUIDE_W != 0.0f) {
    forward(img, t, cvec_zero, eps2);
    for (int k = 0; k < NPIX; k++) out[k] = (1 + GUIDE_W) * out[k] - GUIDE_W * eps2[k];
  }
}
int main(int argc, char **argv) {
#ifdef X86
  const char *dir = argc > 1 ? argv[1] : ".";
  char path[256];
  snprintf(path, sizeof path, "%s/%s.bin", dir, MODEL_NAME); FILE *f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); model_bin = malloc(sz); if (fread(model_bin, 1, sz, f) != (size_t)sz) return 1; fclose(f);
  snprintf(path, sizeof path, "%s/%s_check.bin", dir, MODEL_NAME); f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET); check_bin = malloc(sz); if (fread(check_bin, 1, sz, f) != (size_t)sz) return 1; fclose(f);
#endif
  for (int i = 0; i < 512; i++) ones[i] = 1.0f;
  load_model();
  int fail = 0;
  // 1. one forward against PyTorch
  {
    const unsigned char *p = check_bin;
    int t = *(const int *)p; p += 4;
    float tf = 0; int has_tf = 0;
#if defined(MODEL_TEAPEARCE) || defined(MODEL_AESTUANS)
    memcpy(&tf, p, 4); p += 4; has_tf = 1;
#endif
    memcpy(img, p, NPIX * sizeof(float)); p += NPIX * sizeof(float);
    memset(cvec, 0, sizeof cvec);
#ifdef MODEL_BOT66
    memcpy(eps_pt, p, NPIX * sizeof(float));
    printf("%s: PyTorch check at t=%d\n", MODEL_NAME, t);
    forward(img, t, cvec, eps);
    fail |= compare("  forward vs PyTorch", eps, eps_pt, NPIX, 0.02f);
#elif defined(MODEL_TEAPEARCE)
    memcpy(eps_pt, p, 2 * NPIX * sizeof(float));
    int ti = (int)(tf * 400 + 0.5f); (void)t;
    printf("%s: PyTorch check at t=%s, class %d\n", MODEL_NAME, fmtf(tf), t);
    cvec[t] = -1.0f;   // upstream: c = one_hot * (-(1 - mask)), i.e. -one_hot when the context is used
    forward(img, ti, cvec, eps);
    fail |= compare("  forward (context) vs PyTorch", eps, eps_pt, NPIX, 0.02f);
    forward(img, ti, cvec_zero, eps);
    fail |= compare("  forward (no context) vs PyTorch", eps, eps_pt + NPIX, NPIX, 0.02f);
#else
    memcpy(eps_pt, p, NPIX * sizeof(float));
    int ti = (int)(tf * 500 + 0.5f);
    printf("%s: PyTorch check at t=%s, class %d\n", MODEL_NAME, fmtf(tf), t);
    cvec[t] = 1.0f;
    forward(img, ti, cvec, eps);
    fail |= compare("  forward vs PyTorch", eps, eps_pt, NPIX, 0.02f);
#endif
    (void)has_tf;
  }
  // 2. sampling
  memset(cvec, 0, sizeof cvec);
#ifdef MODEL_TEAPEARCE
  cvec[CLASS] = -1.0f;
#else
  cvec[CLASS] = 1.0f;
#endif
  for (int k = 0; k < NPIX; k++) img[k] = randn();
  long total = 0; int step = 0;
#ifdef MODEL_BOT66
  int T = T_STEPS;
  printf("sampling %d of %d steps (cosine schedule, clipped x0), checking %d\n", STEPS < T ? STEPS : T, T, CHECK_STEPS);
  static float alphas_cp[1024];
  float acc = 1; for (int i = 0; i < T; i++) { acc *= 1 - W.betas[i]; alphas_cp[i] = acc; }
  for (int t = T - 1; t >= 0 && step < STEPS; t--, step++) {
    long c0 = rdcycle(); predict(t, eps); long c1 = rdcycle(); total += c1 - c0;
    if (step < CHECK_STEPS && use_hw) { use_hw = 0; predict(t, eps_ref); use_hw = 1; printf("step %d (t=%d) ", step, t); fail |= compare("hwacha vs scalar", eps, eps_ref, NPIX, 0.02f); }
    else if (step % 100 == 0) printf("step %d (t=%d): %ld cycles\n", step, t, c1 - c0);
    float beta = W.betas[t], alpha = 1 - beta, acp = alphas_cp[t];
    for (int k = 0; k < NPIX; k++) {
      float x0 = sqrtf(1.0f / acp) * img[k] - sqrtf(1.0f / acp - 1.0f) * eps[k];
      if (x0 < -1) x0 = -1; if (x0 > 1) x0 = 1;
      float mean, std;
      if (t > 0) { float acp_prev = alphas_cp[t - 1]; mean = (beta * sqrtf(acp_prev) / (1 - acp)) * x0 + ((1 - acp_prev) * sqrtf(alpha) / (1 - acp)) * img[k]; std = sqrtf(beta * (1 - acp_prev) / (1 - acp)); }
      else { mean = (beta / (1 - acp)) * x0; std = 0; }
      img[k] = mean + std * randn();
    }
  }
  for (int k = 0; k < NPIX; k++) img[k] = (img[k] + 1) / 2;
  float lo = 0, hi = 1;
#elif defined(MODEL_TEAPEARCE)
  int T = N_T;
  printf("sampling %d of %d steps, class %d, guidance w=%s, checking %d\n", STEPS < T ? STEPS : T, T, CLASS, fmtf(GUIDE_W), CHECK_STEPS);
  static float b_t[512], a_t[512], ab_t[512];
  for (int i = 0; i <= T; i++) { b_t[i] = (0.02f - 1e-4f) * i / T + 1e-4f; a_t[i] = 1 - b_t[i]; }
  ab_t[0] = 1; for (int i = 1; i <= T; i++) ab_t[i] = ab_t[i - 1] * a_t[i];
  for (int i = T; i > 0 && step < STEPS; i--, step++) {
    long c0 = rdcycle(); predict(i, eps); long c1 = rdcycle(); total += c1 - c0;
    if (step < CHECK_STEPS && use_hw) { use_hw = 0; predict(i, eps_ref); use_hw = 1; printf("step %d (t=%d) ", step, i); fail |= compare("hwacha vs scalar", eps, eps_ref, NPIX, 0.02f); }
    else if (step % 100 == 0) printf("step %d (t=%d): %ld cycles\n", step, i, c1 - c0);
    for (int k = 0; k < NPIX; k++) {
      float z = i > 1 ? randn() : 0;
      img[k] = (1.0f / sqrtf(a_t[i])) * (img[k] - eps[k] * ((1 - a_t[i]) / sqrtf(1 - ab_t[i]))) + sqrtf(b_t[i]) * z;
    }
  }
  float lo = 0, hi = 1;
#else
  int T = N_STEPS;
  printf("sampling %d of %d steps, class %d, guidance w=%s, checking %d\n", STEPS < T ? STEPS : T, T, CLASS, fmtf(GUIDE_W), CHECK_STEPS);
  static float b_t[512], a_t[512], ab_t[512];
  for (int i = 0; i <= T; i++) { b_t[i] = (0.02f - 1e-4f) * i / T + 1e-4f; a_t[i] = 1 - b_t[i]; }
  ab_t[0] = a_t[0]; for (int i = 1; i <= T; i++) ab_t[i] = ab_t[i - 1] * a_t[i];
  for (int t = T - 1; t >= 0 && step < STEPS; t--, step++) {
    long c0 = rdcycle(); predict(t, eps); long c1 = rdcycle(); total += c1 - c0;
    if (step < CHECK_STEPS && use_hw) { use_hw = 0; predict(t, eps_ref); use_hw = 1; printf("step %d (t=%d) ", step, t); fail |= compare("hwacha vs scalar", eps, eps_ref, NPIX, 0.02f); }
    else if (step % 100 == 0) printf("step %d (t=%d): %ld cycles\n", step, t, c1 - c0);
    for (int k = 0; k < NPIX; k++) {
      img[k] = (1.0f / sqrtf(a_t[t])) * (img[k] - eps[k] * (b_t[t] / sqrtf(1 - ab_t[t])));
      if (t != 0) img[k] += sqrtf(b_t[t]) * randn();   // upstream default path uses rand_like (uniform) by mistake; its gif path (same_rand) is Gaussian
    }
  }
  float lo = 0, hi = 1;
#endif
  printf("forward cycles: total %ld, per step %ld\n", total, total / (step ? step : 1));
  print_image(img, lo, hi);
  printf(fail ? "%s FAIL\n" : "%s PASS\n", MODEL_NAME);
  return fail;
}

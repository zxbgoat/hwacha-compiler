// diffusion.c (hku) on Hwacha: DDPM sampling of a 16x16 sprite with the ContextUnet.
// The reference forward is the original diffusion.c code (scalar, unpadded); the Hwacha forward
// runs the same network with the kernels of diffusion.cl on zero-padded planes. A deterministic
// LCG replaces rand() so the x86 build (-DX86: reference only) and the Hwacha build draw the same
// noise and can be compared step by step.
//   STEPS        denoising steps to run (default 200, the setting of diffusion.c)
//   CHECK_STEPS  steps whose predicted noise is checked against the scalar reference (default 2)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifndef STEPS
#define STEPS 200
#endif
#ifndef CHECK_STEPS
#define CHECK_STEPS 2
#endif
#define OCB 8   // output channels per conv3x3 pass (must match diffusion.cl)
#ifdef DEBUG_STAGES
#define STAGE(name) printf("  %s\n", name)
#else
#define STAGE(name)
#endif
#ifndef CLASS
#define CLASS 0   // 0 human, 1 non-human, 2 food, 3 spell, 4 side-facing
#endif

#ifdef X86
static unsigned char *model_bin;
static long rdcycle(void) { return 0; }
#else
asm(".section .rodata\n.balign 8\n.globl model_bin\nmodel_bin:\n.incbin \"ckpt.bin\"\n.previous");
extern const unsigned char model_bin[];
static long rdcycle(void) { long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
// newlib's libm references these error helpers, missing from this libm build
float __math_oflowf(unsigned sign) { return sign ? -1.0f / 0.0f : 1.0f / 0.0f; }
float __math_uflowf(unsigned sign) { return sign ? -0.0f : 0.0f; }
int *__errno(void) { static int e; return &e; }
// report an unexpected trap instead of the environment's silent tohost=1337
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  uintptr_t tval; asm volatile("csrr %0, mtval" : "=r"(tval));
  printf("TRAP cause %lx epc %lx tval %lx ra %lx sp %lx\n", cause, epc, tval, regs[1], regs[2]);
  exit(1);
}
#endif

// ------------------------------------------------------------------ deterministic noise
static uint64_t rng = 0x9E3779B97F4A7C15ull;
static float frand(void) { rng = rng * 6364136223846793005ull + 1442695040888963407ull; return (float)((rng >> 40) & 0xFFFFFF) / 16777216.0f; }
static float randn(void) {
  float u1 = frand(), u2 = frand();
  if (u1 < 1e-7f) u1 = 1e-7f;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

// ------------------------------------------------------------------ model
typedef struct { int width, height, n_in, n_feature, n_cfeature; } Config;
typedef struct { const float *weight, *bias; } LayerParams;
typedef struct { const float *weight, *bias, *weight2, *bias2, *weight3, *bias3; } ResidualConvParams;
typedef struct { const float *weight, *bias, *weight2, *bias2; } OutParams;
typedef struct { LayerParams b1, b2; } EmbFCParams;
typedef struct { ResidualConvParams b1, b2; } UnetDownParams;
typedef struct { LayerParams b0; ResidualConvParams b1, b2; } UnetUpParams;
typedef struct {
  ResidualConvParams block0; UnetDownParams down1, down2;
  EmbFCParams temb1, temb2, cemb1, cemb2;
  LayerParams up0; UnetUpParams up1, up2; OutParams out;
} ContextUnetParams;
static Config cfg;
static ContextUnetParams W;
#define SIZE_K 3
#define SIZE_P 1

static const float *init_fc(LayerParams *w, int n_in, int n_out, const float *p) { w->weight = p; p += n_in * n_out; w->bias = p; return p + n_out; }
static const float *init_embfc(EmbFCParams *w, int n_in, int n_out, const float *p) { p = init_fc(&w->b1, n_in, n_out, p); return init_fc(&w->b2, n_out, n_out, p); }
static const float *init_out(OutParams *w, int n_in, int n_out, const float *p) {
  w->weight = p; p += 2 * n_out * n_out * 9; w->bias = p; p += n_out; w->weight2 = p; p += n_out * n_in * 9; w->bias2 = p; return p + n_in;
}
static const float *init_convtrans(LayerParams *w, int n_in, int n_f, int k, const float *p) { w->weight = p; p += n_in * n_f * k * k; w->bias = p; return p + n_f; }
static const float *init_residual(ResidualConvParams *w, int n_in, int n_f, int is_res, const float *p) {
  w->weight = p; p += n_in * n_f * 9; w->bias = p; p += n_f; w->weight2 = p; p += n_f * n_f * 9; w->bias2 = p; p += n_f;
  w->weight3 = w->bias3 = NULL;
  if (is_res && n_in != n_f) { w->weight3 = p; p += n_in * n_f; w->bias3 = p; p += n_f; }
  return p;
}
static const float *init_down(UnetDownParams *w, int n_in, int n_f, const float *p) { p = init_residual(&w->b1, n_in, n_f, 0, p); return init_residual(&w->b2, n_f, n_f, 0, p); }
static const float *init_up(UnetUpParams *w, int n_in, int n_f, const float *p) {
  p = init_convtrans(&w->b0, n_in, n_f, 2, p); p = init_residual(&w->b1, n_f, n_f, 0, p); return init_residual(&w->b2, n_f, n_f, 0, p);
}
static void load_model(void) {
  memcpy(&cfg, model_bin, sizeof cfg);
  const float *p = (const float *)(model_bin + sizeof cfg);
  int nf = cfg.n_feature;
  p = init_residual(&W.block0, cfg.n_in, nf, 1, p);
  p = init_down(&W.down1, nf, nf, p);
  p = init_down(&W.down2, nf, 2 * nf, p);
  p = init_embfc(&W.temb1, 1, 2 * nf, p);
  p = init_embfc(&W.temb2, 1, nf, p);
  p = init_embfc(&W.cemb1, cfg.n_cfeature, 2 * nf, p);
  p = init_embfc(&W.cemb2, cfg.n_cfeature, nf, p);
  p = init_convtrans(&W.up0, 2 * nf, 2 * nf, cfg.height / 4, p);
  p = init_up(&W.up1, 4 * nf, nf, p);
  p = init_up(&W.up2, 2 * nf, nf, p);
  p = init_out(&W.out, cfg.n_in, nf, p);
}

// ------------------------------------------------------------------ reference forward (diffusion.c, verbatim semantics)
#define RARENA (1 << 19)
static float rarena[RARENA]; static long rtop;
static float *ralloc(long n) { float *p = rarena + rtop; rtop += (n + 3) & ~3L; if (rtop > RARENA) { printf("reference arena overflow\n"); exit(1); } return p; }
static float *rcalloc(long n) { float *p = ralloc(n); memset(p, 0, n * sizeof(float)); return p; }
typedef struct { int C, H, W, size; } Shape3;

static float get_value3(const float *in, int ic, int i, int j, int H, int Wd, int p) {
  int x = i - p, y = j - p;
  if (x < 0 || x >= H || y < 0 || y >= Wd) return 0.0f;
  return in[ic * H * Wd + x * Wd + y];
}
static float get_value4(const float *in, int oc, int ic, int i, int j, int D, int H, int Wd, int p) {
  int x = i - p, y = j - p;
  if (x < 0 || x >= H || y < 0 || y >= Wd) return 0.0f;
  return in[oc * D * H * Wd + ic * H * Wd + x * Wd + y];
}
static float compute_mean(const float *a, int n) { float s = 0.0; for (int i = 0; i < n; i++) s += a[i]; return s / n; }
static float compute_std_dev(const float *a, int n, float mean) { float eps = 1e-05, s = 0.0; for (int i = 0; i < n; i++) s += (a[i] - mean) * (a[i] - mean); return sqrt(s / n + eps); }
static float *r_add(const float *a, const float *b, int n) { float *c = ralloc(n); for (int i = 0; i < n; i++) c[i] = a[i] + b[i]; return c; }
static float *r_concat(const float *x, const float *y, int Cx, int Cy, int H, int Wd) {
  float *z = ralloc((Cx + Cy) * H * Wd); memcpy(z, x, Cx * H * Wd * sizeof(float)); memcpy(z + Cx * H * Wd, y, Cy * H * Wd * sizeof(float)); return z;
}
static void r_gelu(float *xx, int L) {
  float a = 0.044715, s2p = sqrt(2.0 / M_PI);
  for (int i = 0; i < L; i++) { float x = xx[i]; float t = s2p * (x + a * x * x * x); xx[i] = 0.5 * x * (1.0 + tanh(t)); }
}
static void r_relu(float *xx, int L) { for (int i = 0; i < L; i++) if (xx[i] < 0) xx[i] = 0; }
static void r_elm_linear(float *x, const float *w, const float *b, int C, int HW) { for (int c = 0; c < C; c++) for (int i = 0; i < HW; i++) x[c * HW + i] = x[c * HW + i] * w[c] + b[c]; }
static int r_linear(const float *w, const float *b, const float *x, float **y, int n_out, int n_in) {
  *y = ralloc(n_out);
  for (int i = 0; i < n_out; i++) { float s = 0.0; for (int j = 0; j < n_in; j++) s += x[j] * w[i * n_in + j]; (*y)[i] = s + b[i]; }
  return n_out;
}
static int r_embfc(const EmbFCParams *w, const float *x, float **y, int n_in, int n_out) {
  float *y1; int n1 = r_linear(w->b1.weight, w->b1.bias, x, &y1, n_out, n_in); r_gelu(y1, n1);
  return r_linear(w->b2.weight, w->b2.bias, y1, y, n_out, n_out);
}
static Shape3 r_conv2d(const float *w, const float *b, const float *x, float **y, int H, int Wd, int n_in, int n_out, int k, int pd) {
  int Hy = H + 2 * pd - (k - 1), Wy = Wd + 2 * pd - (k - 1);
  *y = rcalloc(n_out * Hy * Wy);
  for (int oc = 0; oc < n_out; oc++)
    for (int i = 0; i < Hy; i++)
      for (int j = 0; j < Wy; j++) {
        float sum = 0.0f;
        for (int ic = 0; ic < n_in; ic++)
          for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++)
              sum += get_value3(x, ic, i + ki, j + kj, H, Wd, pd) * get_value4(w, oc, ic, ki, kj, n_in, k, k, 0);
        (*y)[oc * Hy * Wy + i * Wy + j] = sum + b[oc];
      }
  Shape3 s = {n_out, Hy, Wy, n_out * Hy * Wy}; return s;
}
static Shape3 r_convtrans2d(const float *w, const float *b, const float *x, float **y, int H, int Wd, int n_in, int n_out, int k, int st) {
  int Hy = (H - 1) * st + k, Wy = (Wd - 1) * st + k;
  *y = rcalloc(n_out * Hy * Wy);
  for (int oc = 0; oc < n_out; oc++)
    for (int ic = 0; ic < n_in; ic++)
      for (int i = 0; i < H; i++)
        for (int j = 0; j < Wd; j++)
          for (int m = 0; m < k; m++)
            for (int n = 0; n < k; n++)
              (*y)[oc * Hy * Wy + (i * st + m) * Wy + (j * st + n)] += get_value3(x, ic, i, j, H, Wd, 0) * get_value4(w, ic, oc, m, n, n_out, k, k, 0);
  for (int oc = 0; oc < n_out; oc++) for (int i = 0; i < Hy * Wy; i++) (*y)[oc * Hy * Wy + i] += b[oc];
  Shape3 s = {n_out, Hy, Wy, n_out * Hy * Wy}; return s;
}
static void r_batchnorm(float *x, int C, int H, int Wd) {
  float eps = 1e-05;
  for (int c = 0; c < C; c++) {
    float mean = 0.0, var = 0.0;
    for (int i = 0; i < H * Wd; i++) mean += x[c * H * Wd + i];
    mean /= (H * Wd);
    for (int i = 0; i < H * Wd; i++) { float d = x[c * H * Wd + i] - mean; var += d * d; }
    var /= (H * Wd);
    for (int i = 0; i < H * Wd; i++) x[c * H * Wd + i] = (x[c * H * Wd + i] - mean) / sqrt(var + eps);
  }
}
static void r_groupnorm(float *x, int C, int G, int H, int Wd) {
  int cpg = C / G, n = cpg * H * Wd;
  for (int g = 0; g < G; g++) {
    float mean = compute_mean(x + g * n, n), sd = compute_std_dev(x + g * n, n, mean);
    for (int i = 0; i < n; i++) x[g * n + i] = (x[g * n + i] - mean) / sd;
  }
}
static Shape3 r_resblock(const ResidualConvParams *w, const float *x, float **y, int H, int Wd, int n_in, int n_out, int is_res) {
  float *y2; Shape3 s2 = r_conv2d(w->weight, w->bias, x, &y2, H, Wd, n_in, n_out, SIZE_K, SIZE_P);
  r_batchnorm(y2, s2.C, s2.H, s2.W); r_gelu(y2, s2.size);
  Shape3 s = r_conv2d(w->weight2, w->bias2, y2, y, s2.H, s2.W, s2.C, n_out, SIZE_K, SIZE_P);
  r_batchnorm(*y, s.C, s.H, s.W); r_gelu(*y, s.size);
  if (is_res) {
    if (n_in == n_out) *y = r_add(*y, x, H * Wd * n_out);
    else { float *sc; Shape3 ss = r_conv2d(w->weight3, w->bias3, x, &sc, H, Wd, n_in, n_out, 1, 0); *y = r_add(*y, sc, ss.size); }
    for (int i = 0; i < s.size; i++) (*y)[i] = (*y)[i] / 1.414;
  }
  return s;
}
static Shape3 r_maxpool(const float *x, float **y, int H, int Wd, int C, int k) {
  int Hy = H / k, Wy = Wd / k; *y = ralloc(C * Hy * Wy);
  for (int c = 0; c < C; c++)
    for (int i = 0; i < H; i += k)
      for (int j = 0; j < Wd; j += k) {
        float m = x[c * H * Wd + i * Wd + j];
        for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) { float v = x[c * H * Wd + (i + a) * Wd + j + b]; if (v > m) m = v; }
        (*y)[c * Hy * Wy + (i / k) * Wy + j / k] = m;
      }
  Shape3 s = {C, Hy, Wy, C * Hy * Wy}; return s;
}
static Shape3 r_avgpool(const float *x, float **y, int H, int Wd, int C, int k) {
  int Hy = H / k, Wy = Wd / k; *y = ralloc(C * Hy * Wy);
  for (int c = 0; c < C; c++)
    for (int i = 0; i < H; i += k)
      for (int j = 0; j < Wd; j += k) {
        float sum = 0;
        for (int a = 0; a < k; a++) for (int b = 0; b < k; b++) sum += x[c * H * Wd + (i + a) * Wd + j + b];
        (*y)[c * Hy * Wy + (i / k) * Wy + j / k] = sum / (k * k);
      }
  Shape3 s = {C, Hy, Wy, C * Hy * Wy}; return s;
}
static Shape3 r_down(const UnetDownParams *w, const float *x, float **y, int H, int Wd, int n_in, int n_out) {
  float *y1, *y2;
  Shape3 s1 = r_resblock(&w->b1, x, &y1, H, Wd, n_in, n_out, 0);
  Shape3 s2 = r_resblock(&w->b2, y1, &y2, s1.H, s1.W, s1.C, n_out, 0);
  return r_maxpool(y2, y, s2.H, s2.W, s2.C, 2);
}
static Shape3 r_up(const UnetUpParams *w, const float *x, const float *skip, float **y, int H, int Wd, int n_in, int n_out) {
  float *x1 = r_concat(x, skip, n_in / 2, n_in / 2, H, Wd), *y1, *y2;
  Shape3 s1 = r_convtrans2d(w->b0.weight, w->b0.bias, x1, &y1, H, Wd, n_in, n_out, 2, 2);
  Shape3 s2 = r_resblock(&w->b1, y1, &y2, s1.H, s1.W, s1.C, n_out, 0);
  return r_resblock(&w->b2, y2, y, s2.H, s2.W, s2.C, n_out, 0);
}
// eps = ContextUnet(x, t, c); x unpadded [3][16][16]
static void ref_forward(const float *x, float t, const float *c, float *eps) {
  rtop = 0;
  STAGE("ref block0");
  int H = cfg.height, Wd = cfg.width, nf = cfg.n_feature;
  float *y1, *d1, *d2, *v, *up0, *temb1, *cemb1, *up1, *temb2, *cemb2, *up2, *o1, *out;
  Shape3 s1 = r_resblock(&W.block0, x, &y1, H, Wd, cfg.n_in, nf, 1);
  STAGE("ref down1"); Shape3 sd1 = r_down(&W.down1, y1, &d1, s1.H, s1.W, s1.C, nf);
  Shape3 sd2 = r_down(&W.down2, d1, &d2, sd1.H, sd1.W, sd1.C, 2 * nf);
  Shape3 sv = r_avgpool(d2, &v, sd2.H, sd2.W, sd2.C, 4); r_gelu(v, sv.size);
  Shape3 su0 = r_convtrans2d(W.up0.weight, W.up0.bias, v, &up0, sv.H, sv.W, sv.C, sv.C, H / 4, H / 4);
  r_groupnorm(up0, su0.C, 8, su0.H, su0.W); r_relu(up0, su0.size);
  r_embfc(&W.temb1, &t, &temb1, 1, su0.C); r_embfc(&W.cemb1, c, &cemb1, cfg.n_cfeature, su0.C);
  r_elm_linear(up0, cemb1, temb1, su0.C, su0.H * su0.W);
  STAGE("ref up1"); Shape3 su1 = r_up(&W.up1, up0, d2, &up1, su0.H, su0.W, su0.C + sd2.C, nf);
  r_embfc(&W.temb2, &t, &temb2, 1, su1.C); r_embfc(&W.cemb2, c, &cemb2, cfg.n_cfeature, su1.C);
  r_elm_linear(up1, cemb2, temb2, su1.C, su1.H * su1.W);
  Shape3 su2 = r_up(&W.up2, up1, d1, &up2, su1.H, su1.W, su1.C + sd1.C, nf);
  STAGE("ref out"); float *x1 = r_concat(up2, y1, nf, nf, H, Wd);
  Shape3 so1 = r_conv2d(W.out.weight, W.out.bias, x1, &o1, H, Wd, 2 * nf, nf, SIZE_K, SIZE_P);
  r_groupnorm(o1, nf, 8, so1.H, so1.W); r_relu(o1, so1.size);
  Shape3 so = r_conv2d(W.out.weight2, W.out.bias2, o1, &out, so1.H, so1.W, so1.C, cfg.n_in, SIZE_K, SIZE_P);
  memcpy(eps, out, so.size * sizeof(float));
  (void)su2;
}

// ------------------------------------------------------------------ Hwacha forward (padded planes)
#if !defined(X86) && !defined(REF_ONLY)
void conv3x3_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int Wp, int oc0);
void conv3x3_1_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int Wp, int oc);
void conv1x1_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int plane, int oc0);
void chan_sum_ct(long n, const float *x, int plane, int Wp, int H, int W, float *out);
void chan_sqdiff_ct(long n, const float *x, int plane, int Wp, int H, int W, const float *mean, int cpg, float *out);
void bn_gelu_ct(long n, float *x, int plane, int Wp, int H, int W, const float *mean, const float *rstd);
void bn_gelu_res_ct(long n, float *x, int plane, int Wp, int H, int W, const float *mean, const float *rstd, const float *res);
void gn_relu_ct(long n, float *x, int plane, int Wp, int H, int W, const float *mean, const float *rstd, int cpg);
void maxpool2_ct(long n, const float *x, int plane, int Wp, int H, int W, float *y, int plane2, int Wp2);
void avgpool4_gelu_ct(long n, const float *x, int plane, int Wp, float *v);
void up0_gemv_ct(long n, const float *v, const float *w, const float *b, float *y, int n_in, int n_out, int plane2, int Wp2);
void convT2_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int n_out, int plane, int Wp, int W, int plane2, int Wp2, int oc0);
void fc_ct(long n, const float *x, const float *w, const float *b, float *y, int n_in, int act);
void elm_linear_ct(long n, float *x, int plane, int Wp, int H, int W, const float *cemb, const float *temb);
void vcopy_ct(long n, const float *src, float *dst);
void vzero_ct(long n, float *dst);

typedef struct { float *d; int C, H, W, Hp, Wp, plane; } Act;
#define HARENA (1 << 19)
static float harena[HARENA]; static long htop;
static Act new_act(int C, int H, int Wd) {
  Act a; a.C = C; a.H = H; a.W = Wd; a.Hp = H + 2; a.Wp = Wd + 2; a.plane = a.Hp * a.Wp;
  int guard = 2 * a.Wp + 8;
  htop = (htop + 15) & ~15L;
  a.d = harena + htop + guard;
  htop += (long)C * a.plane + 2 * guard;
  if (htop > HARENA) { printf("hwacha arena overflow\n"); exit(1); }
  return a;
}
static float stat_a[256], stat_b[256], stat_m[256], stat_r[256];

static void hw_conv3x3(Act x, const float *w, const float *b, Act y, int n_out) {
  int oc = 0;
  for (; oc + OCB <= n_out; oc += OCB) conv3x3_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, oc);
  for (; oc < n_out; oc++) conv3x3_1_ct(x.plane, x.d, w, b, y.d, x.C, x.plane, x.Wp, oc);
}
// batch norm + GELU (+ residual): mean / rstd per channel over the interior
static void hw_bn(Act y, const float *res) {
  int HW = y.H * y.W;
  chan_sum_ct(y.C, y.d, y.plane, y.Wp, y.H, y.W, stat_a);
  for (int c = 0; c < y.C; c++) stat_m[c] = stat_a[c] / HW;
  chan_sqdiff_ct(y.C, y.d, y.plane, y.Wp, y.H, y.W, stat_m, 1, stat_b);
  for (int c = 0; c < y.C; c++) stat_r[c] = 1.0f / sqrtf(stat_b[c] / HW + 1e-5f);
  if (res) bn_gelu_res_ct((long)y.C * y.plane, y.d, y.plane, y.Wp, y.H, y.W, stat_m, stat_r, res);
  else bn_gelu_ct((long)y.C * y.plane, y.d, y.plane, y.Wp, y.H, y.W, stat_m, stat_r);
}
// group norm (8 groups) + ReLU
static void hw_gn(Act y) {
  int G = 8, cpg = y.C / G, n = cpg * y.H * y.W;
  chan_sum_ct(y.C, y.d, y.plane, y.Wp, y.H, y.W, stat_a);
  for (int g = 0; g < G; g++) { float s = 0; for (int c = 0; c < cpg; c++) s += stat_a[g * cpg + c]; stat_m[g] = s / n; }
  chan_sqdiff_ct(y.C, y.d, y.plane, y.Wp, y.H, y.W, stat_m, cpg, stat_b);
  for (int g = 0; g < G; g++) { float s = 0; for (int c = 0; c < cpg; c++) s += stat_b[g * cpg + c]; stat_r[g] = 1.0f / sqrtf(s / n + 1e-5f); }
  gn_relu_ct((long)y.C * y.plane, y.d, y.plane, y.Wp, y.H, y.W, stat_m, stat_r, cpg);
}
static Act hw_resblock(const ResidualConvParams *w, Act x, int n_out, int is_res) {
  Act t = new_act(n_out, x.H, x.W);
  STAGE("   conv3x3 a"); hw_conv3x3(x, w->weight, w->bias, t, n_out);
  STAGE("   bn a"); hw_bn(t, NULL);
  Act y = new_act(n_out, x.H, x.W);
  STAGE("   conv3x3 b"); hw_conv3x3(t, w->weight2, w->bias2, y, n_out);
  STAGE("   bn b"); if (!is_res) { hw_bn(y, NULL); return y; }
  Act sc = x;
  if (x.C != n_out) {
    sc = new_act(n_out, x.H, x.W);
    for (int oc = 0; oc < n_out; oc += OCB) conv1x1_ct(x.plane, x.d, w->weight3, w->bias3, sc.d, x.C, x.plane, oc);
  }
  hw_bn(y, sc.d);
  return y;
}
static Act hw_down(const UnetDownParams *w, Act x, int n_out) {
  Act y1 = hw_resblock(&w->b1, x, n_out, 0);
  Act y2 = hw_resblock(&w->b2, y1, n_out, 0);
  Act y = new_act(n_out, x.H / 2, x.W / 2);
  maxpool2_ct((long)y.C * y.plane, y2.d, y2.plane, y2.Wp, y2.H, y2.W, y.d, y.plane, y.Wp);
  return y;
}
static Act hw_concat(Act x, Act skip) {
  Act c = new_act(x.C + skip.C, x.H, x.W);
  vcopy_ct((long)x.C * x.plane, x.d, c.d);
  vcopy_ct((long)skip.C * skip.plane, skip.d, c.d + (long)x.C * x.plane);
  return c;
}
static Act hw_up(const UnetUpParams *w, Act x, Act skip, int n_out) {
  Act cat = hw_concat(x, skip);
  Act y1 = new_act(n_out, 2 * x.H, 2 * x.W);
  vzero_ct((long)y1.C * y1.plane, y1.d);
  for (int oc = 0; oc < n_out; oc += 4)
    convT2_ct((long)cat.H * cat.W, cat.d, w->b0.weight, w->b0.bias, y1.d, cat.C, n_out, cat.plane, cat.Wp, cat.W, y1.plane, y1.Wp, oc);
  Act y2 = hw_resblock(&w->b1, y1, n_out, 0);
  return hw_resblock(&w->b2, y2, n_out, 0);
}
static void hw_embfc(const EmbFCParams *w, const float *x, int n_in, int n_out, float *out, float *tmp) {
  fc_ct(n_out, x, w->b1.weight, w->b1.bias, tmp, n_in, 1);
  fc_ct(n_out, tmp, w->b2.weight, w->b2.bias, out, n_out, 0);
}
static float vvec[256], temb[256], cemb[256], etmp[256], tvec[1], cvec[8];
static void hw_forward(const float *x, float t, const float *c, float *eps) {
  htop = 0;
  int H = cfg.height, Wd = cfg.width, nf = cfg.n_feature;
  Act xp = new_act(cfg.n_in, H, Wd);
  memset(xp.d, 0, (long)xp.C * xp.plane * sizeof(float));
  for (int ch = 0; ch < cfg.n_in; ch++) for (int i = 0; i < H; i++) for (int j = 0; j < Wd; j++) xp.d[ch * xp.plane + (i + 1) * xp.Wp + j + 1] = x[ch * H * Wd + i * Wd + j];
  STAGE("block0"); Act y1 = hw_resblock(&W.block0, xp, nf, 1);
  STAGE("down1"); Act d1 = hw_down(&W.down1, y1, nf);
  STAGE("down2"); Act d2 = hw_down(&W.down2, d1, 2 * nf);
  STAGE("to_vec"); avgpool4_gelu_ct(d2.C, d2.d, d2.plane, d2.Wp, vvec);
  Act up0 = new_act(2 * nf, 4, 4);
  up0_gemv_ct((long)up0.C * up0.plane, vvec, W.up0.weight, W.up0.bias, up0.d, 2 * nf, 2 * nf, up0.plane, up0.Wp);
  STAGE("gn up0"); hw_gn(up0);
  STAGE("emb1"); tvec[0] = t; memcpy(cvec, c, cfg.n_cfeature * sizeof(float));
  hw_embfc(&W.temb1, tvec, 1, 2 * nf, temb, etmp); hw_embfc(&W.cemb1, cvec, cfg.n_cfeature, 2 * nf, cemb, etmp);
  elm_linear_ct((long)up0.C * up0.plane, up0.d, up0.plane, up0.Wp, up0.H, up0.W, cemb, temb);
  STAGE("up1"); Act up1 = hw_up(&W.up1, up0, d2, nf);
  hw_embfc(&W.temb2, tvec, 1, nf, temb, etmp); hw_embfc(&W.cemb2, cvec, cfg.n_cfeature, nf, cemb, etmp);
  elm_linear_ct((long)up1.C * up1.plane, up1.d, up1.plane, up1.Wp, up1.H, up1.W, cemb, temb);
  STAGE("up2"); Act up2 = hw_up(&W.up2, up1, d1, nf);
  STAGE("out"); Act cat = hw_concat(up2, y1);
  Act o1 = new_act(nf, H, Wd);
  STAGE("   out conv3x3"); hw_conv3x3(cat, W.out.weight, W.out.bias, o1, nf);
  STAGE("   out gn"); hw_gn(o1);
  Act out = new_act(cfg.n_in, H, Wd);
  STAGE("   out conv3x3_1"); hw_conv3x3(o1, W.out.weight2, W.out.bias2, out, cfg.n_in);
  STAGE("   unpad");
  for (int ch = 0; ch < cfg.n_in; ch++) for (int i = 0; i < H; i++) for (int j = 0; j < Wd; j++) eps[ch * H * Wd + i * Wd + j] = out.d[ch * out.plane + (i + 1) * out.Wp + j + 1];
}
#else
#define hw_forward ref_forward
#endif

// ------------------------------------------------------------------ sampler
static void denoise_add_noise(float *x, const float *pred, float a_t, float b_t, float ab_t, int n, int add_noise) {
  for (int i = 0; i < n; i++) {
    float z = add_noise ? randn() : 0.0f;
    float noise = sqrt(b_t) * z;
    float mean = (x[i] - pred[i] * ((1 - a_t) / sqrt(1 - ab_t))) / sqrt(a_t);
    x[i] = mean + noise;
  }
}
static float x[3 * 16 * 16], eps[3 * 16 * 16], eps_ref[3 * 16 * 16];
// the bare-metal printf has no %g: print a float as a fixed-point string with 6 decimals
static const char *fmtf(float v) {
  static char buf[4][32]; static int k;
  char *b = buf[k++ & 3];
  long m = (long)(fabsf(v) * 1e6f + 0.5f), ip = m / 1000000, fp = m % 1000000;
  char *q = b;
  if (v < 0) *q++ = '-';
  char tmp[24]; int n = 0;
  do { tmp[n++] = '0' + ip % 10; ip /= 10; } while (ip);
  while (n) *q++ = tmp[--n];
  *q++ = '.';
  for (long d = 100000; d; d /= 10) *q++ = '0' + (fp / d) % 10;
  *q = 0;
  return b;
}

#if defined(LAYER_BENCH) && !defined(X86)
// one conv3x3 layer (64 -> LB_OUT channels, 16x16) scalar vs Hwacha, same data: the per-layer speedup
#ifndef LB_OUT
#define LB_OUT 16
#endif
static float lb_x[64 * 256], lb_y[64 * 256];
int main(void) {
  load_model();
  int nf = cfg.n_feature, H = cfg.height, Wd = cfg.width;
  for (int i = 0; i < nf * H * Wd; i++) lb_x[i] = randn();
  const float *w = W.down1.b1.weight, *b = W.down1.b1.bias;
  float *yr; rtop = 0;
  long c0 = rdcycle(); r_conv2d(w, b, lb_x, &yr, H, Wd, nf, LB_OUT, SIZE_K, SIZE_P); long c1 = rdcycle();
  htop = 0;
  Act xp = new_act(nf, H, Wd), yp = new_act(LB_OUT, H, Wd);
  memset(xp.d, 0, (long)xp.C * xp.plane * sizeof(float));
  for (int ch = 0; ch < nf; ch++) for (int i = 0; i < H; i++) for (int j = 0; j < Wd; j++) xp.d[ch * xp.plane + (i + 1) * xp.Wp + j + 1] = lb_x[ch * H * Wd + i * Wd + j];
  long c2 = rdcycle(); hw_conv3x3(xp, w, b, yp, LB_OUT); long c3 = rdcycle();
  float md = 0, mx = 0;
  for (int ch = 0; ch < LB_OUT; ch++) for (int i = 0; i < H; i++) for (int j = 0; j < Wd; j++) {
    float r = yr[ch * H * Wd + i * Wd + j], h = yp.d[ch * yp.plane + (i + 1) * yp.Wp + j + 1];
    float d = fabsf(r - h); if (d > md) md = d; if (fabsf(r) > mx) mx = fabsf(r);
  }
  long macs = (long)nf * LB_OUT * H * Wd * 9;
  printf("conv3x3 %dx%d %d->%d (%ld MAC): scalar %ld cycles, hwacha %ld cycles, %ld.%02ldx; max diff %s of %s, %s\n",
         H, Wd, nf, LB_OUT, macs, c1 - c0, c3 - c2, (c1 - c0) / (c3 - c2), ((c1 - c0) * 100 / (c3 - c2)) % 100, fmtf(md), fmtf(mx), md <= 0.02f * mx ? "PASS" : "FAIL");
  return md > 0.02f * mx;
}
#else
int main(int argc, char **argv) {
#ifdef X86
  const char *path = argc > 1 ? argv[1] : "ckpt.bin";
  FILE *f = fopen(path, "rb"); if (!f) { printf("cannot open %s\n", path); return 1; }
  fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
  model_bin = malloc(sz); if (fread(model_bin, 1, sz, f) != (size_t)sz) return 1; fclose(f);
#endif
  load_model();
  int n = cfg.n_in * cfg.height * cfg.width, T = 200;
  printf("diffusion.c ContextUnet %dx%d n_feature %d, class %d, %d of %d steps, checking %d\n", cfg.height, cfg.width, cfg.n_feature, CLASS, STEPS, T, CHECK_STEPS);
  float c[8] = {0}; c[CLASS] = 1.0f;
  float b_t[201], a_t[201], ab_t[201];
  for (int i = 0; i <= T; i++) { float tt = (float)i / T; b_t[i] = (0.02f - 1e-4f) * tt + 1e-4f; a_t[i] = 1 - b_t[i]; }
  ab_t[0] = 1; for (int i = 1; i <= T; i++) ab_t[i] = ab_t[i - 1] * a_t[i];
  for (int i = 0; i < n; i++) x[i] = randn();
  int fail = 0, step = 0;
  long total = 0;
  for (int i = T; i > 0 && step < STEPS; i--, step++) {
    float t = (float)i / T;
    long c0 = rdcycle();
    hw_forward(x, t, c, eps);
    long c1 = rdcycle();
    total += c1 - c0;
    if (step < CHECK_STEPS) {
#ifndef X86
      ref_forward(x, t, c, eps_ref);
      float md = 0, mx = 0;
      for (int k = 0; k < n; k++) { float d = fabsf(eps[k] - eps_ref[k]); if (d > md) md = d; if (fabsf(eps_ref[k]) > mx) mx = fabsf(eps_ref[k]); }
      int ok = md <= 0.02f * mx;
      printf("step %d (t=%d): max |eps - ref| = %s, max |ref| = %s, %s, %ld cycles\n", step, i, fmtf(md), fmtf(mx), ok ? "ok" : "FAIL", c1 - c0);
      if (!ok) fail = 1;
#else
      printf("step %d (t=%d)\n", step, i);
#endif
    } else if (step % 20 == 0 || step == STEPS - 1) printf("step %d (t=%d): %ld cycles\n", step, i, c1 - c0);
    denoise_add_noise(x, eps, a_t[i], b_t[i], ab_t[i], n, i > 1);
  }
  printf("forward cycles: total %ld, per step %ld\n", total, total / (step ? step : 1));
  // final image, per-channel min-max normalized like diffusion.c, one hex RGB triplet per pixel
  int HW = cfg.height * cfg.width;
  float mn[3], mxv[3];
  for (int ch = 0; ch < 3; ch++) { mn[ch] = 1e9f; mxv[ch] = -1e9f; for (int k = 0; k < HW; k++) { float v = x[ch * HW + k]; if (v < mn[ch]) mn[ch] = v; if (v > mxv[ch]) mxv[ch] = v; } }
  printf("image %dx%d:\n", cfg.height, cfg.width);
  for (int i = 0; i < cfg.height; i++) {
    for (int j = 0; j < cfg.width; j++) {
      int k = i * cfg.width + j;
      for (int ch = 0; ch < 3; ch++) printf("%02x", (int)((x[ch * HW + k] - mn[ch]) / (mxv[ch] - mn[ch]) * 255));
      printf(j + 1 < cfg.width ? " " : "\n");
    }
  }
  printf(fail ? "diffusion FAIL\n" : "diffusion PASS\n");
  return fail;
}
#endif

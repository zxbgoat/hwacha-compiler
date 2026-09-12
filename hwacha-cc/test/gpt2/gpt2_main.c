// GPT-2 forward (llm.c) on a small random-initialised model: verbatim scalar functions vs hwacha-cc kernels.
// Build with -DX86 on the host to print the reference lines for cross-checking the RISC-V scalar path.
#include <stdio.h>
#include <math.h>
#include <string.h>
#ifdef X86
#include <stdlib.h>
static unsigned rng_state = 20240911u;
static inline unsigned rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }
static inline float frand(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xffff) / 65535.0f; }
static inline unsigned long cyc(void) { return 0; }
#define REPORT(tag, c0, c1, n) printf("%s: (x86, no cycles)\n", tag)
#else
#include "../apps/common.h"
// newlib's tanhf/expm1f reference these error helpers, missing from this libm build
float __math_oflowf(unsigned sign) { return sign ? -1.0f / 0.0f : 1.0f / 0.0f; }
float __math_uflowf(unsigned sign) { return sign ? -0.0f : 0.0f; }
#endif
#define V 1024
#define C 64
#define NH 4
#define L 2
#define T 16
#define MAXT 64
#define HS (C / NH)
#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

// ---------------- parameters (random init, deterministic) ----------------
static float wte[V * C], wpe[MAXT * C];
typedef struct { float ln1w[C], ln1b[C], qkvw[3 * C * C], qkvb[3 * C], attprojw[C * C], attprojb[C], ln2w[C], ln2b[C], fcw[4 * C * C], fcb[4 * C], fcprojw[C * 4 * C], fcprojb[C]; } Layer;
static Layer layers[L];
static float lnfw[C], lnfb[C];
static int tokens[T];
static void init_params(void) {
  for (int i = 0; i < V * C; i++) wte[i] = frand(-0.05f, 0.05f);
  for (int i = 0; i < MAXT * C; i++) wpe[i] = frand(-0.05f, 0.05f);
  for (int l = 0; l < L; l++) {
    Layer *p = &layers[l];
    for (int i = 0; i < C; i++) { p->ln1w[i] = 1.0f + frand(-0.1f, 0.1f); p->ln1b[i] = frand(-0.1f, 0.1f); p->ln2w[i] = 1.0f + frand(-0.1f, 0.1f); p->ln2b[i] = frand(-0.1f, 0.1f); }
    for (int i = 0; i < 3 * C * C; i++) p->qkvw[i] = frand(-0.1f, 0.1f);
    for (int i = 0; i < 3 * C; i++) p->qkvb[i] = frand(-0.1f, 0.1f);
    for (int i = 0; i < C * C; i++) p->attprojw[i] = frand(-0.1f, 0.1f);
    for (int i = 0; i < C; i++) p->attprojb[i] = frand(-0.1f, 0.1f);
    for (int i = 0; i < 4 * C * C; i++) { p->fcw[i] = frand(-0.1f, 0.1f); p->fcprojw[i] = frand(-0.05f, 0.05f); }
    for (int i = 0; i < 4 * C; i++) p->fcb[i] = frand(-0.1f, 0.1f);
    for (int i = 0; i < C; i++) p->fcprojb[i] = frand(-0.1f, 0.1f);
  }
  for (int i = 0; i < C; i++) { lnfw[i] = 1.0f + frand(-0.1f, 0.1f); lnfb[i] = frand(-0.1f, 0.1f); }
  for (int t = 0; t < T; t++) tokens[t] = rnd() % V;
}

// ---------------- activations ----------------
typedef struct {
  float encoded[T * C], ln1[T * C], mean[T], rstd[T], qkv[T * 3 * C], preatt[NH * T * T], att[NH * T * T], atty[T * C], attproj[T * C],
        residual2[T * C], ln2[T * C], fch[T * 4 * C], fch_gelu[T * 4 * C], fcproj[T * C], residual3[L][T * C], lnf[T * C], logits[T * V], probs[T * V];
} Acts;
static Acts ar, ah;

// ---------------- scalar reference: verbatim llm.c ----------------
static void encoder_forward(float* out, int* inp, float* wte_, float* wpe_, int B, int T_, int C_) {
  for (int b = 0; b < B; b++) for (int t = 0; t < T_; t++) {
    float* out_bt = out + b * T_ * C_ + t * C_; int ix = inp[b * T_ + t]; float* wte_ix = wte_ + ix * C_; float* wpe_t = wpe_ + t * C_;
    for (int i = 0; i < C_; i++) out_bt[i] = wte_ix[i] + wpe_t[i];
  }
}
static void layernorm_forward(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias, int B, int T_, int C_) {
  float eps = 1e-5f;
  for (int b = 0; b < B; b++) for (int t = 0; t < T_; t++) {
    float* x = inp + b * T_ * C_ + t * C_;
    float m = 0.0f; for (int i = 0; i < C_; i++) m += x[i]; m = m/C_;
    float v = 0.0f; for (int i = 0; i < C_; i++) { float xshift = x[i] - m; v += xshift * xshift; } v = v/C_;
    float s = 1.0f / sqrtf(v + eps);
    float* out_bt = out + b * T_ * C_ + t * C_;
    for (int i = 0; i < C_; i++) { float n = (s * (x[i] - m)); float o = n * weight[i] + bias[i]; out_bt[i] = o; }
    mean[b * T_ + t] = m; rstd[b * T_ + t] = s;
  }
}
static void matmul_forward_naive(float* out, const float* inp, const float* weight, const float* bias, int B, int T_, int C_, int OC) {
  for (int b = 0; b < B; b++) for (int t = 0; t < T_; t++) {
    int bt = b * T_ + t;
    for (int o = 0; o < OC; o++) {
      float val = (bias != NULL) ? bias[o] : 0.0f;
      for (int i = 0; i < C_; i++) val += inp[bt * C_ + i] * weight[o*C_ + i];
      out[bt * OC + o] = val;
    }
  }
}
static void attention_forward(float* out, float* preatt, float* att, float* inp, int B, int T_, int C_, int NH_) {
  int C3 = C_*3; int hs = C_ / NH_; float scale = 1.0 / sqrtf(hs);
  for (int b = 0; b < B; b++) for (int t = 0; t < T_; t++) for (int h = 0; h < NH_; h++) {
    float* query_t = inp + b * T_ * C3 + t * C3 + h * hs;
    float* preatt_bth = preatt + b*NH_*T_*T_ + h*T_*T_ + t*T_;
    float* att_bth = att + b*NH_*T_*T_ + h*T_*T_ + t*T_;
    float maxval = -10000.0f;
    for (int t2 = 0; t2 <= t; t2++) {
      float* key_t2 = inp + b * T_ * C3 + t2 * C3 + h * hs + C_;
      float val = 0.0f; for (int i = 0; i < hs; i++) val += query_t[i] * key_t2[i];
      val *= scale; if (val > maxval) maxval = val;
      preatt_bth[t2] = val;
    }
    float expsum = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) { float expv = expf(preatt_bth[t2] - maxval); expsum += expv; att_bth[t2] = expv; }
    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
    for (int t2 = 0; t2 < T_; t2++) { if (t2 <= t) att_bth[t2] *= expsum_inv; else att_bth[t2] = 0.0f; }
    float* out_bth = out + b * T_ * C_ + t * C_ + h * hs;
    for (int i = 0; i < hs; i++) out_bth[i] = 0.0f;
    for (int t2 = 0; t2 <= t; t2++) {
      float* value_t2 = inp + b * T_ * C3 + t2 * C3 + h * hs + C_*2; float att_btht2 = att_bth[t2];
      for (int i = 0; i < hs; i++) out_bth[i] += att_btht2 * value_t2[i];
    }
  }
}
static void gelu_forward(float* out, float* inp, int N) {
  for (int i = 0; i < N; i++) { float x = inp[i]; float cube = 0.044715f * x * x * x; out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube))); }
}
static void residual_forward(float* out, float* inp1, float* inp2, int N) { for (int i = 0; i < N; i++) out[i] = inp1[i] + inp2[i]; }
static void softmax_forward(float* probs, float* logits, int B, int T_, int V_, int Vp) {
  for (int b = 0; b < B; b++) for (int t = 0; t < T_; t++) {
    float* logits_bt = logits + b * T_ * Vp + t * Vp; float* probs_bt = probs + b * T_ * Vp + t * Vp;
    float maxval = -10000.0f; for (int i = 0; i < V_; i++) if (logits_bt[i] > maxval) maxval = logits_bt[i];
    float sum = 0.0f; for (int i = 0; i < V_; i++) { probs_bt[i] = expf(logits_bt[i] - maxval); sum += probs_bt[i]; }
    for (int i = 0; i < V_; i++) probs_bt[i] /= sum;
    for (int i = V_; i < Vp; i++) probs_bt[i] = 0.0f;
  }
}
static void forward_ref(Acts *a) {
  encoder_forward(a->encoded, tokens, wte, wpe, 1, T, C);
  float *residual = a->encoded;
  for (int l = 0; l < L; l++) {
    Layer *p = &layers[l];
    layernorm_forward(a->ln1, a->mean, a->rstd, residual, p->ln1w, p->ln1b, 1, T, C);
    matmul_forward_naive(a->qkv, a->ln1, p->qkvw, p->qkvb, 1, T, C, 3 * C);
    attention_forward(a->atty, a->preatt, a->att, a->qkv, 1, T, C, NH);
    matmul_forward_naive(a->attproj, a->atty, p->attprojw, p->attprojb, 1, T, C, C);
    residual_forward(a->residual2, residual, a->attproj, T * C);
    layernorm_forward(a->ln2, a->mean, a->rstd, a->residual2, p->ln2w, p->ln2b, 1, T, C);
    matmul_forward_naive(a->fch, a->ln2, p->fcw, p->fcb, 1, T, C, 4 * C);
    gelu_forward(a->fch_gelu, a->fch, T * 4 * C);
    matmul_forward_naive(a->fcproj, a->fch_gelu, p->fcprojw, p->fcprojb, 1, T, 4 * C, C);
    residual_forward(a->residual3[l], a->residual2, a->fcproj, T * C);
    residual = a->residual3[l];
  }
  layernorm_forward(a->lnf, a->mean, a->rstd, residual, lnfw, lnfb, 1, T, C);
  matmul_forward_naive(a->logits, a->lnf, wte, NULL, 1, T, C, V);
  softmax_forward(a->probs, a->logits, 1, T, V, V);
}

#ifndef X86
// ---------------- hwacha-cc kernels ----------------
void encoder_ct(long n, float *out, const int *tok, const float *wte, const float *wpe, int C_);
void ln_stats_ct(long n, float *mean, float *rstd, const float *inp, int C_);
void ln_apply_ct(long n, float *out, const float *inp, const float *mean, const float *rstd, const float *w, const float *b, int C_);
void matmul_t_row_ct(long n, float *out, const float *inp, const float *WT, const float *bias, int C_, int OC);
void matmul_t_row_nb_ct(long n, float *out, const float *inp, const float *WT, int C_, int OC);
void att_score_ct(long n, float *preatt, const float *qkv, int T_, int C_, int NH_, float scale);
void att_softmax_ct(long n, float *att, const float *preatt, int T_);
void att_value_ct(long n, float *out, const float *att, const float *qkv, int T_, int C_, int NH_);
void gelu_ct(long n, float *out, const float *inp, float gelu_scale);
void residual_ct(long n, float *out, const float *a, const float *b);
void softmax_row_ct(long n, float *probs, const float *logits, int V_);
enum { K_ENC, K_LN, K_MM, K_ATT, K_GELU, K_RES, K_SMAX, K_N };
static const char *kname[K_N] = {"encoder", "layernorm", "matmul", "attention", "gelu", "residual", "softmax"};
static unsigned long kcyc[K_N];
#define KT(slot, call) do { unsigned long _a = cyc(); call; kcyc[slot] += cyc() - _a; } while (0)
// transposed weights [in][out]
static float qkvwT[L][C * 3 * C], attprojwT[L][C * C], fcwT[L][C * 4 * C], fcprojwT[L][4 * C * C], wteT[C * V];
static void transpose(float *dst, const float *src, int OC, int IC) { for (int o = 0; o < OC; o++) for (int i = 0; i < IC; i++) dst[i * OC + o] = src[o * IC + i]; }
static void make_transposed(void) {
  for (int l = 0; l < L; l++) { transpose(qkvwT[l], layers[l].qkvw, 3 * C, C); transpose(attprojwT[l], layers[l].attprojw, C, C); transpose(fcwT[l], layers[l].fcw, 4 * C, C); transpose(fcprojwT[l], layers[l].fcprojw, C, 4 * C); }
  transpose(wteT, wte, V, C);
}
static void mm_rows(float *out, const float *inp, const float *WT, const float *bias, int IC, int OC) {
  for (int t = 0; t < T; t++) { if (bias) matmul_t_row_ct(OC, out + t * OC, inp + t * IC, WT, bias, IC, OC); else matmul_t_row_nb_ct(OC, out + t * OC, inp + t * IC, WT, IC, OC); }
}
static void forward_hw(Acts *a) {
  const float scale = 1.0 / sqrtf(HS), gs = GELU_SCALING_FACTOR;
  KT(K_ENC, encoder_ct(T * C, a->encoded, tokens, wte, wpe, C));
  float *residual = a->encoded;
  for (int l = 0; l < L; l++) {
    Layer *p = &layers[l];
    KT(K_LN, ln_stats_ct(T, a->mean, a->rstd, residual, C)); KT(K_LN, ln_apply_ct(T * C, a->ln1, residual, a->mean, a->rstd, p->ln1w, p->ln1b, C));
    KT(K_MM, mm_rows(a->qkv, a->ln1, qkvwT[l], p->qkvb, C, 3 * C));
    KT(K_ATT, att_score_ct(NH * T * T, a->preatt, a->qkv, T, C, NH, scale));
    KT(K_ATT, att_softmax_ct(NH * T, a->att, a->preatt, T));
    KT(K_ATT, att_value_ct(T * C, a->atty, a->att, a->qkv, T, C, NH));
    KT(K_MM, mm_rows(a->attproj, a->atty, attprojwT[l], p->attprojb, C, C));
    KT(K_RES, residual_ct(T * C, a->residual2, residual, a->attproj));
    KT(K_LN, ln_stats_ct(T, a->mean, a->rstd, a->residual2, C)); KT(K_LN, ln_apply_ct(T * C, a->ln2, a->residual2, a->mean, a->rstd, p->ln2w, p->ln2b, C));
    KT(K_MM, mm_rows(a->fch, a->ln2, fcwT[l], p->fcb, C, 4 * C));
    KT(K_GELU, gelu_ct(T * 4 * C, a->fch_gelu, a->fch, gs));
    KT(K_MM, mm_rows(a->fcproj, a->fch_gelu, fcprojwT[l], p->fcprojb, 4 * C, C));
    KT(K_RES, residual_ct(T * C, a->residual3[l], a->residual2, a->fcproj));
    residual = a->residual3[l];
  }
  KT(K_LN, ln_stats_ct(T, a->mean, a->rstd, residual, C)); KT(K_LN, ln_apply_ct(T * C, a->lnf, residual, a->mean, a->rstd, lnfw, lnfb, C));
  KT(K_MM, mm_rows(a->logits, a->lnf, wteT, NULL, C, V));
  KT(K_SMAX, softmax_row_ct(T, a->probs, a->logits, V));
}
#endif

static int argmax(const float *p, int n) { int m = 0; for (int i = 1; i < n; i++) if (p[i] > p[m]) m = i; return m; }
static void report(const char *tag, Acts *a) {
  double cs = 0; for (int i = 0; i < T * V; i++) cs += a->logits[i];
  printf("%s argmax:", tag); for (int t = 0; t < T; t++) printf(" %d", argmax(a->logits + t * V, V));
  printf("\n%s logits sum %ld e-3, p[0][argmax] %ld e-6\n", tag, (long)(cs * 1e3), (long)(a->probs[argmax(a->logits, V)] * 1e6));
}
int main(void) {
  init_params();
  printf("gpt2 tiny: V %d C %d NH %d L %d T %d\n", V, C, NH, L, T);
  unsigned long c0 = cyc(); forward_ref(&ar); unsigned long c1 = cyc();
  REPORT("gpt2 scalar", c0, c1, T);
  report("scalar", &ar);
#ifndef X86
  make_transposed();
  c0 = cyc(); forward_hw(&ah); c1 = cyc();
  REPORT("gpt2 hwacha-cc", c0, c1, T);
  report("hwacha", &ah);
  unsigned long tot = 0; for (int i = 0; i < K_N; i++) tot += kcyc[i];
  for (int i = 0; i < K_N; i++) printf("  %s: %lu cycles (%lu%%)\n", kname[i], kcyc[i], tot ? kcyc[i] * 100 / tot : 0);
  float maxd = 0; int mism = 0;
  for (int t = 0; t < T; t++) {
    for (int i = 0; i < V; i++) { float d = ar.logits[t * V + i] - ah.logits[t * V + i]; if (d < 0) d = -d; if (d > maxd) maxd = d; }
    if (argmax(ar.logits + t * V, V) != argmax(ah.logits + t * V, V)) mism++;
  }
  float maxp = 0; for (int i = 0; i < T * V; i++) { float d = ar.probs[i] - ah.probs[i]; if (d < 0) d = -d; if (d > maxp) maxp = d; }
  printf("max |logit diff| %ld e-6, max |prob diff| %ld e-9, argmax mismatches %d / %d\n", (long)(maxd * 1e6), (long)(maxp * 1e9), mism, T);
  printf("gpt2 %s\n", (mism == 0 && maxd < 1e-2f) ? "PASS" : "FAIL");
  return !(mism == 0 && maxd < 1e-2f);
#else
  return 0;
#endif
}

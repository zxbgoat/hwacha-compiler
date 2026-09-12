// llama2.c (stories260K) greedy generation: verbatim scalar forward vs hwacha-cc kernels, same weights.
#include "../apps/common.h"
#include <math.h>
#include <string.h>
#ifndef STEPS
#define STEPS 40
#endif
#define DIM 64
#define HID 172
#define NL 5
#define NH 8
#define NKV 4
#define VOCAB 512
#define HS (DIM / NH)
#define KVD (DIM * NKV / NH)
#define KVMUL (NH / NKV)
#define SEQ STEPS   // kv-cache / att row stride: generation never goes past STEPS positions

asm(".section .rodata\n.balign 8\n.globl model_bin\nmodel_bin:\n.incbin \"stories260K.bin\"\n"
    ".balign 8\n.globl tok_bin\ntok_bin:\n.incbin \"tok512.bin\"\n.previous");
extern const unsigned char model_bin[], tok_bin[];

typedef struct { int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len; } Config;
typedef struct { const float *tok, *rms_att, *wq, *wk, *wv, *wo, *rms_ffn, *w1, *w2, *w3, *rms_final, *wcls; } Weights;
typedef struct {
  float x[DIM], xb[DIM], xb2[DIM], hb[HID], hb2[HID], q[DIM], att[NH * SEQ], logits[VOCAB];
  float key_cache[NL * SEQ * KVD], value_cache[NL * SEQ * KVD];
} State;
static State sr, sh;            // reference and hwacha states
static Config cfg; static Weights w;
static float fcr_tab[SEQ * HS / 2], fci_tab[SEQ * HS / 2];

static void map_weights(void) {
  memcpy(&cfg, model_bin, sizeof cfg);
  const float *p = (const float *)(model_bin + sizeof cfg);
  w.tok = p; p += VOCAB * DIM;
  w.rms_att = p; p += NL * DIM;
  w.wq = p; p += NL * DIM * DIM;
  w.wk = p; p += NL * DIM * KVD;
  w.wv = p; p += NL * DIM * KVD;
  w.wo = p; p += NL * DIM * DIM;
  w.rms_ffn = p; p += NL * DIM;
  w.w1 = p; p += NL * DIM * HID;
  w.w2 = p; p += NL * HID * DIM;
  w.w3 = p; p += NL * DIM * HID;
  w.rms_final = p; p += DIM;
  p += cfg.seq_len * HS / 2; p += cfg.seq_len * HS / 2;
  w.wcls = cfg.vocab_size > 0 ? w.tok : p;
}

// ---------------- scalar reference: verbatim llama2.c ----------------
static void rmsnorm(float *o, const float *x, const float *weight, int size) {
  float ss = 0.0f;
  for (int j = 0; j < size; j++) ss += x[j] * x[j];
  ss /= size; ss += 1e-5f; ss = 1.0f / sqrtf(ss);
  for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}
static void softmax(float *x, int size) {
  float max_val = x[0];
  for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) { x[i] = expf(x[i] - max_val); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}
static void matmul(float *xout, const float *x, const float *w, int n, int d) {
  for (int i = 0; i < d; i++) { float val = 0.0f; for (int j = 0; j < n; j++) val += w[i * n + j] * x[j]; xout[i] = val; }
}
static float *forward_ref(State *s, int token, int pos) {
  float *x = s->x; int dim = DIM, kv_dim = KVD, kv_mul = KVMUL, hidden_dim = HID, head_size = HS;
  memcpy(x, w.tok + token * dim, dim * sizeof(float));
  for (int l = 0; l < NL; l++) {
    rmsnorm(s->xb, x, w.rms_att + l * dim, dim);
    int loff = l * SEQ * kv_dim;
    float *k = s->key_cache + loff + pos * kv_dim, *v = s->value_cache + loff + pos * kv_dim;
    matmul(s->q, s->xb, w.wq + l * dim * dim, dim, dim);
    matmul(k, s->xb, w.wk + l * dim * kv_dim, dim, kv_dim);
    matmul(v, s->xb, w.wv + l * dim * kv_dim, dim, kv_dim);
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val = pos * freq, fcr = cosf(val), fci = sinf(val);
      int rotn = i < kv_dim ? 2 : 1;
      for (int vi = 0; vi < rotn; vi++) {
        float *vec = vi == 0 ? s->q : k;
        float v0 = vec[i], v1 = vec[i + 1];
        vec[i] = v0 * fcr - v1 * fci; vec[i + 1] = v0 * fci + v1 * fcr;
      }
    }
    for (int h = 0; h < NH; h++) {
      float *q = s->q + h * head_size, *att = s->att + h * SEQ;
      for (int t = 0; t <= pos; t++) {
        const float *kk = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += q[i] * kk[i];
        att[t] = score / sqrtf(head_size);
      }
      softmax(att, pos + 1);
      float *xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        const float *vv = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; i++) xb[i] += a * vv[i];
      }
    }
    matmul(s->xb2, s->xb, w.wo + l * dim * dim, dim, dim);
    for (int i = 0; i < dim; i++) x[i] += s->xb2[i];
    rmsnorm(s->xb, x, w.rms_ffn + l * dim, dim);
    matmul(s->hb, s->xb, w.w1 + l * dim * hidden_dim, dim, hidden_dim);
    matmul(s->hb2, s->xb, w.w3 + l * dim * hidden_dim, dim, hidden_dim);
    for (int i = 0; i < hidden_dim; i++) { float val = s->hb[i]; val *= (1.0f / (1.0f + expf(-val))); val *= s->hb2[i]; s->hb[i] = val; }
    matmul(s->xb, s->hb, w.w2 + l * dim * hidden_dim, hidden_dim, dim);
    for (int i = 0; i < dim; i++) x[i] += s->xb[i];
  }
  rmsnorm(x, x, w.rms_final, dim);
  matmul(s->logits, x, w.wcls, dim, VOCAB);
  return s->logits;
}

// ---------------- hwacha-cc kernels ----------------
void matmul_ct(long n, float *xout, const float *x, const float *w, int nn);
void matmul_t_ct(long n, float *xout, const float *x, const float *wt, int nn, int d);
// transposed copies of every weight matrix (a one-time layout change, like any deployment format)
static float wqT[NL * DIM * DIM], wkT[NL * DIM * KVD], wvT[NL * DIM * KVD], woT[NL * DIM * DIM], w1T[NL * DIM * HID], w2T[NL * HID * DIM], w3T[NL * DIM * HID], wclsT[DIM * VOCAB];
static int use_t;   // 0: row-major weights (strided lanes), 1: transposed (unit-stride lanes)
static void transpose(float *dst, const float *src, int n, int d) { for (int i = 0; i < d; i++) for (int j = 0; j < n; j++) dst[j * d + i] = src[i * n + j]; }
static void make_transposed(void) {
  for (int l = 0; l < NL; l++) {
    transpose(wqT + l * DIM * DIM, w.wq + l * DIM * DIM, DIM, DIM); transpose(wkT + l * DIM * KVD, w.wk + l * DIM * KVD, DIM, KVD);
    transpose(wvT + l * DIM * KVD, w.wv + l * DIM * KVD, DIM, KVD); transpose(woT + l * DIM * DIM, w.wo + l * DIM * DIM, DIM, DIM);
    transpose(w1T + l * DIM * HID, w.w1 + l * DIM * HID, DIM, HID); transpose(w3T + l * DIM * HID, w.w3 + l * DIM * HID, DIM, HID);
    transpose(w2T + l * HID * DIM, w.w2 + l * HID * DIM, HID, DIM);
  }
  transpose(wclsT, w.wcls, DIM, VOCAB);
}
#define MM(d, out, in, W, WT, n) (use_t ? matmul_t_ct(d, out, in, WT, n, d) : matmul_ct(d, out, in, W, n))
void rmsnorm_ct(long n, float *o, const float *x, const float *w, float ss);
void rope_ct(long n, float *q, float *k, const float *fcr, const float *fci, int pos, int head_size, int kv_dim);
void att_score_ct(long n, float *att, const float *q, const float *kc, int pos, int head_size, int kv_dim, int kv_mul, int seq, float sqrt_hs);
void att_softmax_ct(long n, float *att, int pos, int seq);
void att_value_ct(long n, float *xb, const float *att, const float *vc, int pos, int head_size, int kv_dim, int kv_mul, int seq);
void residual_ct(long n, float *x, const float *y);
void silu_mul_ct(long n, float *hb, const float *hb2);
enum { K_MATMUL, K_RMS, K_ROPE, K_SCORE, K_SOFTMAX, K_VALUE, K_RESID, K_SILU, K_HOST, K_N };
static const char *kname[K_N] = {"matmul", "rmsnorm", "rope", "att_score", "att_softmax", "att_value", "residual", "silu_mul", "host(reductions)"};
static unsigned long kcyc[K_N];
#define KT(slot, call) do { unsigned long _a = cyc(); call; kcyc[slot] += cyc() - _a; } while (0)
static float rms_scale(const float *x, int size) {
  float ss = 0.0f; for (int j = 0; j < size; j++) ss += x[j] * x[j];
  ss /= size; ss += 1e-5f; return 1.0f / sqrtf(ss);
}
static float *forward_hw(State *s, int token, int pos) {
  float *x = s->x; int dim = DIM, kv_dim = KVD, hidden_dim = HID;
  const float sqrt_hs = sqrtf(HS);
  memcpy(x, w.tok + token * dim, dim * sizeof(float));
  for (int l = 0; l < NL; l++) {
    float ss; KT(K_HOST, ss = rms_scale(x, dim));
    KT(K_RMS, rmsnorm_ct(dim, s->xb, x, w.rms_att + l * dim, ss));
    int loff = l * SEQ * kv_dim;
    float *k = s->key_cache + loff + pos * kv_dim, *v = s->value_cache + loff + pos * kv_dim;
    KT(K_MATMUL, MM(dim, s->q, s->xb, w.wq + l * dim * dim, wqT + l * dim * dim, dim));
    KT(K_MATMUL, MM(kv_dim, k, s->xb, w.wk + l * dim * kv_dim, wkT + l * dim * kv_dim, dim));
    KT(K_MATMUL, MM(kv_dim, v, s->xb, w.wv + l * dim * kv_dim, wvT + l * dim * kv_dim, dim));
    KT(K_ROPE, rope_ct(dim / 2, s->q, k, fcr_tab, fci_tab, pos, HS, kv_dim));
    KT(K_SCORE, att_score_ct(NH * (pos + 1), s->att, s->q, s->key_cache + loff, pos, HS, kv_dim, KVMUL, SEQ, sqrt_hs));
    KT(K_SOFTMAX, att_softmax_ct(NH, s->att, pos, SEQ));
    KT(K_VALUE, att_value_ct(dim, s->xb, s->att, s->value_cache + loff, pos, HS, kv_dim, KVMUL, SEQ));
    KT(K_MATMUL, MM(dim, s->xb2, s->xb, w.wo + l * dim * dim, woT + l * dim * dim, dim));
    KT(K_RESID, residual_ct(dim, x, s->xb2));
    KT(K_HOST, ss = rms_scale(x, dim));
    KT(K_RMS, rmsnorm_ct(dim, s->xb, x, w.rms_ffn + l * dim, ss));
    KT(K_MATMUL, MM(hidden_dim, s->hb, s->xb, w.w1 + l * dim * hidden_dim, w1T + l * dim * hidden_dim, dim));
    KT(K_MATMUL, MM(hidden_dim, s->hb2, s->xb, w.w3 + l * dim * hidden_dim, w3T + l * dim * hidden_dim, dim));
    KT(K_SILU, silu_mul_ct(hidden_dim, s->hb, s->hb2));
    KT(K_MATMUL, MM(dim, s->xb, s->hb, w.w2 + l * dim * hidden_dim, w2T + l * dim * hidden_dim, hidden_dim));
    KT(K_RESID, residual_ct(dim, x, s->xb));
  }
  float ss; KT(K_HOST, ss = rms_scale(x, dim));
  KT(K_RMS, rmsnorm_ct(dim, x, x, w.rms_final, ss));
  KT(K_MATMUL, MM(VOCAB, s->logits, x, w.wcls, wclsT, dim));
  return s->logits;
}

// ---------------- tokenizer (decode only) + driver ----------------
static const char *vocab[VOCAB]; static int vlen[VOCAB];
static void load_tokenizer(void) {
  const unsigned char *p = tok_bin + 4;   // skip max_token_length
  for (int i = 0; i < VOCAB; i++) { p += 4; int len; memcpy(&len, p, 4); p += 4; vocab[i] = (const char *)p; vlen[i] = len; p += len; }
}
static void decode_into(char *buf, int prev, int tok) {
  const char *s = vocab[tok]; int n = vlen[tok];
  if (prev == 1 && n > 0 && s[0] == ' ') { s++; n--; }
  if (n == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {   // raw byte token
    int b = 0; for (int i = 3; i < 5; i++) { char c = s[i]; b = b * 16 + (c >= 'A' ? c - 'A' + 10 : c - '0'); }
    buf[0] = (b >= 32 && b < 127) ? (char)b : '?'; buf[1] = 0; return;
  }
  memcpy(buf, s, n); buf[n] = 0;
}
static int argmax(const float *p, int n) { int m = 0; for (int i = 1; i < n; i++) if (p[i] > p[m]) m = i; return m; }

static int toks_ref[STEPS + 1], toks_hw[STEPS + 1];
static float logit_diff[STEPS];
int main(void) {
  map_weights(); load_tokenizer();
  printf("stories260K: dim %d hid %d layers %d heads %d kv %d vocab %d, %d steps\n", cfg.dim, cfg.hidden_dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.vocab_size, STEPS);
  for (int pos = 0; pos < SEQ; pos++)
    for (int hd = 0; hd < HS; hd += 2) {
      float freq = 1.0f / powf(10000.0f, hd / (float)HS), val = pos * freq;
      fcr_tab[pos * (HS / 2) + hd / 2] = cosf(val); fci_tab[pos * (HS / 2) + hd / 2] = sinf(val);
    }
  char buf[64];
  // scalar reference
  int tok = 1; toks_ref[0] = 1;
  unsigned long c0 = cyc();
  for (int pos = 0; pos < STEPS; pos++) { float *lg = forward_ref(&sr, tok, pos); int nx = argmax(lg, VOCAB); toks_ref[pos + 1] = nx; tok = nx; }
  unsigned long c1 = cyc();
  REPORT("llama scalar", c0, c1, STEPS);
  printf("scalar: ");
  for (int i = 0; i < STEPS; i++) { decode_into(buf, toks_ref[i], toks_ref[i + 1]); printf("%s", buf); }
  printf("\n");
  // hwacha: row-major weights, then transposed weights
  make_transposed();
  for (use_t = 0; use_t < 2; use_t++) {
    memset(&sh, 0, sizeof sh); memset(kcyc, 0, sizeof kcyc);
    tok = 1; toks_hw[0] = 1;
    c0 = cyc();
    for (int pos = 0; pos < STEPS; pos++) { float *lg = forward_hw(&sh, tok, pos); int nx = argmax(lg, VOCAB); toks_hw[pos + 1] = nx; tok = nx; }
    c1 = cyc();
    REPORT(use_t ? "llama hwacha-cc (transposed W)" : "llama hwacha-cc (row-major W)", c0, c1, STEPS);
    printf("hwacha: ");
    for (int i = 0; i < STEPS; i++) { decode_into(buf, toks_hw[i], toks_hw[i + 1]); printf("%s", buf); }
    printf("\n");
    unsigned long tot = 0; for (int i = 0; i < K_N; i++) tot += kcyc[i];
    for (int i = 0; i < K_N; i++) printf("  %s: %lu cycles (%lu%%)\n", kname[i], kcyc[i], tot ? kcyc[i] * 100 / tot : 0);
  }
  use_t = 1;
  int tokdiff = 0; for (int i = 1; i <= STEPS; i++) if (toks_ref[i] != toks_hw[i]) tokdiff++;
#ifdef NO_CHECK
  printf("generated token diffs %d / %d\nllama %s\n", tokdiff, STEPS, tokdiff ? "DIFF" : "PASS");
  return 0;
#endif
  // logits agreement on the reference token stream (re-run both forwards teacher-forced on toks_ref)
  memset(&sr, 0, sizeof sr); memset(&sh, 0, sizeof sh); memset(kcyc, 0, sizeof kcyc);
  float maxd = 0; int mism = 0;
  for (int pos = 0; pos < STEPS; pos++) {
    float *a = forward_ref(&sr, toks_ref[pos], pos), *b = forward_hw(&sh, toks_ref[pos], pos);
    float d = 0; for (int i = 0; i < VOCAB; i++) { float e = a[i] - b[i]; if (e < 0) e = -e; if (e > d) d = e; }
    logit_diff[pos] = d; if (d > maxd) maxd = d;
    if (argmax(a, VOCAB) != argmax(b, VOCAB)) mism++;
  }
  printf("max |logit diff| %ld e-6, argmax mismatches %d / %d (teacher-forced), generated token diffs %d / %d\n", (long)(maxd * 1e6), mism, STEPS, tokdiff, STEPS);
  printf("llama %s\n", (mism == 0 && tokdiff == 0) ? "PASS" : (maxd < 1e-2f ? "PASS (numerical)" : "FAIL"));
  return 0;
}

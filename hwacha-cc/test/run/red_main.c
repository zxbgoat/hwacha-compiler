#include <stdio.h>
#include "util.h"
static float hsqrt(float v) { float r = v > 1 ? v : 1; for (int k = 0; k < 30; k++) r = 0.5f * (r + v / r); return r; }
#define N 256
#define G 64
#define K 3
void rsum_ct(long n, const float *x, float *out);
void rmax_ct(long n, const float *x, float *out);
void risum_ct(long n, const int *x, int *out);
void rnorm_ct(long n, const float *x, float *y);
void rln_ct(long n, const float *x, float *y, int C);
void rrows_ct(long n, const float *a, float *out, int K_, int n_);
static float x[N], y[N], out[N], a[K * N], rout[K * N]; static int xi[N], oi[N];
static int feq(float p, float q) { float d = p - q; if (d < 0) d = -d; float m = q < 0 ? -q : q; return d <= 1e-4f * (m + 1); }
int main(void) {
  int fail = 0;
  for (int i = 0; i < N; i++) { x[i] = (float)((i * 37) % 101) / 10.0f - 5.0f; xi[i] = (i * 7919) % 2003 - 1000; }
  for (int i = 0; i < K * N; i++) a[i] = (float)((i * 13) % 17) - 8.0f;
  // rsum
  rsum_ct(N, x, out); { int bad = 0; for (int g = 0; g < N / G; g++) { float s = 0; for (int j = 0; j < G; j++) s += x[g * G + j]; if (!feq(out[g], s)) bad++; } printf("%-8s %s (%d mismatches)\n", "rsum", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  rmax_ct(N, x, out); { int bad = 0; for (int g = 0; g < N / G; g++) { float m = -1e30f; for (int j = 0; j < G; j++) if (x[g * G + j] > m) m = x[g * G + j]; if (!feq(out[g], m)) bad++; } printf("%-8s %s (%d mismatches)\n", "rmax", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  risum_ct(N, xi, oi); { int bad = 0; for (int g = 0; g < N / G; g++) { int s = 0; for (int j = 0; j < G; j++) s += xi[g * G + j]; if (oi[g] != s) bad++; } printf("%-8s %s (%d mismatches)\n", "risum", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  rnorm_ct(N, x, y); { int bad = 0; for (int g = 0; g < N / G; g++) { float s = 0; for (int j = 0; j < G; j++) if (x[g * G + j] > 0) s += x[g * G + j]; for (int j = 0; j < G; j++) if (!feq(y[g * G + j], x[g * G + j] / s)) bad++; } printf("%-8s %s (%d mismatches)\n", "rnorm", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  rln_ct(N, x, y, G); { int bad = 0; for (int g = 0; g < N / G; g++) { float m = 0; for (int j = 0; j < G; j++) m += x[g * G + j]; m /= G; float v = 0; for (int j = 0; j < G; j++) { float d = x[g * G + j] - m; v += d * d; } v /= G; float r = 1.0f / hsqrt(v + 1e-5f); for (int j = 0; j < G; j++) if (!feq(y[g * G + j], (x[g * G + j] - m) * r)) bad++; } printf("%-8s %s (%d mismatches)\n", "rln", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  rrows_ct(N, a, rout, K, N); { int bad = 0; for (int g = 0; g < N / G; g++) for (int k = 0; k < K; k++) { float s = 0; for (int j = 0; j < G; j++) s += a[k * N + g * G + j]; if (!feq(rout[g * K + k], s)) bad++; } printf("%-8s %s (%d mismatches)\n", "rrows", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  printf("%s\n", fail ? "SOME KERNELS FAILED" : "ALL KERNELS PASSED"); return fail != 0;
}

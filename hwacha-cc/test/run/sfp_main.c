#include <stdio.h>
#include "util.h"
#define N 200
void sfp_mul_ct(long n, const float *x, float *y, float a, float b);
void sfp_cvt_ct(long n, const int *x, float *y, int n_);
void sfp_dbl_ct(long n, const double *x, double *y, double a);
static float x[N], y[N]; static int xi[N]; static double xd[N], yd[N];
static int feq(float p, float q) { float d = p - q; if (d < 0) d = -d; return d <= 1e-4f * (q < 0 ? -q : q) + 1e-5f; }
int main(void) {
  int fail = 0;
  for (int i = 0; i < N; i++) { x[i] = i * 0.25f - 10.0f; xi[i] = i * 3 - 100; xd[i] = i * 0.125 - 5.0; }
  sfp_mul_ct(N, x, y, 2.0f, 3.0f); { int bad = 0; for (int i = 0; i < N; i++) if (!feq(y[i], x[i] * 7.5f)) bad++; printf("%-8s %s (%d mismatches)\n", "sfp_mul", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  sfp_cvt_ct(N, xi, y, 7); { int bad = 0; for (int i = 0; i < N; i++) if (!feq(y[i], (float)xi[i] + 3.5f)) bad++; printf("%-8s %s (%d mismatches)\n", "sfp_cvt", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  sfp_dbl_ct(N, xd, yd, 3.0); { int bad = 0; for (int i = 0; i < N; i++) { double d = yd[i] - (xd[i] + 8.0); if (d < 0) d = -d; if (d > 1e-9) bad++; } printf("%-8s %s (%d mismatches)\n", "sfp_dbl", bad ? "FAIL" : "PASS", bad); fail |= bad; }
  printf("%s\n", fail ? "SOME KERNELS FAILED" : "ALL KERNELS PASSED"); return fail != 0;
}

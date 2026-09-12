#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util.h"

#define N 5000   /* several stripmine iterations plus a fringe */

void saxpy_ct(long n, float a, const float *x, float *y);
void clamp_scale_ct(long n, float a, const float *x, float *y);
void iota_ct(long n, long base, long *c);

static float xf[N], yf[N], yref[N];
static long  cl[N], cref[N];

static unsigned rnd_state = 12345;
static float frand(void) {            /* deterministic, no libc rand */
  rnd_state = rnd_state * 1103515245u + 12345u;
  return ((int)(rnd_state >> 8) % 2000 - 1000) / 100.0f;
}

static int check_f(const char *name, const float *got, const float *ref) {
  int bad = 0;
  for (int i = 0; i < N; i++) {
    float d = got[i] - ref[i]; if (d < 0) d = -d;
    float m = ref[i] < 0 ? -ref[i] : ref[i];
    if (d > 1e-5f * (m + 1.0f)) { if (bad < 3) printf("  %s[%d]: got %d ref %d (x1000)\n", name, i, (int)(got[i]*1000), (int)(ref[i]*1000)); bad++; }
  }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad ? "FAIL" : "PASS", bad, N);
  return bad != 0;
}
static int check_l(const char *name, const long *got, const long *ref) {
  int bad = 0;
  for (int i = 0; i < N; i++) if (got[i] != ref[i]) { if (bad < 3) printf("  %s[%d]: got %ld ref %ld\n", name, i, got[i], ref[i]); bad++; }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad ? "FAIL" : "PASS", bad, N);
  return bad != 0;
}

int main(void) {
  int fail = 0;
  const float a = 3.25f;

  /* K1 saxpy */
  for (int i = 0; i < N; i++) { xf[i] = frand(); yf[i] = frand(); yref[i] = a * xf[i] + yf[i]; }
  saxpy_ct(N, a, xf, yf);
  fail |= check_f("saxpy", yf, yref);

  /* K2 clamp_scale */
  for (int i = 0; i < N; i++) { xf[i] = frand(); yf[i] = -12345.0f; yref[i] = xf[i] > 0.0f ? a * xf[i] : 0.0f; }
  clamp_scale_ct(N, a, xf, yf);
  fail |= check_f("clamp_scale", yf, yref);

  /* K3 iota */
  for (int i = 0; i < N; i++) { cl[i] = -1; cref[i] = 1000000 + 2 * (long)i; }
  iota_ct(N, 1000000, cl);
  fail |= check_l("iota", cl, cref);

  printf("%s\n", fail ? "SOME KERNELS FAILED" : "ALL KERNELS PASSED");
  return fail;
}

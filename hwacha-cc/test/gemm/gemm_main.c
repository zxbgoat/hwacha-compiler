// n x n sgemm: scalar reference vs hwacha-cc (row launches, and one flat launch); same n as vec-sgemm-opt
#include "../apps/common.h"
#ifndef N
#define N 256
#endif
void gemm_row_ct(long n, float *C, const float *A, const float *B, int nn, int i);
void gemm_flat_ct(long n, float *C, const float *A, const float *B, int nn);
static float A[N * N], B[N * N], C[N * N], R[N * N];
static int check(const char *tag) {
  int bad = 0;
  for (int i = 0; i < N * N; i++) { float d = C[i] - R[i]; if (d < 0) d = -d; if (d > 1e-3f * (R[i] < 0 ? -R[i] : R[i]) + 1e-3f) bad++; }
  printf("%s %s (%d mismatches / %d)\n", tag, bad ? "FAIL" : "PASS", bad, N * N); return bad;
}
int main(void) {
  for (int i = 0; i < N * N; i++) { A[i] = frand(-1, 1); B[i] = frand(-1, 1); }
  unsigned long c0, c1;
#ifndef NO_SCALAR   // the 256^3 scalar reference takes ~15 h on RTL: there the row variant is the reference
  c0 = cyc();
  for (int i = 0; i < N; i++) for (int k = 0; k < N; k++) { float acc = 0; for (int j = 0; j < N; j++) acc += A[i * N + j] * B[j * N + k]; R[i * N + k] = acc; }
  c1 = cyc();
  REPORT("gemm scalar", c0, c1, (unsigned long)N * N);
#endif
  c0 = cyc(); for (int i = 0; i < N; i++) gemm_row_ct(N, C, A, B, N, i); c1 = cyc();
  REPORT("gemm_row hwacha-cc", c0, c1, (unsigned long)N * N);
#ifdef NO_SCALAR
  for (int i = 0; i < N * N; i++) R[i] = C[i];
#endif
  int bad = check("gemm_row");
#ifndef ROW_ONLY
  for (int i = 0; i < N * N; i++) C[i] = 0;
  c0 = cyc(); gemm_flat_ct((long)N * N, C, A, B, N); c1 = cyc();
  REPORT("gemm_flat hwacha-cc", c0, c1, (unsigned long)N * N);
  bad += check("gemm_flat");
#endif
  return bad != 0;
}

// Host for the MLIR (gpu dialect) entry tests. Every kernel AND its launch come from a .mlir file:
// hwacha-mlir emits the kernels for hwacha-cc and the host functions (gpu.launch_func lowered to
// <kernel>_ct calls); this file only prepares data and checks results.
#include <stdio.h>
#include <stdlib.h>
// the esp toolchain libm does not link with medany (R_RISCV_HI20 truncation): host-side versions
static float h_fabsf(float v) { return v < 0 ? -v : v; }
static float h_sqrtf(float v) { if (v <= 0) return 0; float r = v; for (int i = 0; i < 40; i++) r = 0.5f * (r + v / r); return r; }
static double h_exp(double x) { int k = (int)(x / 0.6931471805599453 + (x < 0 ? -0.5 : 0.5)); double r = x - k * 0.6931471805599453, s = 1, t = 1;
  for (int i = 1; i < 16; i++) { t *= r / i; s += t; } while (k > 0) { s *= 2; k--; } while (k < 0) { s *= 0.5; k++; } return s; }

// bare-pointer memref convention: memref<4096xf32> -> float*, index -> long
void run_saxpy(float a, float *x, float *y, long n);                 // saxpy_gpu.mlir
void run_gs(float *x, float *y, long n);                             // gs.mlir (grid-stride loop)
void run_wg(float *p);                                               // wg.mlir
void run_ex(float *p);                                               // ex.mlir
void matmul(float *a, float *b, float *c);                           // mm.mlir (linalg.matmul)
void matmul_blk(float *a, float *b, float *c);                       // mmt.mlir (tiled 4 rows, register blocked)
static long rdcycle(void) { long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
// descriptor convention (dynamic shapes): {alloc, aligned, offset, size, stride} per memref
void saxpy(float a, float *xa, float *xp, long xo, long xs, long xst, float *ya, float *yp, long yo, long ys, long yst);  // saxpy_linalg.mlir

#define N 4096
static float x[N], y[N], ref[N], p[N];
static float A[64*64], B[64*64], C[64*64], Cref[64*64];

static int check(const char *name, const float *got, const float *want, int n, float tol) {
  int bad = 0;
  for (int i = 0; i < n; i++) if (h_fabsf(got[i] - want[i]) > tol * (1 + h_fabsf(want[i]))) { if (bad < 3) printf("  %s[%d] = %g want %g\n", name, i, got[i], want[i]); bad++; }
  printf("%-14s %s\n", name, bad ? "FAIL" : "ok");
  return bad != 0;
}

int main(void) {
  int fail = 0;
  for (int i = 0; i < N; i++) { x[i] = (i % 17) * 0.25f - 2; y[i] = (i % 5) * 0.5f; }

  // 1. gpu.func saxpy, 16 blocks x 256, bounds check against n=4000
  for (int i = 0; i < N; i++) ref[i] = i < 4000 ? 1.5f * x[i] + y[i] : y[i];
  run_saxpy(1.5f, x, y, 4000);
  fail |= check("saxpy(gpu)", y, ref, N, 1e-6f);

  // 2. linalg.generic saxpy outlined with block size 1 (block id = work-item), descriptor convention
  for (int i = 0; i < N; i++) y[i] = (i % 5) * 0.5f;
  for (int i = 0; i < N; i++) ref[i] = 2.0f * x[i] + y[i];
  saxpy(2.0f, x, x, 0, N, 1, y, y, 0, N, 1);
  fail |= check("saxpy(linalg)", y, ref, N, 1e-6f);

  // 3. grid-stride loop: 4 blocks x 64 work-items over 4096 elements (uses gpu.grid_dim)
  for (int i = 0; i < N; i++) { y[i] = -1; ref[i] = i < 4000 ? 2 * x[i] : -1; }
  run_gs(x, y, 4000);
  fail |= check("grid-stride", y, ref, N, 0);

  // 4. workgroup memory + barrier
  for (int i = 0; i < N; i++) { p[i] = i * 0.5f; ref[i] = (i / 256 * 256) * 0.5f; }
  run_wg(p);
  fail |= check("workgroup", p, ref, N, 0);

  // 5. math incl. exp
  for (int i = 0; i < N; i++) { p[i] = x[i]; float a = h_fabsf(x[i]); ref[i] = (float)h_exp((h_sqrtf(a) * x[i] + a) * 0.1f); }
  run_ex(p);
  fail |= check("math+exp", p, ref, N, 2e-6f);

  // 6. linalg.matmul 64x64 (2-D grid collapsed to 1-D by hwacha-mlir, k loop on the control thread)
  for (int i = 0; i < 64*64; i++) { A[i] = (i % 7) - 3; B[i] = (i % 11) * 0.5f - 2; C[i] = 0; Cref[i] = 0; }
  for (int i = 0; i < 64; i++) for (int k = 0; k < 64; k++) for (int j = 0; j < 64; j++) Cref[i*64+j] += A[i*64+k] * B[k*64+j];
  long t0 = rdcycle(); matmul(A, B, C); long t1 = rdcycle();
  fail |= check("matmul(linalg)", C, Cref, 64*64, 1e-4f);

  // 7. the same matmul with the embedded transform script (4-row tile, lanes over columns, k on the control thread)
  for (int i = 0; i < 64*64; i++) C[i] = 0;
  long t2 = rdcycle(); matmul_blk(A, B, C); long t3 = rdcycle();
  fail |= check("matmul(tiled)", C, Cref, 64*64, 1e-4f);
  printf("matmul 64^3 cycles: linalg %ld, tiled %ld\n", t1 - t0, t3 - t2);

  printf(fail ? "mlir FAIL\n" : "ALL KERNELS PASSED\n");
  return fail;
}

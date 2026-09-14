// Host for the MLIR (gpu dialect) entry tests: every kernel comes from a .mlir file via
// tools/mlir-to-ll.sh + hwacha-cc, and runs as <kernel>_ct(n, args...) with n work-items.
#include <stdio.h>
#include <stdlib.h>
// the esp toolchain libm sqrtf does not link with medany (R_RISCV_HI20 truncation): host-side versions
static float h_fabsf(float v) { return v < 0 ? -v : v; }
static float h_sqrtf(float v) { if (v <= 0) return 0; float r = v; for (int i = 0; i < 40; i++) r = 0.5f * (r + v / r); return r; }
extern long hwacha_group_size;

// saxpy_gpu.mlir: gpu.func @saxpy(f32 a, memref<4096xf32> x, memref<4096xf32> y, i64 n)  [bare pointers]
void saxpy_ct(long n, float a, float *x, float *y, long nn);
// saxpy_linalg.mlir: linalg.generic outlined by gpu-kernel-outlining (block size 1), memref descriptors
//   args: step, lb, x{alloc,aligned,offset,size,stride}, y{...}, a
void saxpy_kernel_ct(long n, long step, long lb, float *xa, float *xp, long xo, long xs, long xst,
                     float *ya, float *yp, long yo, long ys, long yst, float a);
// wg.mlir: workgroup memref + gpu.barrier; p[g] = p[first element of the block]
void wg_ct(long n, float *p);
// ex.mlir: math.absf / math.sqrt / math.fma
void ex_ct(long n, float *p);
// mm.mlir: linalg.matmul 64x64, 2-D parallel loops collapsed to 1-D (blocks only); inner k loop stays
//   args: step, lb, 64 (collapse factor), a, b, c, k_lb, k_ub, k_step
void matmul_kernel_ct(long n, long step, long lb, long ncols, float *a, float *b, float *c, long klb, long kub, long kstep);

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

  // 1. gpu.func saxpy, CUDA-style index, bounds check against n (run n=4000 to exercise the guard)
  for (int i = 0; i < N; i++) ref[i] = i < 4000 ? 1.5f * x[i] + y[i] : y[i];
  hwacha_group_size = 256;
  saxpy_ct(N, 1.5f, x, y, 4000);
  fail |= check("saxpy(gpu)", y, ref, N, 1e-6f);

  // 2. linalg.generic saxpy outlined with block size 1: block id = work-item, descriptor convention
  for (int i = 0; i < N; i++) y[i] = (i % 5) * 0.5f;
  for (int i = 0; i < N; i++) ref[i] = 2.0f * x[i] + y[i];
  hwacha_group_size = 0;
  saxpy_kernel_ct(N, 1, 0, x, x, 0, N, 1, y, y, 0, N, 1, 2.0f);
  fail |= check("saxpy(linalg)", y, ref, N, 1e-6f);

  // 3. workgroup memory + barrier
  for (int i = 0; i < N; i++) { p[i] = i * 0.5f; ref[i] = (i / 256 * 256) * 0.5f; }
  hwacha_group_size = 256;
  wg_ct(N, p);
  fail |= check("workgroup", p, ref, N, 0);

  // 4. math: fma(sqrt(|v|), v, |v|)
  for (int i = 0; i < N; i++) { p[i] = x[i]; float a = h_fabsf(x[i]); ref[i] = h_sqrtf(a) * x[i] + a; }
  hwacha_group_size = 0;
  ex_ct(N, p);
  fail |= check("math", p, ref, N, 1e-5f);

  // 5. linalg.matmul 64x64 (collapsed 2-D grid, block size 1, uniform k loop on the control thread)
  for (int i = 0; i < 64*64; i++) { A[i] = (i % 7) - 3; B[i] = (i % 11) * 0.5f - 2; C[i] = 0; Cref[i] = 0; }
  for (int i = 0; i < 64; i++) for (int k = 0; k < 64; k++) for (int j = 0; j < 64; j++) Cref[i*64+j] += A[i*64+k] * B[k*64+j];
  matmul_kernel_ct(64*64, 1, 0, 64, A, B, C, 0, 64, 1);
  fail |= check("matmul(linalg)", C, Cref, 64*64, 1e-4f);

  printf(fail ? "mlir FAIL\n" : "ALL KERNELS PASSED\n");
  return fail;
}

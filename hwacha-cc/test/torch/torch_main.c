// Host for a PyTorch model compiled through torch-mlir -> hwacha-mlir -> hwacha-cc.
// The MLIR host function keeps the memref.alloc's of the bufferized graph: they become malloc calls,
// served here by a bump allocator reset per forward. Checks one forward against the PyTorch output
// exported next to the model (<model>_check.bin).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#ifdef X86
static long rdcycle(void) { return 0; }
#else
asm(".section .rodata\n.balign 8\n.globl check_bin\ncheck_bin:\n.incbin \"" CHECK_FILE "\"\n.previous");
extern const unsigned char check_bin[];
static long rdcycle(void) { long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
float __math_oflowf(unsigned sign) { return sign ? -1.0f / 0.0f : 1.0f / 0.0f; }
float __math_uflowf(unsigned sign) { return sign ? -0.0f : 0.0f; }
int *__errno(void) { static int e; return &e; }
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  uintptr_t tval; asm volatile("csrr %0, mtval" : "=r"(tval));
  printf("TRAP cause %lx epc %lx tval %lx ra %lx sp %lx\n", cause, epc, tval, regs[1], regs[2]);
  exit(1);
}
// the bufferized graph allocates its intermediates with malloc: a bump allocator
#define ARENA (1 << 24)
static char arena[ARENA] __attribute__((aligned(64))); static long atop;
void *malloc(size_t n) { long p = (atop + 63) & ~63L; atop = p + n; if (atop > ARENA) { printf("arena overflow\n"); exit(1); } return arena + p; }
void free(void *p) { (void)p; }
void *aligned_alloc(size_t a, size_t n) { (void)a; return malloc(n); }
#endif
static const char *fmtf(float v) {
  static char buf[8][32]; static int k;
  char *b = buf[k++ & 7], *q = b;
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

// the MLIR host function (bare-pointer memref convention): memref<1x1x28x28xf32> -> float*
float *unet(float *x, float *t, float *c);
static float x[784], tvec[1], cvec[10];

int main(int argc, char **argv) {
#ifdef X86
  (void)argc; (void)argv; printf("x86 build only checks that the host code links\n"); return 0;
#else
  const unsigned char *p = check_bin;
  int cls = *(const int *)p; p += 4;
  float t; memcpy(&t, p, 4); p += 4;
  memcpy(x, p, sizeof x); p += sizeof x;
  const float *ref = (const float *)p;
  tvec[0] = t; memset(cvec, 0, sizeof cvec); cvec[cls] = 1.0f;
  printf("torch-mlir unet: t=%s class %d\n", fmtf(t), cls);
  atop = 0;
  long c0 = rdcycle();
  float *y = unet(x, tvec, cvec);
  long c1 = rdcycle();
  float md = 0, mx = 0;
  for (int k = 0; k < 784; k++) { float d = fabsf(y[k] - ref[k]); if (d > md) md = d; if (fabsf(ref[k]) > mx) mx = fabsf(ref[k]); }
  int ok = md <= 0.02f * mx;
  printf("forward vs PyTorch: max |diff| = %s, max |ref| = %s, %s; %ld cycles, arena %ld KB\n", fmtf(md), fmtf(mx), ok ? "ok" : "FAIL", c1 - c0, atop / 1024);
  printf(ok ? "torch PASS\n" : "torch FAIL\n");
  return !ok;
#endif
}

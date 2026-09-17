// Host for an ONNX model compiled through torch-mlir's ONNX importer -> hwacha-mlir -> hwacha-cc.
// Inputs, the entry function's arity and an onnxruntime reference output come from <model>_check.bin
// (written by export_onnx.py); the entry is the ONNX graph name (`main_graph` for torch.onnx.export).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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
#define ARENA (1 << 24)
static char arena[ARENA] __attribute__((aligned(64))); static long atop;
void *malloc(size_t n) { long p = (atop + 63) & ~63L; atop = p + n; if (atop > ARENA) { printf("arena overflow\n"); exit(1); } return arena + p; }
void free(void *p) { (void)p; }
void *aligned_alloc(size_t a, size_t n) { (void)a; return malloc(n); }
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
#ifndef ENTRY
#define ENTRY main_graph
#endif
#ifndef NARGS
#define NARGS 1
#endif
#if NARGS == 1
float *ENTRY(float *);
#elif NARGS == 2
float *ENTRY(float *, float *);
#elif NARGS == 3
float *ENTRY(float *, float *, float *);
#else
float *ENTRY(float *, float *, float *, float *);
#endif
#define STR2(x) #x
#define STR(x) STR2(x)

int main(void) {
  const unsigned char *p = check_bin;
  int nin = *(const int *)p; p += 4;
  if (nin != NARGS) { printf("check file has %d inputs, host built for %d\n", nin, NARGS); return 1; }
  float *in[4] = {0}; int size[4];
  for (int i = 0; i < nin; i++) { size[i] = *(const int *)p; p += 4; in[i] = malloc(size[i] * sizeof(float)); memcpy(in[i], p, size[i] * sizeof(float)); p += size[i] * sizeof(float); }
  int nout = *(const int *)p; p += 4;
  const float *ref = (const float *)p;
  long mark = atop;
  printf("onnx %s: %d inputs, %d outputs\n", STR(ENTRY), nin, nout);
  long c0 = rdcycle();
#if NARGS == 1
  float *y = ENTRY(in[0]);
#elif NARGS == 2
  float *y = ENTRY(in[0], in[1]);
#elif NARGS == 3
  float *y = ENTRY(in[0], in[1], in[2]);
#else
  float *y = ENTRY(in[0], in[1], in[2], in[3]);
#endif
  long c1 = rdcycle();
  float md = 0, mx = 0;
  for (int k = 0; k < nout; k++) { float d = fabsf(y[k] - ref[k]); if (d > md) md = d; if (fabsf(ref[k]) > mx) mx = fabsf(ref[k]); }
  int ok = md <= 0.02f * mx;
  printf("forward vs onnxruntime: max |diff| = %s, max |ref| = %s, %s; %ld cycles, arena %ld KB\n", fmtf(md), fmtf(mx), ok ? "ok" : "FAIL", c1 - c0, (atop - mark) / 1024);
  if (nout <= 16) { printf("output:"); for (int k = 0; k < nout; k++) printf(" %s", fmtf(y[k])); printf("\n"); }
  printf(ok ? "onnx PASS\n" : "onnx FAIL\n");
  return !ok;
}

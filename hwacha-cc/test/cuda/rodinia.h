// Shared by the Rodinia CUDA hosts: a deterministic RNG, cycle counter, bare-metal errno / trap report.
#pragma once
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
static unsigned rng_state = 20240911u;
static inline unsigned rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }
static inline float frand(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xffff) / 65535.0f; }
static inline unsigned long cyc(void) { unsigned long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
#define REPORT(tag, c0, c1, n) printf("%s: %lu cycles, %lu.%02lu cyc/elem\n", tag, (c1)-(c0), ((c1)-(c0))/(n), (((c1)-(c0))*100/(n))%100)
extern "C" {
static int hw_errno; int *__errno(void) { return &hw_errno; }
static int in_trap;
uintptr_t handle_trap(uintptr_t cause, uintptr_t epc, uintptr_t regs[32]) {
  uintptr_t tval; asm volatile("csrr %0, mtval" : "=r"(tval));
  if (!in_trap++) printf("TRAP cause %lx epc %lx tval %lx\n", cause, epc, tval);
  exit(1);
}
}

#include <stdio.h>
#include "util.h"
static unsigned rng_state = 20240911u;
static inline unsigned rnd(void) { rng_state = rng_state * 1103515245u + 12345u; return rng_state >> 8; }
static inline float frand(float lo, float hi) { return lo + (hi - lo) * (float)(rnd() & 0xffff) / 65535.0f; }
static inline unsigned long cyc(void) { unsigned long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
#define REPORT(tag, c0, c1, n) printf("%s: %lu cycles, %lu.%02lu cyc/elem\n", tag, (c1)-(c0), ((c1)-(c0))/(n), (((c1)-(c0))*100/(n))%100)
extern long hwacha_group_size, hwacha_vl_short;
// newlib libm wrappers (exp, pow, ...) set errno through __errno; the bare-metal env has none
static int hw_errno; int *__errno(void) { return &hw_errno; }

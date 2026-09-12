// probe: vectorized expf accuracy against newlib expf on Spike / RTL
#include "../apps/common.h"
#include <math.h>
#define N 1024
void expk_ct(long n, const float *in, float *out);
static float in[N], out[N];
int main(void) {
  for (int i = 0; i < N; i++) in[i] = -90.0f + 180.0f * i / (N - 1);
  in[0] = 0; in[1] = 1; in[2] = -1; in[3] = 0.5f; in[4] = -0.5f; in[5] = 88.5f; in[6] = -87.5f; in[7] = 1e-6f; in[8] = -1e-6f; in[9] = 10.0f;
  unsigned long c0 = cyc(); expk_ct(N, in, out); unsigned long c1 = cyc();
  REPORT("expk hwacha-cc", c0, c1, N);
  double maxrel = 0; int bad = 0, worst = 0;
  for (int i = 0; i < N; i++) {
    double r = exp((double)in[i]); double e = fabs((double)out[i] - r) / (r > 1e-37 ? r : 1e-37);
    if (in[i] < -87.0f || in[i] > 88.0f) continue;   // clamped region
    if (e > maxrel) { maxrel = e; worst = i; }
    if (e > 4e-7) { bad++; if (bad < 6) printf("  bad x=%ld/1000 rel=%ld e-9\n", (long)(in[i]*1000), (long)(e*1e9)); }
  }
  printf("expk max rel err %ld e-9 at x=%ld/1000 (%d > 4e-7 of %d)\n", (long)(maxrel * 1e9), (long)(in[worst] * 1000), bad, N);
  printf("expk %s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}

// Rodinia streamcluster pgain_kernel: cost of opening center x for every point
#include "common.h"
#define NUM 1024
#define DIM 8
#define K 4
#define GROUP 256
typedef struct { float weight; long assign; float cost; } Point_Struct;
void pgain_kernel_ct(long n, Point_Struct *p, float *coord, float *work_mem, int *center_table, char *switch_membership, float *coord_s, int num, int dim, long x, int K_);
static Point_Struct p[NUM]; static float coord[NUM*DIM], work[NUM*(K+1)], wref[NUM*(K+1)], coord_s[GROUP > DIM ? GROUP : DIM]; static int center_table[NUM]; static char sw[NUM], swref[NUM];
int main(void) {
  const long x = 7;
  for (int i = 0; i < NUM; i++) { p[i].weight = frand(0.5f, 2.0f); p[i].assign = rnd() % NUM; p[i].cost = frand(0, 5000); center_table[i] = rnd() % K; }
  for (int i = 0; i < NUM*DIM; i++) coord[i] = frand(0, 10);
  for (int i = 0; i < NUM*(K+1); i++) { work[i] = 0; wref[i] = 0; }
  unsigned long c0 = cyc();
  for (int t = 0; t < NUM; t++) { float xc = 0; for (int i = 0; i < DIM; i++) { float d = coord[i*NUM+t] - coord[i*NUM+x]; xc += d*d; } xc *= p[t].weight;
    float cur = p[t].cost; int base = t*(K+1);
    if (xc < cur) { swref[t] = '1'; wref[base+K] = xc - cur; } else { swref[t] = 0; wref[base + center_table[p[t].assign]] += cur - xc; } }
  unsigned long c1 = cyc(); REPORT("pgain scalar", c0, c1, NUM);
  hwacha_group_size = GROUP;
  for (int i = 0; i < NUM; i++) sw[i] = 0;
  c0 = cyc(); pgain_kernel_ct(NUM, p, coord, work, center_table, sw, coord_s, NUM, DIM, x, K); c1 = cyc();
  REPORT("pgain hwacha-cc", c0, c1, NUM);
  int bad = 0; for (int i = 0; i < NUM*(K+1); i++) { float d = work[i] - wref[i]; if (d < 0) d = -d; if (d > 1e-3f * (wref[i] < 0 ? -wref[i] : wref[i]) + 1e-2f) bad++; }
  for (int i = 0; i < NUM; i++) if (sw[i] != swref[i]) bad++;
  printf("pgain %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad); return bad != 0;
}

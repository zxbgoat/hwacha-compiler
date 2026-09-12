// Rodinia kmeans: one membership-assignment step (kmeans_kernel_c) plus the feature transpose (kmeans_swap)
#include "common.h"
#define NP 1024
#define NF 8
#define NC 5
void kmeans_kernel_c_ct(long n, float *feature, float *clusters, int *membership, int npoints, int nclusters, int nfeatures, int offset, int size);
void kmeans_swap_ct(long n, float *feature, float *feature_swap, int npoints, int nfeatures);
static float feature[NP*NF], fswap[NP*NF], fref[NP*NF], clusters[NC*NF]; static int membership[NP], mref[NP];
int main(void) {
  for (int i = 0; i < NP*NF; i++) feature[i] = frand(0, 100);
  for (int i = 0; i < NC*NF; i++) clusters[i] = frand(0, 100);
  /* reference: transpose then assign */
  unsigned long c0 = cyc();
  for (int p = 0; p < NP; p++) for (int f = 0; f < NF; f++) fref[f*NP + p] = feature[p*NF + f];
  for (int p = 0; p < NP; p++) { float best = 3.40282347e+38f; int idx = 0;
    for (int c = 0; c < NC; c++) { float d = 0; for (int f = 0; f < NF; f++) { float t = fref[f*NP+p] - clusters[c*NF+f]; d += t*t; } if (d < best) { best = d; idx = c; } }
    mref[p] = idx; }
  unsigned long c1 = cyc(); REPORT("kmeans scalar", c0, c1, NP);
  c0 = cyc(); kmeans_swap_ct(NP, feature, fswap, NP, NF); c1 = cyc(); REPORT("kmeans_swap hwacha-cc", c0, c1, NP);
  int bad = 0; for (int i = 0; i < NP*NF; i++) if (fswap[i] != fref[i]) bad++;
  printf("kmeans_swap %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);
  for (int i = 0; i < NP; i++) membership[i] = -1;
  c0 = cyc(); kmeans_kernel_c_ct(NP, fswap, clusters, membership, NP, NC, NF, 0, 0); c1 = cyc(); REPORT("kmeans_c hwacha-cc", c0, c1, NP);
  int bad2 = 0; for (int i = 0; i < NP; i++) if (membership[i] != mref[i]) bad2++;
  printf("kmeans_c %s (%d mismatches / %d)\n", bad2 ? "FAIL" : "PASS", bad2, NP);
  return (bad || bad2) != 0;
}

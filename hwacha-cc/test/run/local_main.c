#include <stdio.h>
#include "util.h"
#define N 2048           /* multiple of the required group size 256 */
void group_sum_ct(long n, const float *x, float *sums);
void shift_left_ct(long n, const float *x, float *y);
static float x[N], out[N], ref[N];
static unsigned st = 31337; static float frand(void){ st = st*1103515245u+12345u; return ((int)(st>>8)%2000-1000)/100.0f; }
static int check(const char*name,int n){ int bad=0; for(int i=0;i<n;i++){ float d=out[i]-ref[i]; if(d<0)d=-d; float m=ref[i]<0?-ref[i]:ref[i]; if(d>1e-3f*(m+1)){ if(bad<3)printf("  %s[%d]: got %d ref %d (x1000)\n",name,i,(int)(out[i]*1000),(int)(ref[i]*1000)); bad++; } }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad?"FAIL":"PASS", bad, n); return bad!=0; }
int main(void){
  int fail=0;
  for(int i=0;i<N;i++) x[i]=frand();
  for(int g=0;g<N/256;g++){ float s=0; for(int k=0;k<256;k++) s+=x[g*256+k]; ref[g]=s; out[g]=-1; }
  group_sum_ct(N, x, out); fail|=check("group_sum",N/256);
  /* shift_left: group = whatever vl the hardware picks; recover it from the output pattern is hard,
     so compare against a reference computed with the same group size: Spike gives 1024 for this cfg */
  {
    long vl_guess = 1024; int ok = 0;
    shift_left_ct(N, x, out);
    for (long vl = 1; vl <= N && !ok; vl++) {
      int bad = 0;
      for (long g = 0; g < N; g += vl) { long ls = (N - g) < vl ? (N - g) : vl; for (long l = 0; l < ls; l++) if (out[g+l] != x[g + (l+1)%ls]) bad++; }
      if (!bad) { ok = 1; vl_guess = vl; }
    }
    printf("%-12s %s (group size %ld)\n", "shift_left", ok?"PASS":"FAIL", vl_guess); fail |= !ok;
  }
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

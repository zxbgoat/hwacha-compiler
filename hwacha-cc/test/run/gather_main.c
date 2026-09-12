#include <stdio.h>
#include "util.h"
#define N 3000
void gather_ct(long n, const int *idx, const float *tab, float *out);
void scatter_add_ct(long n, const int *idx, const float *v, float *out);
void mixed_ct(long n, int m, const float *x, const float *tab, float *out);
static int idx[N]; static float tab[N], v[N], out[N], ref[N];
static unsigned st = 777; static float frand(void){ st = st*1103515245u+12345u; return ((int)(st>>8)%2000-1000)/100.0f; }
static int check(const char*name){ int bad=0; for(int i=0;i<N;i++){ float d=out[i]-ref[i]; if(d<0)d=-d; float m=ref[i]<0?-ref[i]:ref[i]; if(d>1e-5f*(m+1)){ if(bad<3)printf("  %s[%d]: got %d ref %d (x1000)\n",name,i,(int)(out[i]*1000),(int)(ref[i]*1000)); bad++; } }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad?"FAIL":"PASS", bad, N); return bad!=0; }
int main(void){
  int fail=0;
  for(int i=0;i<N;i++){ tab[i]=frand(); idx[i]=(i*7919)%N; }        /* permutation */
  for(int i=0;i<N;i++){ ref[i]=tab[idx[i]]; out[i]=-1; }
  gather_ct(N, idx, tab, out); fail|=check("gather");
  for(int i=0;i<N;i++){ v[i]=frand(); out[i]=-1; } for(int i=0;i<N;i++) ref[idx[i]]=v[i]*2.0f;
  scatter_add_ct(N, idx, v, out); fail|=check("scatter");
  int m=123; for(int i=0;i<N;i++){ v[i]=frand(); float t=tab[m]; ref[i]=v[i]>t? v[i]-t : t-v[i]; out[i]=-1; }
  mixed_ct(N, m, v, tab, out); fail|=check("mixed");
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

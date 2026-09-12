#include <stdio.h>
#include "util.h"
#define N 2500
void cond_store_ct(long n, const int *x, int *y);
void nested_if_ct(long n, const float *x, float *y);
void divloop_ct(long n, const int *cnt, const float *x, float *out);
void uloop_ct(long n, int m, const float *x, float *out);
void divloop_vv_ct(long n, const int *cnt, int *out);
static int xi[N], yi[N], refi[N], cnt[N]; static float xf[N], yf[N], reff[N];
static unsigned st = 4242; static int irand(void){ st = st*1103515245u+12345u; return (int)(st>>8)%2001-1000; }
static float frand(void){ return irand()/50.0f; }
static int checkf(const char*name){ int bad=0; for(int i=0;i<N;i++){ float d=yf[i]-reff[i]; if(d<0)d=-d; float m=reff[i]<0?-reff[i]:reff[i]; if(d>1e-4f*(m+1)){ if(bad<3)printf("  %s[%d]: got %d ref %d (x1000)\n",name,i,(int)(yf[i]*1000),(int)(reff[i]*1000)); bad++; } }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad?"FAIL":"PASS", bad, N); return bad!=0; }
static int checki(const char*name){ int bad=0; for(int i=0;i<N;i++) if(yi[i]!=refi[i]){ if(bad<3)printf("  %s[%d]: got %d ref %d\n",name,i,yi[i],refi[i]); bad++; }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad?"FAIL":"PASS", bad, N); return bad!=0; }
int main(void){
  int fail=0;
  for(int i=0;i<N;i++){ xi[i]=irand(); yi[i]=-7; refi[i]= xi[i]>0 ? xi[i]*2 : -7; }
  cond_store_ct(N, xi, yi); fail|=checki("cond_store");
  for(int i=0;i<N;i++){ xf[i]=frand(); float v=xf[i], r; if(v>0){ r = v>10.0f?10.0f:v; } else r=-v; reff[i]=r*2.0f; yf[i]=-1; }
  nested_if_ct(N, xf, yf); fail|=checkf("nested_if");
  for(int i=0;i<N;i++){ cnt[i]=(irand()&15); xf[i]=frand(); }
  for(int i=0;i<N;i++){ float s=0; for(int k=0;k<cnt[i];k++) s+=xf[k]; reff[i]=s; yf[i]=-1; }
  divloop_ct(N, cnt, xf, yf); fail|=checkf("divloop");
  for(int i=0;i<N;i++){ float s=0,v=xf[i]; for(int k=0;k<7;k++) s=s*0.5f+v; reff[i]=s; yf[i]=-1; }
  uloop_ct(N, 7, xf, yf); fail|=checkf("uloop");
  for(int i=0;i<N;i++){ int acc=i; for(int k=0;k<cnt[i];k++) acc=acc*3+k; refi[i]=acc; yi[i]=-1; }
  divloop_vv_ct(N, cnt, yi); fail|=checki("divloop_vv");
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

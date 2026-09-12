#include <stdio.h>
#include "util.h"
#define N 1200
#define M 6
void window_ct(long n, int m, const float *x, float *out);
void strided_ct(long n, int ld, const float *a, float *out);
void stencil_ct(long n, const float *x, float *out);
void matvec_ct(long n, int nn, const float *A, const float *v, float *y);
static float x[N*M+8], out[N+8], ref[N+8], A[N*N/8], v[N];
static unsigned st = 99; static float frand(void){ st = st*1103515245u+12345u; return ((int)(st>>8)%2000-1000)/100.0f; }
static int check(const char*name,int n){ int bad=0; for(int i=0;i<n;i++){ float d=out[i]-ref[i]; if(d<0)d=-d; float m=ref[i]<0?-ref[i]:ref[i]; if(d>1e-4f*(m+1)){ if(bad<3)printf("  %s[%d]: got %d ref %d (x1000)\n",name,i,(int)(out[i]*1000),(int)(ref[i]*1000)); bad++; } }
  printf("%-12s %s (%d mismatches / %d)\n", name, bad?"FAIL":"PASS", bad, n); return bad!=0; }
int main(void){
  int fail=0;
  for(int i=0;i<N*M+8;i++) x[i]=frand();
  for(int i=0;i<N;i++){ float s=0; for(int k=0;k<M;k++) s+=x[i+k]; ref[i]=s; out[i]=-1; }
  window_ct(N, M, x, out); fail|=check("window",N);
  for(int i=0;i<N;i++){ ref[i]=x[i*M]; out[i]=-1; }
  strided_ct(N, M, x, out); fail|=check("strided",N);
  for(int i=1;i<=N;i++){ ref[i]=0.25f*x[i-1]+0.5f*x[i]+0.25f*x[i+1]; out[i]=-1; }
  stencil_ct(N, x, out); fail|=check("stencil",N+1);   /* out[0] untouched: ref[0]=out[0] */
  ref[0]=out[0];
  int n=64; for(int i=0;i<n*n;i++) A[i]=frand(); for(int j=0;j<n;j++) v[j]=frand();
  for(int i=0;i<n;i++){ float s=0; for(int j=0;j<n;j++) s+=A[i*n+j]*v[j]; ref[i]=s; out[i]=-1; }
  matvec_ct(n, n, A, v, out); fail|=check("matvec",n);
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

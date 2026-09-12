// Cycle measurements: each kernel as (1) hwacha-cc output, (2) hand-written Hwacha asm where
// available (from scratch/spmd-spec), (3) plain scalar C on the Rocket core.
#include <stdio.h>
#include "util.h"
#ifndef N
#define N 4096
#endif
void saxpy_ct(long n, float a, const float *x, float *y);
void clamp_scale_ct(long n, float a, const float *x, float *y);
void divloop_ct(long n, const int *cnt, const float *x, float *out);
void stencil_ct(long n, const float *x, float *out);
void gather_ct(long n, const int *idx, const float *tab, float *out);
#ifdef HAVE_HAND
void hand_saxpy_ct(long n, float a, const float *x, float *y);
void hand_clamp_scale_ct(long n, float a, const float *x, float *y);
#endif
static float x[N + 8], y[N + 8], ref[N + 8]; static int cnt[N], idx[N];
static unsigned st = 7; static float frand(void){ st = st*1103515245u+12345u; return ((int)(st>>8)%2000-1000)/100.0f; }
static inline unsigned long cycles_now(void){ unsigned long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
static void scalar_saxpy(long n, float a, const float *x, float *y){ for(long i=0;i<n;i++) y[i]=a*x[i]+y[i]; }
static void scalar_clamp(long n, float a, const float *x, float *y){ for(long i=0;i<n;i++){ float v=x[i]; y[i]= v>0.0f ? a*v : 0.0f; } }
static void scalar_divloop(long n, const int *c, const float *x, float *o){ for(long i=0;i<n;i++){ float s=0; for(int k=0;k<c[i];k++) s+=x[k]; o[i]=s; } }
static void scalar_stencil(long n, const float *x, float *o){ for(long i=1;i<=n;i++) o[i]=0.25f*x[i-1]+0.5f*x[i]+0.25f*x[i+1]; }
static void scalar_gather(long n, const int *idx, const float *t, float *o){ for(long i=0;i<n;i++) o[i]=t[idx[i]]; }
static int verify_f(const char *tag, const float *got, const float *want, long n){ int bad=0; for(long i=0;i<n;i++){ float d=got[i]-want[i]; if(d<0)d=-d; float m=want[i]<0?-want[i]:want[i]; if(d>1e-4f*(m+1)) bad++; } if(bad) printf("  !! %s: %d mismatches\n",tag,bad); return bad; }
#define TIME(tag, call, chk) do { unsigned long c0=cycles_now(); call; unsigned long c1=cycles_now(); unsigned long cyc=c1-c0; \
  printf("%s: %lu cycles, %lu.%02lu cyc/elem%s\n", tag, cyc, cyc/N, (cyc*100/N)%100, (chk)?"  MISMATCH":""); } while(0)
int main(void){
  int fail = 0; const float a = 1.5f;
  printf("N=%d\n", N);
  /* saxpy */
  for(int i=0;i<N+8;i++){ x[i]=frand(); y[i]=frand(); ref[i]=y[i]; }
  TIME("saxpy scalar", scalar_saxpy(N,a,x,ref), 0);
  for(int i=0;i<N;i++) y[i]=frand(); for(int i=0;i<N;i++){ ref[i]=a*x[i]+y[i]; }
  TIME("saxpy hwacha-cc", saxpy_ct(N,a,x,y), fail|=verify_f("saxpy",y,ref,N));
#ifdef HAVE_HAND
  for(int i=0;i<N;i++){ y[i]=frand(); ref[i]=a*x[i]+y[i]; }
  TIME("saxpy hand-written", hand_saxpy_ct(N,a,x,y), fail|=verify_f("saxpy-hand",y,ref,N));
#endif
  /* clamp_scale */
  TIME("clamp scalar", scalar_clamp(N,a,x,ref), 0);
  for(int i=0;i<N;i++){ ref[i]= x[i]>0.0f ? a*x[i] : 0.0f; y[i]=-1; }
  TIME("clamp hwacha-cc", clamp_scale_ct(N,a,x,y), fail|=verify_f("clamp",y,ref,N));
#ifdef HAVE_HAND
  for(int i=0;i<N;i++) y[i]=-1;
  TIME("clamp hand-written", hand_clamp_scale_ct(N,a,x,y), fail|=verify_f("clamp-hand",y,ref,N));
#endif
  /* divloop: trip counts 0..15 */
  for(int i=0;i<N;i++) cnt[i]=(int)(st=st*1103515245u+12345u, (st>>8)&15);
  TIME("divloop scalar", scalar_divloop(N,cnt,x,ref), 0);
  for(int i=0;i<N;i++) y[i]=-1;
  TIME("divloop hwacha-cc", divloop_ct(N,cnt,x,y), fail|=verify_f("divloop",y,ref,N));
  /* stencil */
  TIME("stencil scalar", scalar_stencil(N,x,ref), 0);
  for(int i=0;i<N+2;i++) y[i]=-1;
  TIME("stencil hwacha-cc", stencil_ct(N,x,y), fail|=verify_f("stencil",y+1,ref+1,N));
  /* gather (random permutation) */
  for(int i=0;i<N;i++) idx[i]=(int)(((long)i*7919)%N);
  TIME("gather scalar", scalar_gather(N,idx,x,ref), 0);
  for(int i=0;i<N;i++) y[i]=-1;
  TIME("gather hwacha-cc", gather_ct(N,idx,x,y), fail|=verify_f("gather",y,ref,N));
  printf("%s\n", fail ? "VERIFY FAILED" : "ALL VERIFIED");
  return fail;
}

#include <stdio.h>
#include "util.h"
#define N 64
void divloop_ct(long n, const int *cnt, const float *x, float *out);
static int cnt[N]; static float x[N], out[N], ref[N];
int main(void){
  for(int i=0;i<N;i++){ cnt[i]=i%5; x[i]=i*0.5f; float s=0; for(int k=0;k<cnt[i];k++) s+=x[k]; ref[i]=s; out[i]=-1; }
  printf("start\n");
  divloop_ct(N, cnt, x, out);
  int bad=0; for(int i=0;i<N;i++) if(out[i]!=ref[i]) bad++;
  printf("divloop N=%d bad=%d\n", N, bad); return bad; }

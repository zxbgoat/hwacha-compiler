#include <stdio.h>
#include "util.h"
#define N 300
extern long hwacha_group_size;
void flagtest_ct(long n, const int *x, int *out, int iters);
void flag_noloop_ct(long n, const int *x, int *out);
void flag_loop_nobreak_ct(long n, const int *x, int *out, int iters);
static int x[N], out[N];
static int check(const char *name, int iters){ int bad=0; for(int i=0;i<N;i++){ int tx=i%128; int c = (iters<0)? (tx>=1&&tx<=100) : (tx>=iters&&tx<=100); int want = c? x[i]+1 : -1; if(out[i]!=want){ if(bad<4) printf("  %s[%d]: got %d want %d\n",name,i,out[i],want); bad++; } }
  printf("%-18s %s (%d mismatches)\n", name, bad?"FAIL":"PASS", bad); return bad!=0; }
int main(void){ int fail=0; hwacha_group_size = 128; for(int i=0;i<N;i++) x[i]=i*3;
  for(int i=0;i<N;i++) out[i]=-1; flag_noloop_ct(N,x,out); fail|=check("flag_noloop",-1);
  for(int i=0;i<N;i++) out[i]=-1; flag_loop_nobreak_ct(N,x,out,3); fail|=check("flag_loop_nobreak",3);
  for(int i=0;i<N;i++) out[i]=-1; flagtest_ct(N,x,out,3); fail|=check("flagtest",3);
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

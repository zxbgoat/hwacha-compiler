#include <stdio.h>
#include "util.h"
#define N 300
void sparse_bs_ct(long n, char *out, const int *sel); void half_bs_ct(long n, char *out); void sparse_sh_ct(long n, short *out, const int *sel);
static int sel[N]; static char bo[N]; static short so[N];
int main(void){ int fail=0; for(int i=0;i<N;i++){ sel[i]=(i%3==0); bo[i]=-1; so[i]=-1; }
  sparse_bs_ct(N, bo, sel); { int bad=0; for(int i=0;i<N;i++){ char want = sel[i]?1:-1; if(bo[i]!=want) bad++; } printf("sparse_bs %s (%d)\n", bad?"FAIL":"PASS", bad); fail|=bad; }
  for(int i=0;i<N;i++) bo[i]=-1; half_bs_ct(N, bo); { int bad=0; for(int i=0;i<N;i++){ char want = (i&1)?1:-1; if(bo[i]!=want) bad++; } printf("half_bs %s (%d)\n", bad?"FAIL":"PASS", bad); fail|=bad; }
  sparse_sh_ct(N, so, sel); { int bad=0; for(int i=0;i<N;i++){ short want = sel[i]?1:-1; if(so[i]!=want) bad++; } printf("sparse_sh %s (%d)\n", bad?"FAIL":"PASS", bad); fail|=bad; }
  return fail; }

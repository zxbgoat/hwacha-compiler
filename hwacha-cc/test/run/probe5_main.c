#include <stdio.h>
#include "util.h"
#define N 300
void sparse_sxb_ct(long n, char *out, const int *sel, const int *idx); void sparse_sh_ct(long n, short *out, const int *sel);
void sparse_bs3_ct(long n, char *out, const int *sel); void half_bs_ct(long n, char *out);
static int sel[N], sel3[N], idx[N]; static char bo[N]; static short so[N];
int main(void){ for(int i=0;i<N;i++){ sel[i]=(i==0); sel3[i]=(i==3); idx[i]=(i*7)%N; }
  printf("start half_bs\n"); half_bs_ct(N, bo); printf("half_bs done\n");
  printf("start sparse_sh\n"); sparse_sh_ct(N, so, sel); printf("sparse_sh done\n");
  printf("start sparse_sxb\n"); sparse_sxb_ct(N, bo, sel, idx); printf("sparse_sxb done\n");
  printf("start sparse_bs3\n"); sparse_bs3_ct(N, bo, sel3); printf("sparse_bs3 done\n");
  printf("ALL DONE\n"); return 0; }

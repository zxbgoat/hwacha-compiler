#include <stdio.h>
#include "util.h"
#define N 300
void sparse_st_ct(long n, int *out, const int *sel); void sparse_sc_ct(long n, int *out, const int *sel, const int *idx);
void sparse_bs_ct(long n, char *out, const int *sel); void sparse_ld_ct(long n, int *out, const int *sel, const int *a);
static int out[N], sel[N], idx[N], a[N]; static char bo[N];
int main(void){ for(int i=0;i<N;i++){ sel[i]=(i==0); idx[i]=(i*7)%N; a[i]=i; }
  printf("start sparse_ld\n"); sparse_ld_ct(N, out, sel, a); printf("sparse_ld done out0=%d\n", out[0]);
  printf("start sparse_st\n"); sparse_st_ct(N, out, sel); printf("sparse_st done out0=%d\n", out[0]);
  printf("start sparse_sc\n"); sparse_sc_ct(N, out, sel, idx); printf("sparse_sc done\n");
  printf("start sparse_bs\n"); sparse_bs_ct(N, bo, sel); printf("sparse_bs done bo0=%d\n", bo[0]);
  printf("ALL DONE\n"); return 0; }

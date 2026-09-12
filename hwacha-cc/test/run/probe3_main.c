#include <stdio.h>
#include "util.h"
#define N 200
void st_ld_ct(long n, int *a, const int *idx, int *out); void ld_st_ct(long n, int *a, const int *idx, int *out);
void st_ld_loop_ct(long n, int *a, const int *idx, int *out, int iters); void bst_bld_ct(long n, char *m, const int *idx, int *out);
static int a[N], idx[N], out[N]; static char m[N];
int main(void){ for(int i=0;i<N;i++){ idx[i]=(i*7)%N; a[i]=0; m[i]=0; }
  printf("start st_ld\n"); st_ld_ct(N, a, idx, out); printf("st_ld done\n");
  printf("start ld_st\n"); ld_st_ct(N, a, idx, out); printf("ld_st done\n");
  printf("start st_ld_loop\n"); st_ld_loop_ct(N, a, idx, out, 3); printf("st_ld_loop done\n");
  printf("start bst_bld\n"); bst_bld_ct(N, m, idx, out); printf("bst_bld done\n");
  printf("ALL DONE\n"); return 0; }

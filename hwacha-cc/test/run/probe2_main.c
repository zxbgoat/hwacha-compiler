#include <stdio.h>
#include "util.h"
#define N 200
void uni_store_ct(long n, int *flag, char *bflag, int *out); void mul64_ct(long n, const long *a, const long *b, long *c);
void dblfma_ct(long n, const double *x, double *y, double a); void atom_ct(long n, const int *bin, int *hist); void mul32_ct(long n, const int *a, const int *b, int *c);
static int flag, io[N], ia[N], ib[N], ic[N], bin[N], hist[16]; static char bflag; static long la[N], lb[N], lc[N]; static double dx[N], dy[N];
int main(void){ for(int i=0;i<N;i++){ ia[i]=i; ib[i]=3; la[i]=i*1000003L; lb[i]=7; dx[i]=i*0.5; dy[i]=1.0; bin[i]=i&15; }
  printf("start mul32\n"); mul32_ct(N, ia, ib, ic); { int bad=0; for(int i=0;i<N;i++) if(ic[i]!=i*3) bad++; printf("mul32 %s\n", bad?"FAIL":"PASS"); }
  printf("start uni_store\n"); uni_store_ct(N, &flag, &bflag, io); printf("uni_store %s (flag=%d bflag=%d)\n", (flag==7&&bflag==3)?"PASS":"FAIL", flag, bflag);
  printf("start mul64\n"); mul64_ct(N, la, lb, lc); { int bad=0; for(int i=0;i<N;i++) if(lc[i]!=la[i]*7) bad++; printf("mul64 %s\n", bad?"FAIL":"PASS"); }
  printf("start dblfma\n"); dblfma_ct(N, dx, dy, 2.0); { int bad=0; for(int i=0;i<N;i++) if(dy[i]!=2.0*dx[i]+1.0) bad++; printf("dblfma %s\n", bad?"FAIL":"PASS"); }
  printf("start atom\n"); atom_ct(N, bin, hist); { int bad=0; for(int b=0;b<16;b++){ int c=0; for(int i=0;i<N;i++) if(bin[i]==b) c++; if(hist[b]!=c) bad++; } printf("atom %s\n", bad?"FAIL":"PASS"); }
  printf("ALL DONE\n"); return 0; }

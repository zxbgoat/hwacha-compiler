#include <stdio.h>
#include "util.h"
#define N 300
void bscatter_ct(long n, const int *idx, char *out);
void bgather_ct(long n, const int *idx, const char *tab, char *out);
void wscatter_ct(long n, const int *idx, int *out);
static int idx[N], wout[N]; static char bout[N], tab[N];
static inline unsigned long cyc(void){ unsigned long c; asm volatile("rdcycle %0":"=r"(c)); return c; }
int main(void){ int fail=0; for(int i=0;i<N;i++){ idx[i]=(i*7)%N; tab[i]=(char)(i*3); }
  unsigned long c0=cyc(); for(int i=0;i<N;i++) wout[i]=-1; wscatter_ct(N, idx, wout); unsigned long c1=cyc(); { int bad=0; for(int i=0;i<N;i++) if(wout[idx[i]]!=i) bad++; printf("wscatter %s (%d) %lu cycles\n", bad?"FAIL":"PASS", bad, c1-c0); fail|=bad!=0; }
  c0=cyc(); for(int i=0;i<N;i++) bout[i]=-1; bscatter_ct(N, idx, bout); c1=cyc(); { int bad=0; for(int i=0;i<N;i++) if(bout[idx[i]]!=(char)(i&0x7f)) bad++; printf("bscatter %s (%d) %lu cycles\n", bad?"FAIL":"PASS", bad, c1-c0); fail|=bad!=0; }
  c0=cyc(); for(int i=0;i<N;i++) bout[i]=-1; bgather_ct(N, idx, tab, bout); c1=cyc(); { int bad=0; for(int i=0;i<N;i++) if(bout[i]!=tab[idx[i]]) bad++; printf("bgather %s (%d) %lu cycles\n", bad?"FAIL":"PASS", bad, c1-c0); fail|=bad!=0; }
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

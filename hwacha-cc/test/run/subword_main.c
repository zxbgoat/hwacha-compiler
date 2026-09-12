#include <stdio.h>
#include "util.h"
#define N 200
void u8copy_ct(long n, const signed char *a, signed char *b); void u8load_ct(long n, const signed char *a, int *b); void u8store_ct(long n, const int *a, signed char *b);
void u16copy_ct(long n, const short *a, short *b); void u8gath_ct(long n, const signed char *a, const int *idx, int *b);
static signed char ca[N], cb[N];   /* OpenCL char is signed; RISC-V C char is unsigned */ static int ia[N], ib[N], idx[N]; static short sa[N], sb[N];
static inline unsigned long cyc(void){ unsigned long c; asm volatile("rdcycle %0":"=r"(c)); return c; }
#define T(name, call, chk) do { unsigned long c0=cyc(); call; unsigned long c1=cyc(); int bad=0; for(int i=0;i<N;i++) if(!(chk)) bad++; printf("%-8s %s (%d) %lu cycles\n", name, bad?"FAIL":"PASS", bad, c1-c0); } while(0)
int main(void){ for(int i=0;i<N;i++){ ca[i]=(char)(i*5); ia[i]=i*11; sa[i]=(short)(i*77); idx[i]=(i*7)%N; }
  T("u8store", u8store_ct(N, ia, cb), cb[i]==(signed char)ia[i]);
  T("u8load",  u8load_ct(N, ca, ib),  ib[i]==ca[i]);
  T("u8copy",  u8copy_ct(N, ca, cb),  cb[i]==ca[i]);
  T("u16copy", u16copy_ct(N, sa, sb), sb[i]==sa[i]);
  T("u8gath",  u8gath_ct(N, ca, idx, ib), ib[i]==ca[idx[i]]);
  printf("ALL DONE\n"); return 0; }

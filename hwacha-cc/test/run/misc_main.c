#include <stdio.h>
#include "util.h"
#define N 4000
void hist_ct(long n, const unsigned char *pix, int *hist);
void widen_ct(long n, const short *a, const unsigned short *b, long *out);
void dbl_ct(long n, const double *x, double *y, double a);
void bytes_ct(long n, const unsigned char *a, unsigned char *b);
static unsigned char pix[N], bo[N]; static int hist[256], hr[256]; static short sa[N]; static unsigned short ub[N]; static long lo[N], lr[N];
static double dx[N], dy[N], dr[N];
static unsigned st = 5; static unsigned r32(void){ st = st*1103515245u+12345u; return st>>8; }
int main(void){
  int fail=0, bad;
  for(int i=0;i<N;i++){ pix[i]=r32()&255; hr[pix[i]]++; }
  hist_ct(N, pix, hist); bad=0; for(int i=0;i<256;i++) if(hist[i]!=hr[i]) bad++;
  printf("%-12s %s (%d mismatches / 256)\n","hist",bad?"FAIL":"PASS",bad); fail|=bad!=0;
  for(int i=0;i<N;i++){ sa[i]=(short)r32(); ub[i]=(unsigned short)r32(); lr[i]=(long)sa[i]*(long)ub[i]; lo[i]=-1; }
  widen_ct(N, sa, ub, lo); bad=0; for(int i=0;i<N;i++) if(lo[i]!=lr[i]) bad++;
  printf("%-12s %s (%d mismatches / %d)\n","widen",bad?"FAIL":"PASS",bad,N); fail|=bad!=0;
  for(int i=0;i<N;i++){ dx[i]=(r32()%2000-1000)/100.0; dy[i]=(r32()%2000-1000)/100.0; dr[i]=2.5*dx[i]+dy[i]; }
  dbl_ct(N, dx, dy, 2.5); bad=0; for(int i=0;i<N;i++){ double d=dy[i]-dr[i]; if(d<0)d=-d; if(d>1e-9) bad++; }
  printf("%-12s %s (%d mismatches / %d)\n","dbl",bad?"FAIL":"PASS",bad,N); fail|=bad!=0;
  bytes_ct(N, pix, bo); bad=0; for(int i=0;i<N;i++) if(bo[i]!=(unsigned char)(pix[i]+100)) bad++;
  printf("%-12s %s (%d mismatches / %d)\n","bytes",bad?"FAIL":"PASS",bad,N); fail|=bad!=0;
  printf("%s\n", fail?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return fail; }

#include <stdio.h>
#include "util.h"
#define COLS 500
#define BLOCK 128
extern long hwacha_group_size, hwacha_vl_short;
void k1_ct(long n, int *src, int *out, int cols, int border, int *prev);
void k2_ct(long n, int *src, int *out, int cols, int border);
void k3_ct(long n, int *src, int *out, int cols);
void k4_ct(long n, int iteration, int *wall, int *src, int *res, int cols, int startStep, int border, int *prev, int *result);
static int wall[2*COLS], resb[BLOCK], dbg[COLS*6];
void k5_ct(long n, int iteration, int *src, int *dbg, int cols, int border, int *prev);
void k6_ct(long n, int *dbg, int vmin, int vmax);
void k7_ct(long n, int iteration, int *dbg, int cols, int border);
static int src[COLS], out[COLS], prev[BLOCK];
static int run(const char *name, int which){ int small=BLOCK-2, border=1, blocks=COLS/small+1; for(int i=0;i<COLS;i++){ src[i]=i*7; out[i]=-1; }
  if(which==1) k1_ct((long)blocks*BLOCK, src, out, COLS, border, prev); else if(which==2) k2_ct((long)blocks*BLOCK, src, out, COLS, border); else k3_ct(COLS, src, out, COLS);
  int bad=0; for(int i=0;i<COLS;i++){ int want = which==3 ? src[i] + (i/BLOCK)*1000 + (i%BLOCK) : src[i]+1; if(out[i]!=want){ if(bad<3) printf("  %s[%d]: got %d want %d\n",name,i,out[i],want); bad++; } }
  printf("%-4s %s (%d mismatches)%s\n", name, bad?"FAIL":"PASS", bad, hwacha_vl_short?"  VL SHORT":""); return bad!=0; }
static int run4(void){ int iteration=1, small=BLOCK-2, border=1, blocks=COLS/small+1; for(int i=0;i<2*COLS;i++) wall[i]=i%9; for(int i=0;i<COLS;i++){ src[i]=i*7; out[i]=-1; }
  k4_ct((long)blocks*BLOCK, iteration, wall+COLS, src, out, COLS, 0, border, prev, resb);
  int bad=0; for(int i=0;i<COLS;i++){ int want = src[i] + wall[COLS+i]; if(out[i]!=want){ if(bad<3) printf("  k4[%d]: got %d want %d\n",i,out[i],want); bad++; } }
  printf("k4   %s (%d mismatches)\n", bad?"FAIL":"PASS", bad); return bad!=0; }
int main(void){ hwacha_group_size = BLOCK; int f=0;
  { for(int i=0;i<COLS*4;i++) dbg[i]=-9; k7_ct(256, 1, dbg, COLS, 1); for (int g = 0; g < 200; g += 40) printf("  k7 gid=%d: b1=%d b2=%d valid=%d vminmax=%d\n", g, dbg[g*4], dbg[g*4+1], dbg[g*4+2], dbg[g*4+3]); }
  { for(int i=0;i<COLS*4;i++) dbg[i]=-9; k6_ct(16, dbg, 1, 10); for (int g = 0; g < 13; g += 3) printf("  k6 tx=%d: b1=%d b2=%d b3=%d sel=%d\n", g, dbg[g*4], dbg[g*4+1], dbg[g*4+2], dbg[g*4+3]); }
  { int small=BLOCK-2, border=1, blocks=COLS/small+1; for(int i=0;i<COLS*6;i++) dbg[i]=-9; k5_ct((long)blocks*BLOCK, 1, src, dbg, COLS, border, prev);
    for (int x = 0; x < COLS; x += 63) printf("  k5 x=%d: vmin=%d vmax=%d valid=%d computed=%d blkX=%d small=%d\n", x, dbg[x*6], dbg[x*6+1], dbg[x*6+2], dbg[x*6+3], dbg[x*6+4], dbg[x*6+5]); } f|=run("k3",3); f|=run("k2",2); f|=run("k1",1); f|=run4(); printf("%s\n", f?"SOME KERNELS FAILED":"ALL KERNELS PASSED"); return f; }

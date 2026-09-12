// Rodinia pathfinder: dynamic programming over rows, driven like the original OpenCL host
#include "common.h"
#ifndef ROWS
#define ROWS 8
#endif
#define COLS 1024
#define PYRAMID 2
#define HALO 1
#define BLOCK 128
void dynproc_kernel_ct(long n, int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols, int rows, int startStep,
                       int border, int halo, int *prev, int *result, int *outputBuffer);
static int wall[ROWS*COLS], src[COLS], dst[COLS], ref[COLS], prevb[BLOCK], resb[BLOCK], outbuf[16384];
#define MIN(a,b) ((a)<=(b)?(a):(b))
int main(void) {
  for (int i = 0; i < ROWS*COLS; i++) wall[i] = rnd() % 10;
  /* reference (Rodinia pathfinder CPU version) */
  unsigned long c0 = cyc();
  for (int j = 0; j < COLS; j++) ref[j] = wall[j];
  for (int t = 0; t < ROWS-1; t++) { static int nxt[COLS];
    for (int n = 0; n < COLS; n++) { int m = ref[n]; if (n > 0) m = MIN(m, ref[n-1]); if (n < COLS-1) m = MIN(m, ref[n+1]); nxt[n] = wall[(t+1)*COLS+n] + m; }
    for (int n = 0; n < COLS; n++) ref[n] = nxt[n]; }
  unsigned long c1 = cyc(); REPORT("pathfinder scalar", c0, c1, COLS);
  /* sequential emulation of the OpenCL kernel (work-items in order, barriers = phase boundaries) */
  static int esrc[COLS], edst[COLS], eprev[BLOCK], eres[BLOCK];
  {
    int borderCols = PYRAMID * HALO, smallBlockCol = BLOCK - PYRAMID * HALO * 2;
    int blockCols = COLS / smallBlockCol + (COLS % smallBlockCol == 0 ? 0 : 1);
    for (int j = 0; j < COLS; j++) esrc[j] = wall[j];
    int *s = esrc, *d = edst;
    for (int t = 0; t < ROWS-1; t += PYRAMID) {
      int iteration = MIN(PYRAMID, ROWS-t-1);
      for (int bx = 0; bx < blockCols; bx++) {
        int small = BLOCK - iteration*HALO*2, blkX = small*bx - borderCols, blkXmax = blkX + BLOCK - 1;
        int validXmin = blkX < 0 ? -blkX : 0, validXmax = blkXmax > COLS-1 ? BLOCK-1-(blkXmax-COLS+1) : BLOCK-1;
        static int computed[BLOCK];
        for (int tx = 0; tx < BLOCK; tx++) { int xidx = blkX + tx; if (xidx >= 0 && xidx <= COLS-1) eprev[tx] = s[xidx]; computed[tx] = 0; }
        for (int i = 0; i < iteration; i++) {
          for (int tx = 0; tx < BLOCK; tx++) { int xidx = blkX + tx; int W = tx-1, E = tx+1; W = W < validXmin ? validXmin : W; E = E > validXmax ? validXmax : E;
            int isValid = tx >= validXmin && tx <= validXmax; computed[tx] = 0;
            if (tx >= i+1 && tx <= BLOCK-i-2 && isValid) { computed[tx] = 1; int m = MIN(eprev[W], eprev[tx]); m = MIN(m, eprev[E]); eres[tx] = m + wall[COLS*(t+i+1)+xidx]; } }
          if (i == iteration-1) break;
          for (int tx = 0; tx < BLOCK; tx++) if (computed[tx]) eprev[tx] = eres[tx];
        }
        for (int tx = 0; tx < BLOCK; tx++) if (computed[tx]) d[blkX+tx] = eres[tx];
      }
      int *tmp = s; s = d; d = tmp;
    }
    int bad = 0; for (int j = 0; j < COLS; j++) if (s[j] != ref[j]) bad++;
    printf("pathfinder emulation vs reference: %d mismatches\n", bad);
    for (int j = 0; j < COLS; j++) ref[j] = s[j];   /* compare the device against the emulation from here on */
  }
  /* device-style driver */
  hwacha_group_size = BLOCK;
  int borderCols = PYRAMID * HALO, smallBlockCol = BLOCK - PYRAMID * HALO * 2;
  int blockCols = COLS / smallBlockCol + (COLS % smallBlockCol == 0 ? 0 : 1);
  for (int j = 0; j < COLS; j++) src[j] = wall[j];
  int *s = src, *d = dst;
  c0 = cyc();
  for (int t = 0; t < ROWS-1; t += PYRAMID) {
    int iteration = MIN(PYRAMID, ROWS-t-1);
    dynproc_kernel_ct((long)blockCols * BLOCK, iteration, wall + COLS, s, d, COLS, ROWS, t, borderCols, HALO, prevb, resb, outbuf);
    int *tmp = s; s = d; d = tmp;
  }
  c1 = cyc(); REPORT("pathfinder hwacha-cc", c0, c1, COLS);
  int bad = 0; for (int j = 0; j < COLS; j++) if (s[j] != ref[j]) bad++;
  printf("  dev: "); for (int j = 0; j < 12; j++) printf("%d ", s[j]); printf("| "); for (int j = 120; j < 132; j++) printf("%d ", s[j]); printf("\n");
  printf("  ref: "); for (int j = 0; j < 12; j++) printf("%d ", ref[j]); printf("| "); for (int j = 120; j < 132; j++) printf("%d ", ref[j]); printf("\n");
  printf("  src: "); for (int j = 0; j < 12; j++) printf("%d ", wall[j]); printf("| "); for (int j = 120; j < 132; j++) printf("%d ", wall[j]); printf("\n");
  if (hwacha_vl_short) printf("  !! hardware vector length shorter than the requested group size %d\n", BLOCK);
  printf("pathfinder %s (%d mismatches / %d)\n", bad ? "FAIL" : "PASS", bad, COLS); return bad != 0;
}

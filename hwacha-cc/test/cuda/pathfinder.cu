// Rodinia pathfinder (pathfinder.cu): dynproc_kernel with its __shared__ prev/result, barriers inside
// a loop, `computed` carried across the loop; the host drives it in pyramids like the original.
#include <cuda_runtime.h>
#include <stdio.h>
#include "rodinia.h"

#define BLOCK_SIZE 128
#define HALO 1
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

__global__ void dynproc_kernel(int iteration, int *gpuWall, int *gpuSrc, int *gpuResults, int cols, int rows, int startStep, int border) {
  __shared__ int prev[BLOCK_SIZE];
  __shared__ int result[BLOCK_SIZE];
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int small_block_cols = BLOCK_SIZE - iteration * HALO * 2;
  int blkX = small_block_cols * bx - border;
  int blkXmax = blkX + BLOCK_SIZE - 1;
  int xidx = blkX + tx;
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - cols + 1) : BLOCK_SIZE - 1;
  int W = tx - 1;
  int E = tx + 1;
  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;
  bool isValid = IN_RANGE(tx, validXmin, validXmax);
  if (IN_RANGE(xidx, 0, cols - 1)) prev[tx] = gpuSrc[xidx];
  __syncthreads();
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && isValid) {
      computed = true;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      int index = cols * (startStep + i) + xidx;
      result[tx] = shortest + gpuWall[index];
    }
    __syncthreads();
    if (i == iteration - 1) break;
    if (computed) prev[tx] = result[tx];
    __syncthreads();
  }
  if (computed) gpuResults[xidx] = result[tx];
}

#define ROWS 8
#define COLS 1024
#define PYRAMID 2
static int wall[ROWS * COLS], ref[COLS], out[COLS];

int main() {
  for (int i = 0; i < ROWS * COLS; i++) wall[i] = rnd() % 10;
  for (int j = 0; j < COLS; j++) ref[j] = wall[j];
  for (int t = 0; t < ROWS - 1; t++) { static int nxt[COLS];
    for (int n = 0; n < COLS; n++) { int m = ref[n]; if (n > 0) m = MIN(m, ref[n - 1]); if (n < COLS - 1) m = MIN(m, ref[n + 1]); nxt[n] = wall[(t + 1) * COLS + n] + m; }
    for (int n = 0; n < COLS; n++) ref[n] = nxt[n]; }

  // the CUDA host (pathfinder.cu run()): gpuWall = rows 1.., gpuResult[2] ping-pong
  int borderCols = PYRAMID * HALO, smallBlockCol = BLOCK_SIZE - PYRAMID * HALO * 2;
  int blockCols = COLS / smallBlockCol + (COLS % smallBlockCol == 0 ? 0 : 1);
  int *gpuWall, *gpuResult[2];
  cudaMalloc((void **)&gpuResult[0], sizeof(int) * COLS);
  cudaMalloc((void **)&gpuResult[1], sizeof(int) * COLS);
  cudaMalloc((void **)&gpuWall, sizeof(int) * (ROWS * COLS - COLS));
  cudaMemcpy(gpuResult[0], wall, sizeof(int) * COLS, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuWall, wall + COLS, sizeof(int) * (ROWS * COLS - COLS), cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK_SIZE), dimGrid(blockCols);
  int src = 1, dst = 0;
  unsigned long c0 = cyc();
  for (int t = 0; t < ROWS - 1; t += PYRAMID) {
    int temp = src; src = dst; dst = temp;
    dynproc_kernel<<<dimGrid, dimBlock>>>(MIN(PYRAMID, ROWS - t - 1), gpuWall, gpuResult[src], gpuResult[dst], COLS, ROWS, t, borderCols);
  }
  cudaDeviceSynchronize();
  unsigned long c1 = cyc(); REPORT("pathfinder hwacha-cc", c0, c1, COLS);
  cudaMemcpy(out, gpuResult[dst], sizeof(int) * COLS, cudaMemcpyDeviceToHost);
  int bad = 0; for (int j = 0; j < COLS; j++) if (out[j] != ref[j]) bad++;
  printf("pathfinder %s (%d mismatches / %d)\n", bad ? "FAIL" : "PASS", bad, COLS); return bad != 0;
}

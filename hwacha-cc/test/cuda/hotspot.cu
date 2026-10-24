// Rodinia hotspot (hotspot.cu): calculate_temp, a 2-D block (BLOCK_SIZE x BLOCK_SIZE) per tile with
// three 2-D __shared__ arrays and a pyramid of `iteration` steps per launch; 2-D grid of tiles.
#include <cuda_runtime.h>
#include <stdio.h>
#include "rodinia.h"

#define BLOCK_SIZE 8
#define IN_RANGE(x, min, max) ((x) >= (min) && (x) <= (max))
#define EXPAND_RATE 2

__global__ void calculate_temp(int iteration, float *power, float *temp_src, float *temp_dst, int grid_cols, int grid_rows,
                               int border_cols, int border_rows, float Cap, float Rx, float Ry, float Rz, float step, float time_elapsed) {
  __shared__ float temp_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float power_on_cuda[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float temp_t[BLOCK_SIZE][BLOCK_SIZE];
  float amb_temp = 80.0;
  float step_div_Cap;
  float Rx_1, Ry_1, Rz_1;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  step_div_Cap = step / Cap;
  Rx_1 = 1 / Rx;
  Ry_1 = 1 / Ry;
  Rz_1 = 1 / Rz;
  int small_block_rows = BLOCK_SIZE - iteration * 2;
  int small_block_cols = BLOCK_SIZE - iteration * 2;
  int blkY = small_block_rows * by - border_rows;
  int blkX = small_block_cols * bx - border_cols;
  int blkYmax = blkY + BLOCK_SIZE - 1;
  int blkXmax = blkX + BLOCK_SIZE - 1;
  int yidx = blkY + ty;
  int xidx = blkX + tx;
  int loadYidx = yidx, loadXidx = xidx;
  int index = grid_cols * loadYidx + loadXidx;
  if (IN_RANGE(loadYidx, 0, grid_rows - 1) && IN_RANGE(loadXidx, 0, grid_cols - 1)) {
    temp_on_cuda[ty][tx] = temp_src[index];
    power_on_cuda[ty][tx] = power[index];
  }
  __syncthreads();
  int validYmin = (blkY < 0) ? -blkY : 0;
  int validYmax = (blkYmax > grid_rows - 1) ? BLOCK_SIZE - 1 - (blkYmax - grid_rows + 1) : BLOCK_SIZE - 1;
  int validXmin = (blkX < 0) ? -blkX : 0;
  int validXmax = (blkXmax > grid_cols - 1) ? BLOCK_SIZE - 1 - (blkXmax - grid_cols + 1) : BLOCK_SIZE - 1;
  int N = ty - 1;
  int S = ty + 1;
  int W = tx - 1;
  int E = tx + 1;
  N = (N < validYmin) ? validYmin : N;
  S = (S > validYmax) ? validYmax : S;
  W = (W < validXmin) ? validXmin : W;
  E = (E > validXmax) ? validXmax : E;
  bool computed;
  for (int i = 0; i < iteration; i++) {
    computed = false;
    if (IN_RANGE(tx, i + 1, BLOCK_SIZE - i - 2) && IN_RANGE(ty, i + 1, BLOCK_SIZE - i - 2) &&
        IN_RANGE(tx, validXmin, validXmax) && IN_RANGE(ty, validYmin, validYmax)) {
      computed = true;
      temp_t[ty][tx] = temp_on_cuda[ty][tx] + step_div_Cap * (power_on_cuda[ty][tx] +
                       (temp_on_cuda[S][tx] + temp_on_cuda[N][tx] - 2.0 * temp_on_cuda[ty][tx]) * Ry_1 +
                       (temp_on_cuda[ty][E] + temp_on_cuda[ty][W] - 2.0 * temp_on_cuda[ty][tx]) * Rx_1 +
                       (amb_temp - temp_on_cuda[ty][tx]) * Rz_1);
    }
    __syncthreads();
    if (i == iteration - 1) break;
    if (computed) temp_on_cuda[ty][tx] = temp_t[ty][tx];
    __syncthreads();
  }
  if (computed) temp_dst[index] = temp_t[ty][tx];
}

#define ROWS 64
#define COLS 64
#define PYR 2
#define TOTAL 4
static float power[ROWS * COLS], t0[ROWS * COLS], r0[ROWS * COLS], r1[ROWS * COLS], out[ROWS * COLS];
static float h_fabsf(float v) { return v < 0 ? -v : v; }

int main() {
  for (int i = 0; i < ROWS * COLS; i++) { power[i] = frand(0, 1e-3f); t0[i] = r0[i] = frand(320, 340); }
  float t_chip = 0.0005f, chip_height = 0.016f, chip_width = 0.016f, K_SI = 100, FACTOR_CHIP = 0.5f, SPEC_HEAT_SI = 1.75e6f, MAX_PD = 3e6f, PRECISION = 0.001f;
  float grid_height = chip_height / ROWS, grid_width = chip_width / COLS;
  float Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  float Rx = grid_width / (2.0f * K_SI * t_chip * grid_height), Ry = grid_height / (2.0f * K_SI * t_chip * grid_width), Rz = t_chip / (K_SI * grid_height * grid_width);
  float max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI), step = PRECISION / max_slope;
  // scalar reference: TOTAL plain stencil steps with clamped neighbours
  float step_div_Cap = step / Cap, Rx_1 = 1 / Rx, Ry_1 = 1 / Ry, Rz_1 = 1 / Rz, amb = 80.0f;
  float *src = r0, *dst = r1;
  for (int it = 0; it < TOTAL; it++) {
    for (int y = 0; y < ROWS; y++) for (int x = 0; x < COLS; x++) {
      int N = y > 0 ? y - 1 : 0, S = y < ROWS - 1 ? y + 1 : ROWS - 1, W = x > 0 ? x - 1 : 0, E = x < COLS - 1 ? x + 1 : COLS - 1;
      float t = src[y * COLS + x];
      dst[y * COLS + x] = t + step_div_Cap * (power[y * COLS + x] + (src[S * COLS + x] + src[N * COLS + x] - 2.0f * t) * Ry_1 + (src[y * COLS + E] + src[y * COLS + W] - 2.0f * t) * Rx_1 + (amb - t) * Rz_1);
    }
    float *tmp = src; src = dst; dst = tmp;
  }
  float *ref = src;

  // the CUDA host (hotspot.cu compute_tran_temp)
  int borderCols = PYR * EXPAND_RATE / 2, borderRows = PYR * EXPAND_RATE / 2;
  int smallBlockCol = BLOCK_SIZE - PYR * EXPAND_RATE, smallBlockRow = BLOCK_SIZE - PYR * EXPAND_RATE;
  int blockCols = COLS / smallBlockCol + (COLS % smallBlockCol == 0 ? 0 : 1);
  int blockRows = ROWS / smallBlockRow + (ROWS % smallBlockRow == 0 ? 0 : 1);
  float *MatrixTemp[2], *MatrixPower;
  cudaMalloc((void **)&MatrixTemp[0], sizeof(float) * ROWS * COLS);
  cudaMalloc((void **)&MatrixTemp[1], sizeof(float) * ROWS * COLS);
  cudaMalloc((void **)&MatrixPower, sizeof(float) * ROWS * COLS);
  cudaMemcpy(MatrixTemp[0], t0, sizeof(float) * ROWS * COLS, cudaMemcpyHostToDevice);
  cudaMemcpy(MatrixPower, power, sizeof(float) * ROWS * COLS, cudaMemcpyHostToDevice);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE), dimGrid(blockCols, blockRows);
  int s = 1, d = 0;
  unsigned long c0 = cyc();
  for (int t = 0; t < TOTAL; t += PYR) {
    int temp = s; s = d; d = temp;
    calculate_temp<<<dimGrid, dimBlock>>>(TOTAL - t < PYR ? TOTAL - t : PYR, MatrixPower, MatrixTemp[s], MatrixTemp[d], COLS, ROWS, borderCols, borderRows, Cap, Rx, Ry, Rz, step, 0.001f);
  }
  cudaDeviceSynchronize();
  unsigned long c1 = cyc(); REPORT("hotspot hwacha-cc", c0, c1, (long)ROWS * COLS * TOTAL);
  cudaMemcpy(out, MatrixTemp[d], sizeof(float) * ROWS * COLS, cudaMemcpyDeviceToHost);
  float mx = 0; for (int i = 0; i < ROWS * COLS; i++) if (h_fabsf(ref[i]) > mx) mx = h_fabsf(ref[i]);
  int bad = 0; float md = 0;
  for (int i = 0; i < ROWS * COLS; i++) { float dd = h_fabsf(out[i] - ref[i]); if (dd > md) md = dd; if (dd > 1e-4f * mx) bad++; }
  printf("hotspot %s (%d mismatches / %d, max|diff|=%ld/1e6)\n", bad ? "FAIL" : "PASS", bad, ROWS * COLS, (long)(md * 1e6f)); return bad != 0;
}

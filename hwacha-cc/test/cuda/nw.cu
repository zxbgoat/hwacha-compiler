// Rodinia nw (needle.cu + needle_kernel.cu): Needleman-Wunsch, anti-diagonal blocks of BLOCK_SIZE
// with a __device__ helper, 2-D __shared__ arrays and __syncthreads inside the wavefront loops.
#include <cuda_runtime.h>
#include <stdio.h>
#include "rodinia.h"

#define BLOCK_SIZE 16
#define SCORE(i, j) input_itemsets_l[j + i * (BLOCK_SIZE + 1)]
#define REF(i, j) reference_l[j + i * BLOCK_SIZE]

__device__ __host__ int maximum(int a, int b, int c) {
  int k;
  if (a <= b) k = b; else k = a;
  if (k <= c) return c; else return k;
}

__global__ void needle_cuda_shared_1(int *referrence, int *matrix_cuda, int cols, int penalty, int i, int block_width) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int b_index_x = bx;
  int b_index_y = i - 1 - bx;
  int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);
  int index_n = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1);
  int index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols);
  int index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
  __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
  __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];
  if (tx == 0) temp[tx][0] = matrix_cuda[index_nw];
  for (int ty = 0; ty < BLOCK_SIZE; ty++) ref[ty][tx] = referrence[index + cols * ty];
  __syncthreads();
  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];
  __syncthreads();
  temp[0][tx + 1] = matrix_cuda[index_n];
  __syncthreads();
  for (int m = 0; m < BLOCK_SIZE; m++) {
    if (tx <= m) {
      int t_index_x = tx + 1;
      int t_index_y = m - tx + 1;
      temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[t_index_y - 1][t_index_x - 1],
                                           temp[t_index_y][t_index_x - 1] - penalty, temp[t_index_y - 1][t_index_x] - penalty);
    }
    __syncthreads();
  }
  for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
    if (tx <= m) {
      int t_index_x = tx + BLOCK_SIZE - m;
      int t_index_y = BLOCK_SIZE - tx;
      temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[t_index_y - 1][t_index_x - 1],
                                           temp[t_index_y][t_index_x - 1] - penalty, temp[t_index_y - 1][t_index_x] - penalty);
    }
    __syncthreads();
  }
  for (int ty = 0; ty < BLOCK_SIZE; ty++) matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
}

__global__ void needle_cuda_shared_2(int *referrence, int *matrix_cuda, int cols, int penalty, int i, int block_width) {
  int bx = blockIdx.x;
  int tx = threadIdx.x;
  int b_index_x = bx + block_width - i;
  int b_index_y = block_width - bx - 1;
  int index = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (cols + 1);
  int index_n = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + (1);
  int index_w = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + (cols);
  int index_nw = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;
  __shared__ int temp[BLOCK_SIZE + 1][BLOCK_SIZE + 1];
  __shared__ int ref[BLOCK_SIZE][BLOCK_SIZE];
  for (int ty = 0; ty < BLOCK_SIZE; ty++) ref[ty][tx] = referrence[index + cols * ty];
  __syncthreads();
  if (tx == 0) temp[tx][0] = matrix_cuda[index_nw];
  temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];
  __syncthreads();
  temp[0][tx + 1] = matrix_cuda[index_n];
  __syncthreads();
  for (int m = 0; m < BLOCK_SIZE; m++) {
    if (tx <= m) {
      int t_index_x = tx + 1;
      int t_index_y = m - tx + 1;
      temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[t_index_y - 1][t_index_x - 1],
                                           temp[t_index_y][t_index_x - 1] - penalty, temp[t_index_y - 1][t_index_x] - penalty);
    }
    __syncthreads();
  }
  for (int m = BLOCK_SIZE - 2; m >= 0; m--) {
    if (tx <= m) {
      int t_index_x = tx + BLOCK_SIZE - m;
      int t_index_y = BLOCK_SIZE - tx;
      temp[t_index_y][t_index_x] = maximum(temp[t_index_y - 1][t_index_x - 1] + ref[t_index_y - 1][t_index_x - 1],
                                           temp[t_index_y][t_index_x - 1] - penalty, temp[t_index_y - 1][t_index_x] - penalty);
    }
    __syncthreads();
  }
  for (int ty = 0; ty < BLOCK_SIZE; ty++) matrix_cuda[index + ty * cols] = temp[ty + 1][tx + 1];
}

#define N 64
#define PEN 10
#define MC (N + 1)
static int referrence[MC * MC], input_itemsets[MC * MC], rr[MC * MC], output_itemsets[MC * MC];

int main() {
  for (int i = 1; i < MC; i++) for (int j = 1; j < MC; j++) referrence[i * MC + j] = (int)(rnd() % 16) - 4;
  for (int i = 1; i < MC; i++) { input_itemsets[i * MC] = rr[i * MC] = -i * PEN; input_itemsets[i] = rr[i] = -i * PEN; }
  for (int i = 1; i < MC; i++) for (int j = 1; j < MC; j++)
    rr[i * MC + j] = maximum(rr[(i - 1) * MC + j - 1] + referrence[i * MC + j], rr[i * MC + j - 1] - PEN, rr[(i - 1) * MC + j] - PEN);

  // runTest() from needle.cu
  int max_rows = MC, max_cols = MC, size = max_cols * max_rows;
  int *referrence_cuda, *matrix_cuda;
  cudaMalloc((void **)&referrence_cuda, sizeof(int) * size);
  cudaMalloc((void **)&matrix_cuda, sizeof(int) * size);
  cudaMemcpy(referrence_cuda, referrence, sizeof(int) * size, cudaMemcpyHostToDevice);
  cudaMemcpy(matrix_cuda, input_itemsets, sizeof(int) * size, cudaMemcpyHostToDevice);
  dim3 dimGrid, dimBlock(BLOCK_SIZE, 1);
  int block_width = (max_cols - 1) / BLOCK_SIZE;
  unsigned long c0 = cyc();
  for (int i = 1; i <= block_width; i++) {
    dimGrid.x = i; dimGrid.y = 1;
    needle_cuda_shared_1<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda, max_cols, PEN, i, block_width);
  }
  for (int i = block_width - 1; i >= 1; i--) {
    dimGrid.x = i; dimGrid.y = 1;
    needle_cuda_shared_2<<<dimGrid, dimBlock>>>(referrence_cuda, matrix_cuda, max_cols, PEN, i, block_width);
  }
  cudaDeviceSynchronize();
  unsigned long c1 = cyc(); REPORT("nw hwacha-cc", c0, c1, (long)N * N);
  cudaMemcpy(output_itemsets, matrix_cuda, sizeof(int) * size, cudaMemcpyDeviceToHost);
  int bad = 0; for (int i = 0; i < size; i++) if (output_itemsets[i] != rr[i]) bad++;
  printf("nw %s (%d mismatches / %d)\n", bad ? "FAIL" : "PASS", bad, size); return bad != 0;
}

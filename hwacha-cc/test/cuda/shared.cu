// __shared__ + __syncthreads (a block reduction), expf via clang's CUDA math header, and a 2-D launch
#include <cuda_runtime.h>
#include <stdio.h>

// one block per row: load the row into shared memory, tree-reduce exp(x) over the block
__global__ void rowsumexp(int n, const float *m, float *s) {
  __shared__ float buf[256];
  int t = threadIdx.x;
  buf[t] = expf(m[blockIdx.x * n + t]);
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (t < k) buf[t] += buf[t + k];
    __syncthreads();
  }
  if (t == 0) s[blockIdx.x] = buf[0];
}

// 2-D grid x 2-D block: c[i][j] = a[i][j] * 2 + i - j
__global__ void grid2d(int w, int h, const float *a, float *c) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < h && j < w) c[i * w + j] = 2.0f * a[i * w + j] + (float)(i - j);
}

#define R 8
#define C 256
static float hm[R * C], hs[R], ref[R];
static float h_expf(float x) { int k = (int)(x / 0.6931472f + (x < 0 ? -0.5f : 0.5f)); float r = x - k * 0.6931472f, s = 1, t = 1;
  for (int i = 1; i < 12; i++) { t *= r / i; s += t; } while (k > 0) { s *= 2; k--; } while (k < 0) { s *= 0.5f; k++; } return s; }
static float h_fabsf(float v) { return v < 0 ? -v : v; }

#define W 100
#define H 37
static float ha[W * H], hc[W * H];

int main() {
  int fail = 0;
  float *m, *s;
  for (int i = 0; i < R; i++) { ref[i] = 0; for (int j = 0; j < C; j++) { hm[i * C + j] = ((i * 7 + j) % 23) * 0.1f - 1; ref[i] += h_expf(hm[i * C + j]); } }
  cudaMalloc(&m, sizeof hm); cudaMalloc(&s, sizeof hs);
  cudaMemcpy(m, hm, sizeof hm, cudaMemcpyHostToDevice);
  rowsumexp<<<R, C>>>(C, m, s);
  cudaMemcpy(hs, s, sizeof hs, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < R; i++) if (h_fabsf(hs[i] - ref[i]) > 1e-4f * ref[i]) { if (bad < 3) printf("  s[%d] = %g want %g\n", i, hs[i], ref[i]); bad++; }
  printf("rowsumexp      %s\n", bad ? "FAIL" : "ok"); fail |= bad != 0;

  float *a, *c;
  for (int i = 0; i < W * H; i++) ha[i] = (i % 13) * 0.5f;
  cudaMalloc(&a, sizeof ha); cudaMalloc(&c, sizeof hc);
  cudaMemcpy(a, ha, sizeof ha, cudaMemcpyHostToDevice);
  cudaMemset(c, 0, sizeof hc);
  dim3 block(16, 8), grid((W + 15) / 16, (H + 7) / 8);
  grid2d<<<grid, block>>>(W, H, a, c);
  cudaMemcpy(hc, c, sizeof hc, cudaMemcpyDeviceToHost);
  bad = 0;
  for (int i = 0; i < H; i++) for (int j = 0; j < W; j++) { float want = 2.0f * ha[i * W + j] + (float)(i - j);
    if (hc[i * W + j] != want) { if (bad < 3) printf("  c[%d][%d] = %g want %g\n", i, j, hc[i * W + j], want); bad++; } }
  printf("grid2d         %s\n", bad ? "FAIL" : "ok"); fail |= bad != 0;
  return fail;
}

// The CUDA entry: clang -x cuda (device: --cuda-device-only -> hwacha-cc; host: --cuda-host-only -O0).
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void saxpy(int n, float a, const float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

#define N 4096
static float hx[N], hy[N], ref[N];

int main() {
  for (int i = 0; i < N; i++) { hx[i] = (i % 17) * 0.25f - 2; hy[i] = (i % 5) * 0.5f; ref[i] = i < 4000 ? 1.5f * hx[i] + hy[i] : hy[i]; }
  float *x, *y;
  cudaMalloc(&x, N * sizeof(float)); cudaMalloc(&y, N * sizeof(float));
  cudaMemcpy(x, hx, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(y, hy, N * sizeof(float), cudaMemcpyHostToDevice);
  saxpy<<<(4000 + 255) / 256, 256>>>(4000, 1.5f, x, y);
  cudaDeviceSynchronize();
  cudaMemcpy(hy, y, N * sizeof(float), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int i = 0; i < N; i++) if (hy[i] != ref[i]) { if (bad < 3) printf("  y[%d] = %g want %g\n", i, hy[i], ref[i]); bad++; }
  printf("saxpy          %s\n", bad ? "FAIL" : "ok");
  return bad != 0;
}

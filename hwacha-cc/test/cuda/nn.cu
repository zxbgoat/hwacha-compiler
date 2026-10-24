// Rodinia nn (nearestNeighbor_cuda.cu): the `euclid` kernel with its 2-D grid indexing
// (blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x), driven as the CUDA host does.
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "rodinia.h"

typedef struct latLong { float lat; float lng; } LatLong;

__global__ void euclid(LatLong *d_locations, float *d_distances, int numRecords, float lat, float lng) {
  int globalId = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x;
  LatLong *latLong = d_locations + globalId;
  if (globalId < numRecords) {
    float *dist = d_distances + globalId;
    *dist = (float)sqrt((lat - latLong->lat) * (lat - latLong->lat) + (lng - latLong->lng) * (lng - latLong->lng));
  }
}

#define N 2048
static LatLong loc[N]; static float dist[N], ref[N];

int main() {
  const float lat = 30.0f, lng = -90.0f;
  for (int i = 0; i < N; i++) { loc[i].lat = frand(0, 90); loc[i].lng = frand(-180, 180); }
  for (int i = 0; i < N; i++) ref[i] = sqrtf((lat - loc[i].lat) * (lat - loc[i].lat) + (lng - loc[i].lng) * (lng - loc[i].lng));

  LatLong *d_locations; float *d_distances;
  cudaMalloc((void **)&d_locations, sizeof(LatLong) * N);
  cudaMalloc((void **)&d_distances, sizeof(float) * N);
  cudaMemcpy(d_locations, loc, sizeof(LatLong) * N, cudaMemcpyHostToDevice);
  // the original: threadsPerBlock 256, a grid of gridX x gridY blocks when the record count exceeds maxGridX
  int threadsPerBlock = 256, maxGridX = 4;   // small maxGridX forces a 2-D grid (the original uses 65535)
  int totalBlocks = ceil((float)N / threadsPerBlock);
  int gridY = ceil((float)totalBlocks / maxGridX), gridX = ceil((float)totalBlocks / gridY);
  dim3 gridDim(gridX, gridY);
  unsigned long c0 = cyc();
  euclid<<<gridDim, threadsPerBlock>>>(d_locations, d_distances, N, lat, lng);
  cudaDeviceSynchronize();
  unsigned long c1 = cyc(); REPORT("nn hwacha-cc", c0, c1, N);
  cudaMemcpy(dist, d_distances, sizeof(float) * N, cudaMemcpyDeviceToHost);
  int bad = 0; for (int i = 0; i < N; i++) { float d = dist[i] - ref[i]; if (d < 0) d = -d; if (d > 1e-3f * (ref[i] + 1)) bad++; }
  printf("nn %s (%d mismatches / %d, grid %dx%d)\n", bad ? "FAIL" : "PASS", bad, N, gridX, gridY); return bad != 0;
}

// Rodinia kmeans (kmeans_cuda_kernel.cu, the non-texture path): invert_mapping (transpose) and
// kmeansPoint with the cluster centres in __constant__ memory (cudaMemcpyToSymbol).
#include <cuda_runtime.h>
#include <stdio.h>
#include "rodinia.h"

#define NP 1024
#define NF 8
#define NC 5
#define THREADS_PER_BLOCK 256
#define FLT_MAX 3.40282347e+38f

__constant__ float c_clusters[NC * NF];

__global__ void invert_mapping(float *input, float *output, int npoints, int nfeatures) {
  int point_id = threadIdx.x + blockDim.x * blockIdx.x;
  int i;
  if (point_id < npoints) {
    for (i = 0; i < nfeatures; i++) output[point_id + npoints * i] = input[point_id * nfeatures + i];
  }
}

__global__ void kmeansPoint(float *features, int nfeatures, int npoints, int nclusters, int *membership, float *clusters) {
  const unsigned int block_id = gridDim.x * blockIdx.y + blockIdx.x;
  const unsigned int point_id = block_id * blockDim.x * blockDim.y + threadIdx.x;
  int index = -1;
  if (point_id < npoints) {
    int i, j;
    float min_dist = FLT_MAX;
    float dist;
    for (i = 0; i < nclusters; i++) {
      int cluster_base_index = i * nfeatures;
      float ans = 0.0;
      for (j = 0; j < nfeatures; j++) {
        int addr = point_id + j * npoints;
        float diff = (features[addr] - c_clusters[cluster_base_index + j]);
        ans += diff * diff;
      }
      dist = ans;
      if (dist < min_dist) { min_dist = dist; index = i; }
    }
  }
  if (point_id < npoints) membership[point_id] = index;
}

static float feature[NP * NF], fswap[NP * NF], fref[NP * NF], clusters[NC * NF]; static int membership[NP], mref[NP];

int main() {
  for (int i = 0; i < NP * NF; i++) feature[i] = frand(0, 100);
  for (int i = 0; i < NC * NF; i++) clusters[i] = frand(0, 100);
  for (int p = 0; p < NP; p++) for (int f = 0; f < NF; f++) fref[f * NP + p] = feature[p * NF + f];
  for (int p = 0; p < NP; p++) { float best = FLT_MAX; int idx = -1;
    for (int c = 0; c < NC; c++) { float d = 0; for (int f = 0; f < NF; f++) { float t = fref[f * NP + p] - clusters[c * NF + f]; d += t * t; } if (d < best) { best = d; idx = c; } }
    mref[p] = idx; }

  // allocateMemory() / kmeansCuda() from kmeans_cuda.cu
  float *feature_d, *feature_flipped_d, *clusters_d; int *membership_d;
  cudaMalloc((void **)&feature_flipped_d, NP * NF * sizeof(float));
  cudaMemcpy(feature_flipped_d, feature, NP * NF * sizeof(float), cudaMemcpyHostToDevice);
  cudaMalloc((void **)&feature_d, NP * NF * sizeof(float));
  cudaMalloc((void **)&membership_d, NP * sizeof(int));
  cudaMalloc((void **)&clusters_d, NC * NF * sizeof(float));
  int num_blocks = NP / THREADS_PER_BLOCK; if (NP % THREADS_PER_BLOCK > 0) num_blocks++;
  int num_blocks_perdim = (int)ceil(sqrt((double)num_blocks));
  dim3 grid(num_blocks_perdim, num_blocks_perdim), threads(THREADS_PER_BLOCK, 1);
  unsigned long c0 = cyc();
  invert_mapping<<<num_blocks, THREADS_PER_BLOCK>>>(feature_flipped_d, feature_d, NP, NF);
  cudaDeviceSynchronize();
  unsigned long c1 = cyc(); REPORT("kmeans_swap hwacha-cc", c0, c1, NP);
  cudaMemcpy(fswap, feature_d, NP * NF * sizeof(float), cudaMemcpyDeviceToHost);
  int bad = 0; for (int i = 0; i < NP * NF; i++) if (fswap[i] != fref[i]) bad++;
  printf("kmeans_swap %s (%d mismatches)\n", bad ? "FAIL" : "PASS", bad);

  cudaMemcpy(clusters_d, clusters, NC * NF * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(c_clusters, clusters, NC * NF * sizeof(float), 0, cudaMemcpyHostToDevice);
  c0 = cyc();
  kmeansPoint<<<grid, threads>>>(feature_d, NF, NP, NC, membership_d, clusters_d);
  cudaDeviceSynchronize();
  c1 = cyc(); REPORT("kmeans_c hwacha-cc", c0, c1, NP);
  cudaMemcpy(membership, membership_d, NP * sizeof(int), cudaMemcpyDeviceToHost);
  int bad2 = 0; for (int i = 0; i < NP; i++) if (membership[i] != mref[i]) bad2++;
  printf("kmeans_c %s (%d mismatches / %d)\n", bad2 ? "FAIL" : "PASS", bad2, NP);
  return (bad || bad2) != 0;
}

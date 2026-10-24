// Rodinia bfs (bfs.cu + kernel.cu + kernel2.cu): two kernels per level, bool masks, a `bool *g_over`
// flag written by every work-item that found new frontier nodes, level loop on the host.
#include <cuda_runtime.h>
#include <stdio.h>
#include "rodinia.h"

#define MAX_THREADS_PER_BLOCK 512
struct Node { int starting; int no_of_edges; };

__global__ void Kernel(Node *g_graph_nodes, int *g_graph_edges, bool *g_graph_mask, bool *g_updating_graph_mask, bool *g_graph_visited, int *g_cost, int no_of_nodes) {
  int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
  if (tid < no_of_nodes && g_graph_mask[tid]) {
    g_graph_mask[tid] = false;
    for (int i = g_graph_nodes[tid].starting; i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting); i++) {
      int id = g_graph_edges[i];
      if (!g_graph_visited[id]) {
        g_cost[id] = g_cost[tid] + 1;
        g_updating_graph_mask[id] = true;
      }
    }
  }
}

__global__ void Kernel2(bool *g_graph_mask, bool *g_updating_graph_mask, bool *g_graph_visited, bool *g_over, int no_of_nodes) {
  int tid = blockIdx.x * MAX_THREADS_PER_BLOCK + threadIdx.x;
  if (tid < no_of_nodes && g_updating_graph_mask[tid]) {
    g_graph_mask[tid] = true;
    g_graph_visited[tid] = true;
    *g_over = true;
    g_updating_graph_mask[tid] = false;
  }
}

#define NN 2048
#define DEG 4
static Node h_graph_nodes[NN]; static int h_graph_edges[NN * DEG]; static bool h_graph_mask[NN], h_updating_graph_mask[NN], h_graph_visited[NN];
static int h_cost[NN], ref[NN];

int main() {
  for (int i = 0; i < NN; i++) { h_graph_nodes[i].starting = i * DEG; h_graph_nodes[i].no_of_edges = DEG; for (int j = 0; j < DEG; j++) h_graph_edges[i * DEG + j] = (j == 0) ? (i + 1) % NN : (int)(rnd() % NN); }
  for (int i = 0; i < NN; i++) ref[i] = -1;
  { static int q[NN]; int h = 0, t = 0; ref[0] = 0; q[t++] = 0;
    while (h < t) { int u = q[h++]; for (int j = 0; j < DEG; j++) { int v = h_graph_edges[u * DEG + j]; if (ref[v] < 0) { ref[v] = ref[u] + 1; q[t++] = v; } } } }

  // BFSGraph() from bfs.cu
  int num_of_blocks = 1, num_of_threads_per_block = NN;
  if (NN > MAX_THREADS_PER_BLOCK) { num_of_blocks = (int)ceil(NN / (double)MAX_THREADS_PER_BLOCK); num_of_threads_per_block = MAX_THREADS_PER_BLOCK; }
  for (int i = 0; i < NN; i++) { h_graph_mask[i] = false; h_updating_graph_mask[i] = false; h_graph_visited[i] = false; h_cost[i] = -1; }
  int source = 0; h_graph_mask[source] = true; h_graph_visited[source] = true; h_cost[source] = 0;
  Node *d_graph_nodes; int *d_graph_edges; bool *d_graph_mask, *d_updating_graph_mask, *d_graph_visited, *d_over; int *d_cost;
  cudaMalloc((void **)&d_graph_nodes, sizeof(Node) * NN);      cudaMemcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * NN, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_graph_edges, sizeof(int) * NN * DEG); cudaMemcpy(d_graph_edges, h_graph_edges, sizeof(int) * NN * DEG, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_graph_mask, sizeof(bool) * NN);       cudaMemcpy(d_graph_mask, h_graph_mask, sizeof(bool) * NN, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_updating_graph_mask, sizeof(bool) * NN); cudaMemcpy(d_updating_graph_mask, h_updating_graph_mask, sizeof(bool) * NN, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_graph_visited, sizeof(bool) * NN);    cudaMemcpy(d_graph_visited, h_graph_visited, sizeof(bool) * NN, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_cost, sizeof(int) * NN);              cudaMemcpy(d_cost, h_cost, sizeof(int) * NN, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_over, sizeof(bool));
  dim3 grid(num_of_blocks, 1, 1), threads(num_of_threads_per_block, 1, 1);
  int k = 0; bool stop;
  unsigned long c0 = cyc();
  do {
    stop = false;
    cudaMemcpy(d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice);
    Kernel<<<grid, threads, 0>>>(d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, NN);
    Kernel2<<<grid, threads, 0>>>(d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, NN);
    cudaMemcpy(&stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost);
    k++;
  } while (stop && k < 100);
  unsigned long c1 = cyc(); REPORT("bfs hwacha-cc", c0, c1, NN);
  cudaMemcpy(h_cost, d_cost, sizeof(int) * NN, cudaMemcpyDeviceToHost);
  int bad = 0; for (int i = 0; i < NN; i++) if (h_cost[i] != ref[i]) bad++;
  printf("bfs %s (%d mismatches / %d, %d levels)\n", bad ? "FAIL" : "PASS", bad, NN, k); return bad != 0;
}

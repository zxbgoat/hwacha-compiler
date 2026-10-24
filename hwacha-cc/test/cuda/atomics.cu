// CUDA atomics on Hwacha's vector AMOs (vamoadd/swap/and/or/xor/min/max .w/.d): every pattern a
// CUDA kernel uses them for, checked against a scalar reference.
//   hist_global   atomicAdd on 64 contended global bins
//   hist_shared   per-block __shared__ histogram (atomicAdd on __shared__) + atomicAdd flush
//   minmax        atomicMax / atomicMin / unsigned atomicMax, all lanes on ONE address (uniform pointer)
//   ticket        k = atomicAdd(&counter, 1); out[k] = i  -- the returned old values must be a permutation
//   bits          atomicOr builds a bitmap, atomicXor toggles, atomicAnd clears, atomicSub counts down
//   sum64         atomicAdd(unsigned long long*)  (vamoadd.d)
//   exch          atomicExch: last writer wins, the value in the slot must be one of the writers'
#include <cuda_runtime.h>
#include "rodinia.h"
#define N 4096
#define BINS 64
#define BLOCK 256

__global__ void hist_global(const int *x, int *h, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&h[x[i] & (BINS - 1)], 1);
}
__global__ void hist_shared(const int *x, int *h, int n) {
  __shared__ int s[BINS];
  int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
  if (t < BINS) s[t] = 0;
  __syncthreads();
  if (i < n) atomicAdd(&s[x[i] & (BINS - 1)], 1);
  __syncthreads();
  if (t < BINS) atomicAdd(&h[t], s[t]);
}
__global__ void minmax(const int *x, int *m, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicMax(&m[0], x[i]);
    atomicMin(&m[1], x[i]);
    atomicMax((unsigned *)&m[2], (unsigned)x[i]);
    atomicMin((unsigned *)&m[3], (unsigned)x[i]);
  }
}
__global__ void ticket(int *counter, int *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && (i & 3) != 1) {   // a quarter of the lanes stay out: the predicate must gate the AMO
    int k = atomicAdd(counter, 1);
    out[k] = i;
  }
}
__global__ void bits(const int *x, int *bitmap, int *flags, int *down, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicOr(&bitmap[x[i] >> 5], 1 << (x[i] & 31));          // set bit x[i]
    atomicXor(&flags[0], 1);                                   // n toggles
    if (i < 32) atomicAnd(&flags[1], ~(1 << i));              // clear the low 32 bits of a -1 word
    atomicSub(down, x[i] & 7);
    __threadfence();
  }
}
__global__ void sum64(const int *x, unsigned long long *s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(s, (unsigned long long)x[i] * 1000003ull);
}
__global__ void exch(int *slot, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) { int old = atomicExch(&slot[i & 7], i); if (old != -1 && (old & 7) != (i & 7)) slot[8] = 1; }   // every old value came from this slot's writers
}

static int hx[N], hh[BINS], href[BINS], hm[4], hout[N], hbitmap[N / 32], hflags[2], hslot[9];
int main(void) {
  int fail = 0;
  for (int i = 0; i < N; i++) hx[i] = (int)(rnd() % N);
  for (int i = 0; i < N; i++) href[hx[i] & (BINS - 1)]++;
  int *dx, *dh, *dm, *dcnt, *dout, *dbitmap, *dflags, *ddown, *dslot; unsigned long long *ds;
  cudaMalloc((void **)&dx, sizeof hx); cudaMalloc((void **)&dh, sizeof hh); cudaMalloc((void **)&dm, sizeof hm);
  cudaMalloc((void **)&dcnt, 4); cudaMalloc((void **)&dout, sizeof hout); cudaMalloc((void **)&dbitmap, sizeof hbitmap);
  cudaMalloc((void **)&dflags, sizeof hflags); cudaMalloc((void **)&ddown, 4); cudaMalloc((void **)&dslot, sizeof hslot); cudaMalloc((void **)&ds, 8);
  cudaMemcpy(dx, hx, sizeof hx, cudaMemcpyHostToDevice);
  dim3 grid((N + BLOCK - 1) / BLOCK), block(BLOCK);

  // hist_global / hist_shared
  for (int v = 0; v < 2; v++) {
    cudaMemset(dh, 0, sizeof hh);
    unsigned long c0 = cyc();
    if (v == 0) hist_global<<<grid, block>>>(dx, dh, N); else hist_shared<<<grid, block>>>(dx, dh, N);
    cudaDeviceSynchronize();
    unsigned long c1 = cyc();
    cudaMemcpy(hh, dh, sizeof hh, cudaMemcpyDeviceToHost);
    int bad = 0; for (int b = 0; b < BINS; b++) if (hh[b] != href[b]) { if (bad < 3) printf("  bin %d = %d want %d\n", b, hh[b], href[b]); bad++; }
    printf("%-12s %s (%lu cycles)\n", v == 0 ? "hist_global" : "hist_shared", bad ? "FAIL" : "ok", c1 - c0); fail |= bad != 0;
  }
  // minmax: one address per AMO, every lane
  { int init[4] = {-(1 << 30), 1 << 30, 0, -1}; cudaMemcpy(dm, init, sizeof init, cudaMemcpyHostToDevice);
    minmax<<<grid, block>>>(dx, dm, N); cudaMemcpy(hm, dm, sizeof hm, cudaMemcpyDeviceToHost);
    int mx = hx[0], mn = hx[0]; for (int i = 1; i < N; i++) { if (hx[i] > mx) mx = hx[i]; if (hx[i] < mn) mn = hx[i]; }
    int bad = hm[0] != mx || hm[1] != mn || hm[2] != mx || hm[3] != mn;
    if (bad) printf("  got %d %d %d %d want %d %d\n", hm[0], hm[1], hm[2], hm[3], mx, mn);
    printf("%-12s %s\n", "minmax", bad ? "FAIL" : "ok"); fail |= bad; }
  // ticket: returned old values are distinct slots
  { cudaMemset(dcnt, 0, 4); cudaMemset(dout, 0xff, sizeof hout);
    ticket<<<grid, block>>>(dcnt, dout, N); int cnt; cudaMemcpy(&cnt, dcnt, 4, cudaMemcpyDeviceToHost); cudaMemcpy(hout, dout, sizeof hout, cudaMemcpyDeviceToHost);
    static char seen[N]; int bad = cnt != N * 3 / 4;
    for (int k = 0; k < cnt && k < N; k++) { int i = hout[k]; if (i < 0 || i >= N || (i & 3) == 1 || seen[i]) bad++; else seen[i] = 1; }
    for (int k = cnt; k < N; k++) if (hout[k] != -1) bad++;
    if (bad) printf("  counter %d (want %d), %d bad slots\n", cnt, N * 3 / 4, bad);
    printf("%-12s %s\n", "ticket", bad ? "FAIL" : "ok"); fail |= bad != 0; }
  // bits
  { int f0[2] = {0, -1}, down0 = 1 << 20; cudaMemset(dbitmap, 0, sizeof hbitmap); cudaMemcpy(dflags, f0, 8, cudaMemcpyHostToDevice); cudaMemcpy(ddown, &down0, 4, cudaMemcpyHostToDevice);
    bits<<<grid, block>>>(dx, dbitmap, dflags, ddown, N);
    cudaMemcpy(hbitmap, dbitmap, sizeof hbitmap, cudaMemcpyDeviceToHost); cudaMemcpy(hflags, dflags, 8, cudaMemcpyDeviceToHost); int down; cudaMemcpy(&down, ddown, 4, cudaMemcpyDeviceToHost);
    static int rbm[N / 32]; int rdown = 1 << 20; for (int i = 0; i < N; i++) { rbm[hx[i] >> 5] |= 1 << (hx[i] & 31); rdown -= hx[i] & 7; }
    int bad = 0; for (int w = 0; w < N / 32; w++) if (hbitmap[w] != rbm[w]) bad++;
    if (hflags[0] != (N & 1)) bad++; if (hflags[1] != (int)0xffffffff00000000ll) bad++; if (down != rdown) bad++;
    if (bad) printf("  flags %x %x down %d (want %d)\n", hflags[0], hflags[1], down, rdown);
    printf("%-12s %s\n", "bits", bad ? "FAIL" : "ok"); fail |= bad != 0; }
  // sum64
  { cudaMemset(ds, 0, 8); sum64<<<grid, block>>>(dx, ds, N); unsigned long long s; cudaMemcpy(&s, ds, 8, cudaMemcpyDeviceToHost);
    unsigned long long r = 0; for (int i = 0; i < N; i++) r += (unsigned long long)hx[i] * 1000003ull;
    int bad = s != r; if (bad) printf("  got %llu want %llu\n", s, r);
    printf("%-12s %s\n", "sum64", bad ? "FAIL" : "ok"); fail |= bad; }
  // exch
  { cudaMemset(dslot, 0xff, 32); cudaMemset(dslot + 8, 0, 4); exch<<<grid, block>>>(dslot, N); cudaMemcpy(hslot, dslot, sizeof hslot, cudaMemcpyDeviceToHost);
    int bad = hslot[8]; for (int k = 0; k < 8; k++) if (hslot[k] < 0 || hslot[k] >= N || (hslot[k] & 7) != k) bad++;
    printf("%-12s %s\n", "exch", bad ? "FAIL" : "ok"); fail |= bad != 0; }
  printf("atomics %s\n", fail ? "FAIL" : "PASS");
  return fail;
}

// CUDA host runtime for hwacha-cc programs: one flat address space, so "device" memory is a bump
// allocator and cudaMemcpy is memcpy. Launch configurations are parked by __cudaPushCallConfiguration
// and consumed by the launch stub hwacha-cc rewrites (see lowerCUDALaunches in GPUAdapt.cpp).
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifndef HWCUDA_ARENA
#define HWCUDA_ARENA (1 << 24)
#endif
static char hwcuda_arena[HWCUDA_ARENA] __attribute__((aligned(64)));
static long hwcuda_top;

typedef struct { unsigned x, y, z; } dim3_t;
static struct { dim3_t grid, block; size_t shm; void *stream; } hwcuda_cfg;

int cudaMalloc(void **p, size_t n) {
  long a = (hwcuda_top + 63) & ~63L;
  if (a + (long)n > HWCUDA_ARENA) { printf("cudaMalloc: arena overflow (%ld + %ld)\n", a, (long)n); *p = 0; return 2; }
  hwcuda_top = a + n; *p = hwcuda_arena + a; return 0;
}
int cudaFree(void *p) { (void)p; return 0; }
int cudaMemcpy(void *d, const void *s, size_t n, int kind) { (void)kind; memcpy(d, s, n); return 0; }
int cudaMemset(void *d, int v, size_t n) { memset(d, v, n); return 0; }
int cudaDeviceSynchronize(void) { return 0; }
int cudaDeviceReset(void) { hwcuda_top = 0; return 0; }
int cudaGetLastError(void) { return 0; }
int cudaPeekAtLastError(void) { return 0; }
const char *cudaGetErrorString(int e) { return e ? "error" : "no error"; }

// events: rdcycle stamps (from the arena, bare metal has no malloc), elapsed in "ms" at an assumed
// 1 GHz so timings print as cycles / 1e6
static long rdcycle(void) { long c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
int cudaEventCreate(void **e) { return cudaMalloc(e, sizeof(long)); }
int cudaEventRecord(void *e, void *s) { (void)s; *(long *)e = rdcycle(); return 0; }
int cudaEventSynchronize(void *e) { (void)e; return 0; }
int cudaEventElapsedTime(float *ms, void *a, void *b) { *ms = (*(long *)b - *(long *)a) / 1e6f; return 0; }
int cudaEventDestroy(void *e) { (void)e; return 0; }

// dim3 is passed as two i64 (lp64d: a 12-byte struct of unsigned goes in integer registers)
unsigned __cudaPushCallConfiguration(uint64_t g0, uint64_t g1, uint64_t b0, uint64_t b1, size_t shm, void *stream) {
  hwcuda_cfg.grid.x = (unsigned)g0; hwcuda_cfg.grid.y = (unsigned)(g0 >> 32); hwcuda_cfg.grid.z = (unsigned)g1;
  hwcuda_cfg.block.x = (unsigned)b0; hwcuda_cfg.block.y = (unsigned)(b0 >> 32); hwcuda_cfg.block.z = (unsigned)b1;
  hwcuda_cfg.shm = shm; hwcuda_cfg.stream = stream;
  return 0;
}
int __cudaPopCallConfiguration(dim3_t *grid, dim3_t *block, size_t *shm, void *stream) {
  *grid = hwcuda_cfg.grid; *block = hwcuda_cfg.block; *shm = hwcuda_cfg.shm; *(void **)stream = hwcuda_cfg.stream;
  return 0;
}

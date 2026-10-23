// CUDA runtime surface for hwacha-cc: used in place of NVIDIA's headers (clang -nocudainc -nocudalib).
// Device side: clang's own CUDA headers give threadIdx/blockIdx/..., __syncthreads and the math
// functions (expf -> __nv_expf, which hwacha-cc maps back to libm names / inlines).
// Host side: the functions below are implemented by hwcuda_rt.c (bump-allocated "device" memory
// in the one flat address space; launches are rewritten by hwacha-cc into <kernel>_ct calls).
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>   // the host-side pass also parses the kernels: expf etc. must be declared

#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

#include <__clang_cuda_builtin_vars.h>

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1) : x(x), y(y), z(z) {}
};
typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 11 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };

extern "C" {
cudaError_t cudaMalloc(void **p, size_t n);
cudaError_t cudaFree(void *p);
cudaError_t cudaMemcpy(void *d, const void *s, size_t n, cudaMemcpyKind kind);
cudaError_t cudaMemset(void *d, int v, size_t n);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char *cudaGetErrorString(cudaError_t e);
cudaError_t cudaEventCreate(cudaEvent_t *e);
cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t s);
cudaError_t cudaEventSynchronize(cudaEvent_t e);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t a, cudaEvent_t b);
cudaError_t cudaEventDestroy(cudaEvent_t e);
// launch protocol clang emits for <<<>>> (the stub it calls is replaced by hwacha-cc)
unsigned __cudaPushCallConfiguration(dim3 grid, dim3 block, size_t shm = 0, cudaStream_t s = 0);
cudaError_t __cudaPopCallConfiguration(dim3 *grid, dim3 *block, size_t *shm, void *s);
cudaError_t cudaLaunchKernel(const void *fn, dim3 grid, dim3 block, void **args, size_t shm, cudaStream_t s);
}
template <class T> static inline cudaError_t cudaMalloc(T **p, size_t n) { return cudaMalloc((void **)p, n); }
static inline cudaError_t cudaEventRecord(cudaEvent_t e) { return cudaEventRecord(e, 0); }

// clang's device-side math / intrinsic wrappers (__device__ expf -> __nv_expf, __syncthreads, ...).
// The host pass parses the kernels too, so it needs the same declarations: pretend to be a device
// there, as clang's own runtime wrapper does.
#define CUDA_VERSION 12000
#pragma push_macro("__CUDA_ARCH__")
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 9999
#endif
#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>
#pragma pop_macro("__CUDA_ARCH__")

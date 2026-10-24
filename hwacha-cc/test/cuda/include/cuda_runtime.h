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
cudaError_t cudaMemcpyToSymbol(const void *sym, const void *s, size_t n, size_t off = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice);
cudaError_t cudaMemcpyFromSymbol(void *d, const void *sym, size_t n, size_t off = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost);
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
#define __CLANG_CUDA_INTRINSICS_H__   // its __ldg / shuffles need NVIDIA vector types; warp primitives have no Hwacha equivalent anyway
// the public atomic names are NVIDIA's device_atomic_functions.hpp wrappers of clang's __*Atomic* builtins
#define __HW_ATOMIC(NAME, T, IMPL) static __device__ __forceinline__ T NAME(T *p, T v) { return IMPL(p, v); }
__HW_ATOMIC(atomicAdd, int, __iAtomicAdd)  __HW_ATOMIC(atomicAdd, unsigned, __uAtomicAdd)  __HW_ATOMIC(atomicAdd, unsigned long long, __ullAtomicAdd)  __HW_ATOMIC(atomicAdd, float, __fAtomicAdd)
#undef __HW_ATOMIC
static __device__ __forceinline__ int atomicSub(int *p, int v) { return __iAtomicAdd(p, -v); }
static __device__ __forceinline__ unsigned atomicSub(unsigned *p, unsigned v) { return __uAtomicAdd(p, 0u - v); }
static __device__ __forceinline__ int atomicExch(int *p, int v) { return __iAtomicExch(p, v); }
static __device__ __forceinline__ unsigned atomicExch(unsigned *p, unsigned v) { return __uAtomicExch(p, v); }
static __device__ __forceinline__ float atomicExch(float *p, float v) { return __fAtomicExch(p, v); }
static __device__ __forceinline__ int atomicMin(int *p, int v) { return __iAtomicMin(p, v); }
static __device__ __forceinline__ unsigned atomicMin(unsigned *p, unsigned v) { return __uAtomicMin(p, v); }
static __device__ __forceinline__ int atomicMax(int *p, int v) { return __iAtomicMax(p, v); }
static __device__ __forceinline__ unsigned atomicMax(unsigned *p, unsigned v) { return __uAtomicMax(p, v); }
static __device__ __forceinline__ unsigned atomicInc(unsigned *p, unsigned v) { return __uAtomicInc(p, v); }
static __device__ __forceinline__ unsigned atomicDec(unsigned *p, unsigned v) { return __uAtomicDec(p, v); }
static __device__ __forceinline__ int atomicCAS(int *p, int c, int v) { return __iAtomicCAS(p, c, v); }
static __device__ __forceinline__ unsigned atomicCAS(unsigned *p, unsigned c, unsigned v) { return __uAtomicCAS(p, c, v); }
static __device__ __forceinline__ int atomicAnd(int *p, int v) { return __iAtomicAnd(p, v); }
static __device__ __forceinline__ int atomicOr(int *p, int v) { return __iAtomicOr(p, v); }
static __device__ __forceinline__ int atomicXor(int *p, int v) { return __iAtomicXor(p, v); }
#pragma pop_macro("__CUDA_ARCH__")

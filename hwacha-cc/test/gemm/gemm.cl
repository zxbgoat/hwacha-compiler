// C[i][k] = sum_j A[i][j] * B[j][k], row-major n x n. One launch per row i (the host loops over i,
// like the Berkeley sgemm code); lanes span the columns k, the j loop runs on the control thread
// with B's row j as a unit-stride stream and A[i][j] as a uniform value.
__kernel void gemm_row(__global float *C, __global const float *A, __global const float *B, int n, int i) {
  int k = get_global_id(0);
  float acc = 0.0f;
  for (int j = 0; j < n; j++) acc += A[i * n + j] * B[j * n + k];
  C[i * n + k] = acc;
}
// whole product in one launch: lanes span all n*n outputs (i = gid / n): A becomes a gather, B a gather
__kernel void gemm_flat(__global float *C, __global const float *A, __global const float *B, int n) {
  int gid = get_global_id(0);
  int i = gid / n, k = gid - i * n;
  float acc = 0.0f;
  for (int j = 0; j < n; j++) acc += A[i * n + j] * B[j * n + k];
  C[gid] = acc;
}

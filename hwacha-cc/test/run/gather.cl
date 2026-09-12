__kernel void gather(__global const int *idx, __global const float *tab, __global float *out) {
  int i = get_global_id(0);
  out[i] = tab[idx[i]];
}
__kernel void scatter_add(__global const int *idx, __global const float *v, __global float *out) {
  int i = get_global_id(0);
  out[idx[i]] = v[i] * 2.0f;          // scatter (indices are a permutation, so no conflicts)
}
__kernel void mixed(int m, __global const float *x, __global const float *tab, __global float *out) {
  int i = get_global_id(0);
  float t = tab[m];                    // uniform scalar load inside the kernel
  out[i] = (x[i] > t) ? x[i] - t : t - x[i];
}

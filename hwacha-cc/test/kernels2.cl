// harder cases for the analysis
__kernel void cond_store(__global const int *x, __global int *y) {
  int i = get_global_id(0);
  if (x[i] > 0) y[i] = x[i] * 2;          // store under a divergent condition
}
__kernel void divloop(__global const int *n, __global int *out) {
  int i = get_global_id(0);
  int s = 0;
  for (int k = 0; k < n[i]; k++) s += k;  // divergent trip count
  out[i] = s;
}
__kernel void gather(__global const int *idx, __global const float *tab, __global float *out) {
  int i = get_global_id(0);
  out[i] = tab[idx[i]];                   // gather
}
__kernel void uniform_loop(int m, __global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = 0;
  for (int k = 0; k < m; k++) s += x[i + k];   // uniform trip count, sliding window
  out[i] = s;
}
__kernel void divloop2(__global const int *n, __global const float *x, __global float *out) {
  int i = get_global_id(0);
  float s = 0;
  for (int k = 0; k < n[i]; k++) s += x[k];   // divergent trip count, loads inside: no closed form
  out[i] = s;
}
__kernel void strided(int ld, __global const float *a, __global float *out) {
  int i = get_global_id(0);
  out[i] = a[i * ld];                          // runtime (uniform) stride
}

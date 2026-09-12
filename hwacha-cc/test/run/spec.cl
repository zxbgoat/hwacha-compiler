// The three OpenCL kernels that kernels_hwacha.S translates by hand.
// Work-item i == vector element; a work-group == one stripmine iteration.

// K1: unit-stride streams + one uniform scalar (broadcast via vs register)
__kernel void saxpy(float a, __global const float *x, __global float *y) {
  int i = get_global_id(0);
  y[i] = a * x[i] + y[i];
}

// K2: divergent control flow -> predication (no branch in the vf block)
__kernel void clamp_scale(float a, __global const float *x, __global float *y) {
  int i = get_global_id(0);
  float v = x[i];
  if (v > 0.0f) y[i] = a * v;
  else          y[i] = 0.0f;
}

// K3: global id used as data -> veidx + per-stripmine offset in a vs register
__kernel void iota(long base, __global long *c) {
  int i = get_global_id(0);
  c[i] = base + 2 * (long)i;
}

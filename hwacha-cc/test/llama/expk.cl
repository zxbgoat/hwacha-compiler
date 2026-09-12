#include "hwacha_math.h"
__kernel void expk(__global const float *in, __global float *out) {
  int i = get_global_id(0);
  out[i] = hw_expf(in[i]);
}
__kernel void siluk(__global float *hb, __global const float *hb2) {
  int i = get_global_id(0);
  float v = hb[i];
  v *= 1.0f / (1.0f + hw_expf(-v));
  hb[i] = v * hb2[i];
}

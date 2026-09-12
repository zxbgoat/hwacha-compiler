// Vectorizable math for hwacha-cc kernels: plain OpenCL C that lowers to vf-block instructions
// (fmadd, cvt, shifts, as_int/as_float bitcasts). No libm calls, no divergence.
#ifndef HWACHA_MATH_H
#define HWACHA_MATH_H
// expf: x = k*ln2 + r, |r| <= ln2/2 (k = round(x*log2e) via the 1.5*2^23 magic add, RNE),
// degree-6 polynomial (Cephes coefficients), scale by 2^k by adding k to the exponent field.
static inline float hw_expf(float x) {
  x = fmin(fmax(x, -87.0f), 88.0f);
  float t = x * 1.44269504088896341f + 12582912.0f;
  int k = as_int(t) - 0x4B400000;
  float kf = t - 12582912.0f;
  float r = x - kf * 0.693145751953125f;
  r = r - kf * 1.428606765330187e-06f;
  float p = 1.9875691500E-4f;
  p = p * r + 1.3981999507E-3f;
  p = p * r + 8.3334519073E-3f;
  p = p * r + 4.1665795894E-2f;
  p = p * r + 1.6666665459E-1f;
  p = p * r + 5.0000001201E-1f;
  p = p * (r * r) + r + 1.0f;
  return as_float(as_int(p) + (k << 23));
}
#endif

// Three MNIST diffusion models on Hwacha (bot66/MNISTDiffusion, TeaPearce/Conditional_Diffusion_MNIST,
// aestuans/mnist-diffusion): the union of their ops as OpenCL kernels.
//
// Layout (as in test/diffusion): every activation is a zero-padded plane [C][Hp*Wp], Hp = H+2, Wp = W+2,
// with a guard of 2*Wp+8 floats before and after the buffer. Stride-1 convolutions compute every padded
// position with shifted unit-stride streams; stride-2 ones gather. Border positions come out as garbage
// and are re-zeroed by the normalization / activation kernel that always follows.
#define OCB 8   // output channels per dense conv pass

// activations: 0 none, 1 SiLU, 2 GELU (erf), 3 ReLU
static inline float erf_f(float x) {   // Abramowitz-Stegun 7.1.26, |err| < 1.5e-7
  float s = x < 0.0f ? -1.0f : 1.0f, a = fabs(x);
  float t = 1.0f / (1.0f + 0.3275911f * a);
  float p = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
  return s * (1.0f - p * exp(-a * a));
}
static inline float act_f(float x, int act) {   // branch-free: all three evaluated, selected by act
  float silu = x / (1.0f + exp(-x));
  float gelu = 0.5f * x * (1.0f + erf_f(x * 0.70710678118654752f));
  float relu = fmax(x, 0.0f);
  float r = act == 1 ? silu : x;
  r = act == 2 ? gelu : r;
  return act == 3 ? relu : r;
}

// ---------------------------------------------------------------- dense convolutions
// 3x3, padding 1, stride 1: lane = padded output position, control-thread loops over ic and tap
__kernel void conv3x3(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                      int n_in, int plane, int Wp, int oc0) {
  int p = get_global_id(0);
  int ws = n_in * 9;
  float a0 = b[oc0], a1 = b[oc0 + 1], a2 = b[oc0 + 2], a3 = b[oc0 + 3], a4 = b[oc0 + 4], a5 = b[oc0 + 5], a6 = b[oc0 + 6], a7 = b[oc0 + 7];
  for (int ic = 0; ic < n_in; ic++) {
    __global const float *xs = x + ic * plane + p;
    __global const float *k = w + (oc0 * n_in + ic) * 9;
    for (int t = 0; t < 9; t++) {
      float v = xs[(t / 3 - 1) * Wp + (t % 3 - 1)];
      a0 += k[t] * v; a1 += k[ws + t] * v; a2 += k[2 * ws + t] * v; a3 += k[3 * ws + t] * v;
      a4 += k[4 * ws + t] * v; a5 += k[5 * ws + t] * v; a6 += k[6 * ws + t] * v; a7 += k[7 * ws + t] * v;
    }
  }
  y[(oc0 + 0) * plane + p] = a0; y[(oc0 + 1) * plane + p] = a1; y[(oc0 + 2) * plane + p] = a2; y[(oc0 + 3) * plane + p] = a3;
  y[(oc0 + 4) * plane + p] = a4; y[(oc0 + 5) * plane + p] = a5; y[(oc0 + 6) * plane + p] = a6; y[(oc0 + 7) * plane + p] = a7;
}
__kernel void conv3x3_1(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                        int n_in, int plane, int Wp, int oc) {
  int p = get_global_id(0);
  float a = b[oc];
  for (int ic = 0; ic < n_in; ic++) {
    __global const float *xs = x + ic * plane + p;
    __global const float *k = w + (oc * n_in + ic) * 9;
    a += k[0] * xs[-Wp - 1] + k[1] * xs[-Wp] + k[2] * xs[-Wp + 1] + k[3] * xs[-1] + k[4] * xs[0] + k[5] * xs[1] + k[6] * xs[Wp - 1] + k[7] * xs[Wp] + k[8] * xs[Wp + 1];
  }
  y[oc * plane + p] = a;
}
// 3x3, padding 1, stride 2: lane = padded output position of the half-size plane, gathered taps
__kernel void conv3x3_s2(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                         int n_in, int plane, int Wp, int plane2, int Wp2, int oc0) {
  int p = get_global_id(0);
  int i = p / Wp2, j = p - i * Wp2;
  int base = (2 * i - 2) * Wp + (2 * j - 2);
  int ws = n_in * 9;
  float a0 = b[oc0], a1 = b[oc0 + 1], a2 = b[oc0 + 2], a3 = b[oc0 + 3];
  for (int ic = 0; ic < n_in; ic++) {
    __global const float *xs = x + ic * plane + base;
    __global const float *k = w + (oc0 * n_in + ic) * 9;
    for (int t = 0; t < 9; t++) {
      float v = xs[(t / 3) * Wp + (t % 3)];
      a0 += k[t] * v; a1 += k[ws + t] * v; a2 += k[2 * ws + t] * v; a3 += k[3 * ws + t] * v;
    }
  }
  y[(oc0 + 0) * plane2 + p] = a0; y[(oc0 + 1) * plane2 + p] = a1; y[(oc0 + 2) * plane2 + p] = a2; y[(oc0 + 3) * plane2 + p] = a3;
}
// 1x1 convolutions
__kernel void conv1x1(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                      int n_in, int plane, int oc0) {
  int p = get_global_id(0);
  float a0 = b[oc0], a1 = b[oc0 + 1], a2 = b[oc0 + 2], a3 = b[oc0 + 3], a4 = b[oc0 + 4], a5 = b[oc0 + 5], a6 = b[oc0 + 6], a7 = b[oc0 + 7];
  for (int ic = 0; ic < n_in; ic++) {
    float v = x[ic * plane + p];
    __global const float *k = w + oc0 * n_in + ic;
    a0 += k[0] * v; a1 += k[n_in] * v; a2 += k[2 * n_in] * v; a3 += k[3 * n_in] * v;
    a4 += k[4 * n_in] * v; a5 += k[5 * n_in] * v; a6 += k[6 * n_in] * v; a7 += k[7 * n_in] * v;
  }
  y[(oc0 + 0) * plane + p] = a0; y[(oc0 + 1) * plane + p] = a1; y[(oc0 + 2) * plane + p] = a2; y[(oc0 + 3) * plane + p] = a3;
  y[(oc0 + 4) * plane + p] = a4; y[(oc0 + 5) * plane + p] = a5; y[(oc0 + 6) * plane + p] = a6; y[(oc0 + 7) * plane + p] = a7;
}
__kernel void conv1x1_1(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                        int n_in, int plane, int oc) {
  int p = get_global_id(0);
  float a = b[oc];
  for (int ic = 0; ic < n_in; ic++) a += w[oc * n_in + ic] * x[ic * plane + p];
  y[oc * plane + p] = a;
}
__kernel void conv1x1_s2(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                         int n_in, int plane, int Wp, int plane2, int Wp2, int oc0) {
  int p = get_global_id(0);
  int i = p / Wp2, j = p - i * Wp2;
  int src = (2 * i - 1) * Wp + (2 * j - 1);
  float a0 = b[oc0], a1 = b[oc0 + 1], a2 = b[oc0 + 2], a3 = b[oc0 + 3];
  for (int ic = 0; ic < n_in; ic++) {
    float v = x[ic * plane + src];
    __global const float *k = w + oc0 * n_in + ic;
    a0 += k[0] * v; a1 += k[n_in] * v; a2 += k[2 * n_in] * v; a3 += k[3 * n_in] * v;
  }
  y[(oc0 + 0) * plane2 + p] = a0; y[(oc0 + 1) * plane2 + p] = a1; y[(oc0 + 2) * plane2 + p] = a2; y[(oc0 + 3) * plane2 + p] = a3;
}

// ---------------------------------------------------------------- depthwise 3x3 (lane = position, loop over channels)
__kernel void dwconv3x3(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                        int n_ch, int plane, int Wp) {
  int p = get_global_id(0);
  for (int c = 0; c < n_ch; c++) {
    __global const float *xs = x + c * plane + p;
    __global const float *k = w + c * 9;
    y[c * plane + p] = b[c] + k[0] * xs[-Wp - 1] + k[1] * xs[-Wp] + k[2] * xs[-Wp + 1] + k[3] * xs[-1] + k[4] * xs[0] + k[5] * xs[1]
                     + k[6] * xs[Wp - 1] + k[7] * xs[Wp] + k[8] * xs[Wp + 1];
  }
}
__kernel void dwconv3x3_s2(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                           int n_ch, int plane, int Wp, int plane2, int Wp2) {
  int p = get_global_id(0);
  int i = p / Wp2, j = p - i * Wp2;
  int base = (2 * i - 2) * Wp + (2 * j - 2);
  for (int c = 0; c < n_ch; c++) {
    __global const float *xs = x + c * plane + base;
    __global const float *k = w + c * 9;
    y[c * plane2 + p] = b[c] + k[0] * xs[0] + k[1] * xs[1] + k[2] * xs[2] + k[3] * xs[Wp] + k[4] * xs[Wp + 1] + k[5] * xs[Wp + 2]
                      + k[6] * xs[2 * Wp] + k[7] * xs[2 * Wp + 1] + k[8] * xs[2 * Wp + 2];
  }
}

// ---------------------------------------------------------------- per-channel affine + activation (batch norm in eval mode,
// time / context embedding injection), interior only, border set to 0
__kernel void bn_act(__global float *x, int plane, int Wp, int H, int W, __global const float *scale, __global const float *shift, int act) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = act_f(x[p] * scale[c] + shift[c], act);
  x[p] = v;
}
// same with a residual: act(x*scale + shift + res)
__kernel void bn_add_act(__global float *x, int plane, int Wp, int H, int W, __global const float *scale, __global const float *shift,
                         __global const float *res, int act) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = act_f(x[p] * scale[c] + shift[c] + res[p], act);
  x[p] = v;
}
// group norm statistics: per-channel interior sum, then sum of squared deviations from the group mean
__kernel void chan_sum(__global const float *x, int plane, int Wp, int H, int W, __global float *out) {
  int c = get_global_id(0);
  float s = 0.0f;
  for (int i = 1; i <= H; i++)
    for (int j = 1; j <= W; j++) s += x[c * plane + i * Wp + j];
  out[c] = s;
}
__kernel void chan_sqdiff(__global const float *x, int plane, int Wp, int H, int W, __global const float *mean, int cpg, __global float *out) {
  int c = get_global_id(0);
  float m = mean[c / cpg], s = 0.0f;
  for (int i = 1; i <= H; i++)
    for (int j = 1; j <= W; j++) { float d = x[c * plane + i * Wp + j] - m; s += d * d; }
  out[c] = s;
}
// group norm apply (affine gamma / beta per channel) + ReLU
__kernel void gn_relu(__global float *x, int plane, int Wp, int H, int W, __global const float *mean, __global const float *rstd, int cpg,
                      __global const float *gamma, __global const float *beta) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp, g = c / cpg;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = fmax((x[p] - mean[g]) * rstd[g] * gamma[c] + beta[c], 0.0f);
  x[p] = v;
}

// ---------------------------------------------------------------- pooling / resampling
__kernel void maxpool2(__global const float *x, int plane, int Wp, int H, int W, __global float *y, int plane2, int Wp2) {
  int p = get_global_id(0);
  int c = p / plane2, q = p - c * plane2, i = q / Wp2, j = q - i * Wp2;
  float v = 0.0f;
  if (i >= 1 && i <= H / 2 && j >= 1 && j <= W / 2) {
    __global const float *s = x + c * plane + (2 * i - 1) * Wp + (2 * j - 1);
    v = fmax(fmax(s[0], s[1]), fmax(s[Wp], s[Wp + 1]));
  }
  y[p] = v;
}
// KxK average pool of a KxK plane to a vector, then GELU (lane = channel)
__kernel void avgpool_gelu(__global const float *x, int plane, int Wp, int K, __global float *v) {
  int c = get_global_id(0);
  float s = 0.0f;
  for (int i = 1; i <= K; i++)
    for (int j = 1; j <= K; j++) s += x[c * plane + i * Wp + j];
  v[c] = act_f(s / (float)(K * K), 2);
}
// ConvTranspose2d(k=K, s=K) of a 1x1 input = matrix-vector product written as a padded KxK plane
__kernel void up_gemv(__global const float *v, __global const float *w, __global const float *b, __global float *y,
                      int n_in, int n_out, int plane2, int Wp2, int K) {
  int p = get_global_id(0);
  int oc = p / plane2, q = p - oc * plane2, i = q / Wp2, j = q - i * Wp2;
  float a = 0.0f;
  if (i >= 1 && i <= K && j >= 1 && j <= K) {
    int wi = oc * K * K + (i - 1) * K + (j - 1);
    a = b[oc];
    for (int ic = 0; ic < n_in; ic++) a += v[ic] * w[ic * n_out * K * K + wi];
  }
  y[p] = a;
}
// ConvTranspose2d(k=2, s=2): lane = input pixel, 4 output channels x 4 taps, scattered into the pre-zeroed output
__kernel void convT2(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                     int n_in, int n_out, int plane, int Wp, int W, int plane2, int Wp2, int oc0) {
  int p = get_global_id(0);
  int i = p / W, j = p - i * W;
  int xi = (i + 1) * Wp + (j + 1);
  int yi = (2 * i + 1) * Wp2 + (2 * j + 1);
  float a00 = 0, a01 = 0, a02 = 0, a03 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0,
        a20 = 0, a21 = 0, a22 = 0, a23 = 0, a30 = 0, a31 = 0, a32 = 0, a33 = 0;
  for (int ic = 0; ic < n_in; ic++) {
    float v = x[ic * plane + xi];
    __global const float *k = w + (ic * n_out + oc0) * 4;
    a00 += k[0] * v; a01 += k[1] * v; a02 += k[2] * v; a03 += k[3] * v;
    a10 += k[4] * v; a11 += k[5] * v; a12 += k[6] * v; a13 += k[7] * v;
    a20 += k[8] * v; a21 += k[9] * v; a22 += k[10] * v; a23 += k[11] * v;
    a30 += k[12] * v; a31 += k[13] * v; a32 += k[14] * v; a33 += k[15] * v;
  }
  __global float *o0 = y + (oc0 + 0) * plane2 + yi, *o1 = y + (oc0 + 1) * plane2 + yi, *o2 = y + (oc0 + 2) * plane2 + yi, *o3 = y + (oc0 + 3) * plane2 + yi;
  float b0 = b[oc0], b1 = b[oc0 + 1], b2 = b[oc0 + 2], b3 = b[oc0 + 3];
  o0[0] = a00 + b0; o0[1] = a01 + b0; o0[Wp2] = a02 + b0; o0[Wp2 + 1] = a03 + b0;
  o1[0] = a10 + b1; o1[1] = a11 + b1; o1[Wp2] = a12 + b1; o1[Wp2 + 1] = a13 + b1;
  o2[0] = a20 + b2; o2[1] = a21 + b2; o2[Wp2] = a22 + b2; o2[Wp2 + 1] = a23 + b2;
  o3[0] = a30 + b3; o3[1] = a31 + b3; o3[Wp2] = a32 + b3; o3[Wp2 + 1] = a33 + b3;
}
// bilinear x2 upsample (align_corners = False): lane = padded output position of the double-size plane
__kernel void upsample2(__global const float *x, int plane, int Wp, int H, int W, __global float *y, int plane2, int Wp2) {
  int p = get_global_id(0);
  int c = p / plane2, q = p - c * plane2, i = q / Wp2, j = q - i * Wp2;
  float v = 0.0f;
  if (i >= 1 && i <= 2 * H && j >= 1 && j <= 2 * W) {
    int I = i - 1, J = j - 1;
    int i0 = I == 0 ? 0 : (I - 1) / 2, j0 = J == 0 ? 0 : (J - 1) / 2;
    float li = I == 0 ? 0.0f : ((I & 1) ? 0.25f : 0.75f), lj = J == 0 ? 0.0f : ((J & 1) ? 0.25f : 0.75f);
    int i1 = i0 + 1 < H ? i0 + 1 : H - 1, j1 = j0 + 1 < W ? j0 + 1 : W - 1;
    __global const float *s = x + c * plane;
    float r0 = s[(i0 + 1) * Wp + j0 + 1] * (1.0f - lj) + s[(i0 + 1) * Wp + j1 + 1] * lj;
    float r1 = s[(i1 + 1) * Wp + j0 + 1] * (1.0f - lj) + s[(i1 + 1) * Wp + j1 + 1] * lj;
    v = r0 * (1.0f - li) + r1 * li;
  }
  y[p] = v;
}

// ---------------------------------------------------------------- vectors and layout helpers
__kernel void fc(__global const float *x, __global const float *w, __global const float *b, __global float *y, int n_in, int act) {
  int o = get_global_id(0);
  float a = b[o];
  for (int i = 0; i < n_in; i++) a += x[i] * w[o * n_in + i];
  y[o] = act_f(a, act);
}
// channel shuffle of two groups: dst channel k <- src channel (k % 2) * C/2 + k / 2
__kernel void shuffle2(__global const float *src, __global float *dst, int C, int plane) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane;
  int sc = (c & 1) * (C / 2) + c / 2;
  dst[p] = src[sc * plane + q];
}
// every channel filled with one value on the interior (broadcast embeddings), border 0
__kernel void chan_fill(__global float *dst, int plane, int Wp, int H, int W, __global const float *vals) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp;
  dst[p] = (i >= 1 && i <= H && j >= 1 && j <= W) ? vals[c] : 0.0f;
}
__kernel void vcopy(__global const float *src, __global float *dst) { int p = get_global_id(0); dst[p] = src[p]; }
__kernel void vzero(__global float *dst) { int p = get_global_id(0); dst[p] = 0.0f; }

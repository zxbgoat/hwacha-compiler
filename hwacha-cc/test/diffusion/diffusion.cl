// diffusion.c (hku) ContextUnet on Hwacha: the ops of the 16x16 sprite DDPM as OpenCL kernels.
//
// Layout: every activation is a zero-padded plane [C][Hp*Wp] with Hp = H+2, Wp = W+2 and a guard of
// at least Wp+1 floats before and after the buffer. Convolutions compute every padded position with
// unit-stride shifted streams (no per-lane bounds checks); the border positions come out as garbage
// and are re-zeroed by the normalization / activation kernel that always follows a convolution.
#define OCB 8   // output channels per conv pass: 8 accumulators in vector registers, 8 weights per tap in scalar registers

// 3x3 convolution, padding 1: lane = padded output position, control-thread loops over input channel
// and tap; each tap is one unit-stride shifted stream feeding OCB multiply-adds
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

// one output channel (the 3-channel output layer)
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

// 1x1 convolution (residual shortcut), OCB output channels per pass
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

// per-channel sum over the interior (lane = channel, control-thread loop over pixels)
__kernel void chan_sum(__global const float *x, int plane, int Wp, int H, int W, __global float *out) {
  int c = get_global_id(0);
  float s = 0.0f;
  for (int i = 1; i <= H; i++)
    for (int j = 1; j <= W; j++) s += x[c * plane + i * Wp + j];
  out[c] = s;
}

// per-channel sum of squared deviations from the channel's (cpg = 1) or group's (cpg > 1) mean
__kernel void chan_sqdiff(__global const float *x, int plane, int Wp, int H, int W, __global const float *mean, int cpg, __global float *out) {
  int c = get_global_id(0);
  float m = mean[c / cpg], s = 0.0f;
  for (int i = 1; i <= H; i++)
    for (int j = 1; j <= W; j++) { float d = x[c * plane + i * Wp + j] - m; s += d * d; }
  out[c] = s;
}

static inline float gelu_f(float x) {
  float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);   // tanh via exp: 1 - 2/(1+e^2z)
  float t = 1.0f - 2.0f / (1.0f + exp(2.0f * z));
  return 0.5f * x * (1.0f + t);
}

// batch norm (per-channel mean / rstd) + GELU on the interior, border set to 0
__kernel void bn_gelu(__global float *x, int plane, int Wp, int H, int W, __global const float *mean, __global const float *rstd) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = gelu_f((x[p] - mean[c]) * rstd[c]);
  x[p] = v;
}

// same, then residual add and scale: (gelu(bn(x)) + res) / 1.414
__kernel void bn_gelu_res(__global float *x, int plane, int Wp, int H, int W, __global const float *mean, __global const float *rstd,
                          __global const float *res) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = (gelu_f((x[p] - mean[c]) * rstd[c]) + res[p]) / 1.414f;
  x[p] = v;
}

// group norm (mean / rstd per group of cpg channels) + ReLU on the interior, border set to 0
__kernel void gn_relu(__global float *x, int plane, int Wp, int H, int W, __global const float *mean, __global const float *rstd, int cpg) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp, g = c / cpg;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = fmax((x[p] - mean[g]) * rstd[g], 0.0f);
  x[p] = v;
}

// 2x2 max pool into a padded plane of half the size
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

// 4x4 average pool of a 4x4 plane to a vector, then GELU (lane = channel)
__kernel void avgpool4_gelu(__global const float *x, int plane, int Wp, __global float *v) {
  int c = get_global_id(0);
  float s = 0.0f;
  for (int i = 1; i <= 4; i++)
    for (int j = 1; j <= 4; j++) s += x[c * plane + i * Wp + j];
  v[c] = gelu_f(s * (1.0f / 16.0f));
}

// ConvTranspose2d(k=4, s=4) of a 1x1 input = matrix-vector product written as a padded 4x4 plane
__kernel void up0_gemv(__global const float *v, __global const float *w, __global const float *b, __global float *y,
                       int n_in, int n_out, int plane2, int Wp2) {
  int p = get_global_id(0);
  int oc = p / plane2, q = p - oc * plane2, i = q / Wp2, j = q - i * Wp2;
  float a = 0.0f;
  if (i >= 1 && i <= 4 && j >= 1 && j <= 4) {
    int wi = oc * 16 + (i - 1) * 4 + (j - 1);
    a = b[oc];
    for (int ic = 0; ic < n_in; ic++) a += v[ic] * w[ic * n_out * 16 + wi];
  }
  y[p] = a;
}

// ConvTranspose2d(k=2, s=2): lane = input pixel, 4 output channels x 4 taps accumulated, scattered
// into the (pre-zeroed) padded output plane of twice the size
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

// fully connected layer (time / context embeddings), optional GELU
__kernel void fc(__global const float *x, __global const float *w, __global const float *b, __global float *y, int n_in, int act) {
  int o = get_global_id(0);
  float a = b[o];
  for (int i = 0; i < n_in; i++) a += x[i] * w[o * n_in + i];
  y[o] = act ? gelu_f(a) : a;
}

// x = x * cemb[c] + temb[c] on the interior
__kernel void elm_linear(__global float *x, int plane, int Wp, int H, int W, __global const float *cemb, __global const float *temb) {
  int p = get_global_id(0);
  int c = p / plane, q = p - c * plane, i = q / Wp, j = q - i * Wp;
  float v = 0.0f;
  if (i >= 1 && i <= H && j >= 1 && j <= W) v = x[p] * cemb[c] + temb[c];
  x[p] = v;
}

__kernel void vcopy(__global const float *src, __global float *dst) { int p = get_global_id(0); dst[p] = src[p]; }
__kernel void vzero(__global float *dst) { int p = get_global_id(0); dst[p] = 0.0f; }

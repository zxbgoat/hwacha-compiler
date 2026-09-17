// Kernel library behind hwacha-mlir's convolution pattern (--conv-lib): the hand-written forms of
// test/mnist/mnist.cl for the convolutions torch-mlir emits (linalg.conv_2d_nchw_fchw on a zero-padded
// input), plus the copy from the padded scratch plane back to the dense output.
//   conv3x3    padded input plane (H+2)x(W+2) -> padded output planes, 8 output channels per pass
//   conv3x3_s2 padded input -> padded half-size output planes, 4 output channels per pass
//   conv1x1    dense input -> dense output, 8 output channels per pass (no scratch needed)
//   unpad      interior of the padded scratch planes -> dense NCHW output
#define OCB 8
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
  y[(oc0 + 0) * plane + p] += a0; y[(oc0 + 1) * plane + p] += a1; y[(oc0 + 2) * plane + p] += a2; y[(oc0 + 3) * plane + p] += a3;
  y[(oc0 + 4) * plane + p] += a4; y[(oc0 + 5) * plane + p] += a5; y[(oc0 + 6) * plane + p] += a6; y[(oc0 + 7) * plane + p] += a7;
}
__kernel void conv1x1_1(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                        int n_in, int plane, int oc) {
  int p = get_global_id(0);
  float a = b[oc];
  for (int ic = 0; ic < n_in; ic++) a += w[oc * n_in + ic] * x[ic * plane + p];
  y[oc * plane + p] += a;
}
// dense y[c][i][j] = padded x[c][(i+pad)*Wp + j+pad]; lane = dense index
__kernel void unpad(__global const float *x, __global float *y, int plane, int Wp, int H, int W, int pad) {
  int p = get_global_id(0);
  int HW = H * W;
  int c = p / HW, q = p - c * HW, i = q / W, j = q - i * W;
  y[p] += x[c * plane + (i + pad) * Wp + j + pad];
}

// general KxK convolution, stride 1, on a zero-padded input plane. lane = padded position; the K*K
// taps are a control-thread loop of shifted unit-stride streams, 8 output channels per pass. The output
// is written at the same padded coordinates (stride 1) into a plane of the input's shape; unpad(pad)
// extracts the dense interior.
__kernel void convKxK(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                       int n_in, int plane, int Wp, int K, int pad, int oc0) {
  int p = get_global_id(0);
  int KK = K * K, ws = n_in * KK;
  float a0 = b[oc0], a1 = b[oc0 + 1], a2 = b[oc0 + 2], a3 = b[oc0 + 3], a4 = b[oc0 + 4], a5 = b[oc0 + 5], a6 = b[oc0 + 6], a7 = b[oc0 + 7];
  for (int ic = 0; ic < n_in; ic++) {
    __global const float *xs = x + ic * plane + p;
    __global const float *k = w + (oc0 * n_in + ic) * KK;
    for (int t = 0; t < KK; t++) {
      float v = xs[(t / K - pad) * Wp + (t % K - pad)];
      a0 += k[t] * v; a1 += k[ws + t] * v; a2 += k[2 * ws + t] * v; a3 += k[3 * ws + t] * v;
      a4 += k[4 * ws + t] * v; a5 += k[5 * ws + t] * v; a6 += k[6 * ws + t] * v; a7 += k[7 * ws + t] * v;
    }
  }
  y[(oc0 + 0) * plane + p] = a0; y[(oc0 + 1) * plane + p] = a1; y[(oc0 + 2) * plane + p] = a2; y[(oc0 + 3) * plane + p] = a3;
  y[(oc0 + 4) * plane + p] = a4; y[(oc0 + 5) * plane + p] = a5; y[(oc0 + 6) * plane + p] = a6; y[(oc0 + 7) * plane + p] = a7;
}
__kernel void convKxK_1(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                        int n_in, int plane, int Wp, int K, int pad, int oc) {
  int p = get_global_id(0);
  int KK = K * K;
  float a = b[oc];
  for (int ic = 0; ic < n_in; ic++) {
    __global const float *xs = x + ic * plane + p;
    __global const float *k = w + (oc * n_in + ic) * KK;
    for (int t = 0; t < KK; t++) a += k[t] * xs[(t / K - pad) * Wp + (t % K - pad)];
  }
  y[oc * plane + p] = a;
}

// KxK max pool, stride S, dense NCHW in/out (N=1). lane = flat output index; the KxK window is gathered.
__kernel void poolmax(__global const float *x, __global float *y, int C, int Hi, int Wi, int Ho, int Wo, int K, int S) {
  int p = get_global_id(0);
  int HoWo = Ho * Wo;
  int c = p / HoWo, q = p - c * HoWo, i = q / Wo, j = q - i * Wo;
  __global const float *xs = x + c * Hi * Wi + (i * S) * Wi + j * S;
  float m = -1.0f / 0.0f;
  for (int kh = 0; kh < K; kh++)
    for (int kw = 0; kw < K; kw++) m = fmax(m, xs[kh * Wi + kw]);
  y[p] = m;
}

// depthwise KxK convolution, stride S, on a zero-padded input plane. lane = padded position; control
// loop over channels, KxK taps as shifted unit-stride streams. Output written at the same coords
// (stride 1) or half-size coords (stride 2) into a per-channel plane; unpad(pad) extracts the interior.
__kernel void dwconvKxK(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                        int C, int plane, int Wp, int K, int pad) {
  int p = get_global_id(0);
  int KK = K * K;
  for (int c = 0; c < C; c++) {
    __global const float *xs = x + c * plane + p;
    __global const float *k = w + c * KK;
    float a = b[c];
    for (int t = 0; t < KK; t++) a += k[t] * xs[(t / K - pad) * Wp + (t % K - pad)];
    y[c * plane + p] = a;
  }
}
__kernel void dwconvKxK_s2(__global const float *x, __global const float *w, __global const float *b, __global float *y,
                           int C, int plane, int Wp, int plane2, int Wp2, int K, int pad) {
  int p = get_global_id(0);
  int i = p / Wp2, j = p - i * Wp2;
  int base = (2 * i - pad) * Wp + (2 * j - pad);
  int KK = K * K;
  for (int c = 0; c < C; c++) {
    __global const float *xs = x + c * plane + base;
    __global const float *k = w + c * KK;
    float a = b[c];
    for (int t = 0; t < KK; t++) a += k[t] * xs[(t / K) * Wp + (t % K)];
    y[c * plane2 + p] = a;
  }
}

// spatial sum reduction (global pool numerator): dense y[c] = sum over H*W of x[c*HW + i]. lane = channel.
__kernel void chansum(__global const float *x, __global float *y, int C, int HW) {
  int c = get_global_id(0);
  float s = 0.0f;
  for (int i = 0; i < HW; i++) s += x[c * HW + i];
  y[c] = s;
}

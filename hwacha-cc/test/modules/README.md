# Basic nn layers on Hwacha

Each of PyTorch's basic `torch.nn` layers, taken on its own through the route-A pipeline
(PyTorch -> torch-mlir -> hwacha-mlir -> hwacha-cc -> Spike) and checked against PyTorch for one forward.

- `export_module.py <layer> <mlir> <check.bin>` — builds a tiny module for `<layer>`, exports it to
  linalg-on-tensors, and writes its input + PyTorch output.
- `mod_main.c` — one generic host: `net(float* x)`, compares against the reference with a relative tolerance.
- `make <layer>.riscv` builds one; `make run` builds and runs them all with a PASS/FAIL line each.

`HWMLIRFLAGS` defaults to `--collapse-all`: for these single-layer tests it maps every parallel dimension
to lanes, which vectorizes the pad / copy / reduction loops that the default lane-innermost mapping would
otherwise unroll and overflow the scalar (vs) registers on.

## Coverage (28 / 29 pass)

| group | layers |
|---|---|
| linear / MLP | linear, mlp |
| convolution | conv1x1, conv3x3, conv5x5, conv_s2 (stride 2), dwconv3x3 (depthwise), convtranspose2 |
| pooling | maxpool2, avgpool2, adaptiveavgpool |
| normalization | batchnorm2d, layernorm |
| activation | relu, leakyrelu, elu, gelu, silu, sigmoid, tanh, hardswish, hardsigmoid, softmax |
| shape / resample | flatten, upsample (nearest), upsample_bilinear, pixelshuffle |
| attention | attention (multi-head self-attention: qkv/proj linear + batch_matmul + softmax) |

All of the above match PyTorch to the harness tolerance on Spike.

## Known gap

- **groupnorm** — torch-mlir decomposes GroupNorm into an **f64** grouped reduction
  (`1xGx(C/G)x(HW)xf64`, reducing the last two dims per group). hwacha-cc miscomputes that f64 grouped
  reduction (output diverges badly), so it is excluded from `make run`. LayerNorm and BatchNorm (both f32)
  cover the normalization path; the f64 grouped reduction is the remaining fix.

Layers not attempted here: RNN/LSTM/GRU (recurrent), grouped conv with groups>1 and channels>1
(`conv_2d_ngchw_gfchw`, the regnet/resnext gap), and cross-attention / MMDiT-style two-stream attention.

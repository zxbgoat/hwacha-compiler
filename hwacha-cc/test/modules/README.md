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

## Coverage (51 / 52 pass)

nn-module layers and the extra `torch.nn.functional` ops, each on its own:

| group | layers |
|---|---|
| linear / MLP | linear, mlp |
| convolution | conv1x1, conv3x3, conv5x5, conv_s2 (stride 2), dwconv3x3 (depthwise), convtranspose2 |
| pooling | maxpool2, avgpool2, adaptiveavgpool |
| normalization | batchnorm2d, layernorm, groupnorm, instancenorm, lrn (local response norm), normalize (L2) |
| activation | relu, relu6, leakyrelu, prelu, elu, celu, selu, gelu, silu, sigmoid, tanh, softsign, softplus, mish, hardswish, hardsigmoid, hardtanh, tanhshrink, softshrink, hardshrink, threshold, logsigmoid |
| softmax family | softmax, logsoftmax, glu |
| shape / resample | flatten, upsample (nearest), upsample_bilinear, interpolate, pixelshuffle, pad (constant), unfold |
| attention | attention (manual qkv + batch_matmul + softmax), sdpa (F.scaled_dot_product_attention, fused) |

All of the above match PyTorch to the harness tolerance on Spike.

Compiler primitives these needed, inlined in hwacha-cc as vector arithmetic (see `src/Analysis.cpp`):
`expf`, `logf` (Cephes), `erff`, `tanhf`, `floorf`, and `log1pf`/`expm1f`/`powf` built on log+exp. i1
comparison masks are stored/loaded as 0/1 bytes (a predicate register cannot be vsb/vlb'd directly), and
a select whose two arms are both loads keeps them in distinct registers (the load emits under the block
predicate, not the select arm's, so it cannot be computed straight into the select register).

## Known gap

- **pad_reflect** — `F.pad(mode='reflect')` does not lower through torch-mlir (missing an LLVM lowering
  for the reflection index arithmetic); constant `pad` works.

Not attempted here: RNN/LSTM/GRU (recurrent), grouped conv with groups>1 and channels>1
(`conv_2d_ngchw_gfchw`, the regnet/resnext gap), and cross-attention / MMDiT-style two-stream attention.

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

## Coverage (89 / 92 pass)

Covers the `torch.nn` catalogue from docs.pytorch.org/docs/2.14/nn.html plus the extra
`torch.nn.functional` ops, each layer on its own:

| group | layers |
|---|---|
| linear / shape | linear, bilinear-free mlp, identity, flatten, unflatten |
| convolution | conv1d, conv3x3, conv1x1, conv5x5, conv_s2 (stride 2), dwconv3x3 (depthwise), convtranspose1d/2d/3d |
| pooling | maxpool1d/2d/3d, avgpool1d/2d/3d, lppool2d, adaptiveavgpool1d/2d/3d, adaptivemaxpool1d/2d |
| padding | pad (constant), zeropad1d/2d, constantpad2d, replicationpad2d, circularpad2d |
| normalization | batchnorm1d/2d/3d, layernorm, groupnorm, instancenorm1d/2d, rmsnorm, lrn, normalize (L2) |
| activation | relu, relu6, rrelu, leakyrelu, prelu, elu, celu, selu, gelu, silu, sigmoid, tanh, softsign, softplus, mish, hardswish, hardsigmoid, hardtanh, tanhshrink, softshrink, hardshrink, threshold, logsigmoid |
| softmax family | softmax, softmin, softmax2d, logsoftmax, glu |
| dropout (eval) | dropout, dropout2d, alphadropout |
| vision / shuffle | upsample (nearest), upsample_bilinear, interpolate, upsamplingnearest2d, upsamplingbilinear2d, pixelshuffle, pixelunshuffle, channelshuffle, unfold |
| recurrent | rnn, lstm, gru, rnncell |
| attention / transformer | attention (manual), sdpa (fused F.scaled_dot_product_attention), transformerencoderlayer |

All of the above match PyTorch to the harness tolerance on Spike.

Compiler primitives these needed, inlined in hwacha-cc as vector arithmetic (see `src/Analysis.cpp`):
`expf`, `logf` (Cephes), `erff`, `tanhf`, `floorf`, and `log1pf`/`expm1f`/`powf` built on log+exp. i1
comparison masks are stored/loaded as 0/1 bytes (a predicate register cannot be vsb/vlb'd directly), and
a select whose two arms are both loads keeps them in distinct registers (the load emits under the block
predicate, not the select arm's, so it cannot be computed straight into the select register).

## Known gaps (3)

- **conv3d** — 3D convolution has no library kernel yet, so it takes the generic path and overflows the
  scalar registers. conv1d and conv2d (all variants) work.
- **reflectionpad2d** / **pad_reflect** — reflection padding does not lower through torch-mlir (missing
  an LLVM lowering for the reflection index arithmetic); constant / replication / circular / zero padding
  all work.

Not attempted: Lazy* variants (need a materializing forward), loss functions (need a target), Embedding /
EmbeddingBag (integer input, incompatible with the float harness), MaxUnpool / FractionalMaxPool (need
indices or randomness), SyncBatchNorm and the distributed / container modules; grouped conv with groups>1
and channels>1 (`conv_2d_ngchw_gfchw`, the regnet/resnext gap).

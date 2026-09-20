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

## Coverage (all attempted layers pass)

Covers the `torch.nn` catalogue from docs.pytorch.org/docs/2.14/nn.html plus the extra
`torch.nn.functional` ops, each layer on its own:

| group | layers |
|---|---|
| linear / shape | linear, bilinear-free mlp, identity, flatten, unflatten |
| convolution | conv1d, conv3x3, conv1x1, conv5x5, conv3d, conv_s2 (stride 2), dwconv3x3 (depthwise), groupconv / groupconv_s2 (grouped), convtranspose1d/2d/3d |
| pooling | maxpool1d/2d/3d, avgpool1d/2d/3d, lppool1d/2d/3d, adaptiveavgpool1d/2d/3d, adaptivemaxpool1d/2d |
| padding | pad (constant), zeropad1d/2d, constantpad2d, replicationpad2d, circularpad2d, reflectionpad2d |
| normalization | batchnorm1d/2d/3d, lazybatchnorm2d, layernorm, groupnorm, instancenorm1d/2d/3d, rmsnorm, lrn, normalize (L2) |
| activation | relu, relu6, rrelu, leakyrelu, prelu, elu, celu, selu, gelu, silu, sigmoid, tanh, softsign, softplus, mish, hardswish, hardsigmoid, hardtanh, tanhshrink, softshrink, hardshrink, threshold, logsigmoid |
| softmax family | softmax, softmin, softmax2d, logsoftmax, glu |
| dropout (eval) | dropout, dropout2d, alphadropout |
| vision / shuffle | upsample (nearest), upsample_bilinear, interpolate, upsamplingnearest2d, upsamplingbilinear2d, pixelshuffle, pixelunshuffle, channelshuffle, unfold |
| recurrent | rnn, lstm, gru, rnncell |
| attention / transformer | attention (manual), multiheadattention (nn.MultiheadAttention), sdpa (fused F.scaled_dot_product_attention), transformerencoderlayer, transformerencoder, transformerdecoderlayer, transformerdecoder, transformer (full encoder-decoder) |

All of the above match PyTorch to the harness tolerance on Spike.

Compiler primitives these needed, inlined in hwacha-cc as vector arithmetic (see `src/Analysis.cpp`):
`expf`, `logf` (Cephes), `erff`, `tanhf`, `floorf`, and `log1pf`/`expm1f`/`powf` built on log+exp. i1
comparison masks are stored/loaded as 0/1 bytes (a predicate register cannot be vsb/vlb'd directly), and
a select whose two arms are both loads keeps them in distinct registers (the load emits under the block
predicate, not the select arm's, so it cannot be computed straight into the select register).

## Not attempted

Every layer in `make run` passes. Reflection padding needed two fixes to get there: hwacha-mlir expands
`math.absi` to arith (it has no LLVM translation), and hwacha-cc lowers the `llvm.abs.iN` intrinsic that
NVVM canonicalization re-forms from that.

Pooling layers that do not go through this pipeline: **MaxUnpool1d/2d/3d** lower to `tm_tensor.scatter`, a
torch-mlir dialect the standalone mlir-opt cannot parse; **FractionalMaxPool2d/3d** have no torch-mlir
lowering (`torch.aten.fractional_max_pool2d` is marked illegal); **AdaptiveMaxPool3d** emits a combined
max+argmax generic (two results, an i1 mask, 8 iterators) that overflows the scalar registers at 3D size
(the 1d/2d variants fit). Everything else in the Pooling section works.

Lazy* layers materialize on a first forward and are then identical to their non-lazy form; lazybatchnorm2d
is included as a representative and passes. SyncBatchNorm is a distributed (multi-process) layer and is not
applicable. So of the Normalization section, every layer runs except those two families, which are covered
by their non-lazy / non-distributed equivalents.

Every non-linear activation in the section runs, including nn.MultiheadAttention (self-attention);
AdaptiveLogSoftmaxWithLoss is the one exception -- it is a loss layer that needs a target and returns a
loss, so it belongs with the loss functions below.

Not attempted: the other Lazy* variants (redundant with the non-lazy layers), loss functions incl.
AdaptiveLogSoftmaxWithLoss (need a target), Embedding / EmbeddingBag (integer input, incompatible with the
float harness), and the distributed / container modules.

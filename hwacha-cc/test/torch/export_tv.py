#!/usr/bin/env python3
"""Export a torchvision classification model to linalg-on-tensors MLIR via torch-mlir, with a PyTorch
   reference. usage: export_tv.py <model_name> <out.mlir> <out_check.bin> [HW]  (HW = input size, default 32)"""
import sys, struct, numpy as np, torch, torchvision.models as M
from torch_mlir import fx
name, out, check = sys.argv[1:4]
HW = int(sys.argv[4]) if len(sys.argv) > 4 else 32
torch.manual_seed(0)
m = M.get_model(name, weights=None)
# untrained BN sits at its zero-mean/unit-var fixed point, which drives the logits to ~0 and makes
# argmax meaningless; give BN random running stats so the forward is non-degenerate (same weights feed
# the PyTorch reference and the Hwacha build, so it still tests exact agreement).
for mod in m.modules():
    if isinstance(mod, torch.nn.BatchNorm2d):
        mod.running_mean.normal_(0, 0.1); mod.running_var.uniform_(0.5, 1.5)
        mod.weight.data.uniform_(0.5, 1.5); mod.bias.data.normal_(0, 0.1)
m.eval()
x = torch.randn(1, 3, HW, HW)
with torch.no_grad(): ref = m(x)
mod = fx.export_and_import(m, x, output_type='linalg-on-tensors', func_name='net')
open(out, 'w').write(str(mod))
with open(check, 'wb') as f:
    f.write(struct.pack('i', 1))
    f.write(struct.pack('i', x.numel())); f.write(x.numpy().astype(np.float32).tobytes())
    f.write(struct.pack('i', ref.numel())); f.write(np.ascontiguousarray(ref).astype(np.float32).tobytes())
print("exported", name, "input", list(x.shape), "output", list(ref.shape))

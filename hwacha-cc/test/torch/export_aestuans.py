#!/usr/bin/env python3
"""aestuans/mnist-diffusion UNet -> linalg-on-tensors MLIR via torch-mlir, plus a reference point.
   usage: export_aestuans.py <repo dir> out.mlir out_check.bin"""
import sys, json, struct, numpy as np, torch
from torch_mlir import fx
repo, out, check = sys.argv[1:4]
sys.path.insert(0, repo)
from model import UNet, ModelConfig
cfg = json.load(open(f'{repo}/config.json'))
u = UNet(ModelConfig(**cfg['model_config']), 10)
u.load_state_dict(torch.load(f'{repo}/models/model_01.pt', map_location='cpu')); u.eval()

class Forward(torch.nn.Module):
    """_forward with the context mask as a float multiplier (the boolean-index assignment of the
    original would export as a dynamic-shape scatter)"""
    def __init__(self, u): super().__init__(); self.u = u
    def forward(self, x, t, c):
        u = self.u
        h, skip = u.encoder(x)
        t_e = u.time_mlp(t); c_e = u.context_mlp(c)
        t_e = t_e.view(1, -1, 1, 1).repeat(1, 1, h.size(2), h.size(3))
        c_e = c_e.view(1, -1, 1, 1).repeat(1, 1, h.size(2), h.size(3))
        h = torch.cat([h, t_e, c_e], dim=1)
        h = u.decoder(h)
        h = torch.cat([h, skip], dim=1)
        return u.final_conv(u.final_block(h))

x = torch.randn(1, 1, 28, 28); t = torch.tensor([[0.6]]); c = torch.nn.functional.one_hot(torch.tensor([3]), 10).float()
m = Forward(u)
with torch.no_grad():
    ref = m(x, t, c); ref0 = u(x, t, torch.tensor([3]))
    assert torch.allclose(ref, ref0), "wrapper differs from the model"
mod = fx.export_and_import(m, x, t, c, output_type='linalg-on-tensors', func_name='unet')
open(out, 'w').write(str(mod))
with open(check, 'wb') as f:
    f.write(struct.pack('if', 3, 0.6)); f.write(x.numpy().astype(np.float32).tobytes()); f.write(ref.numpy().astype(np.float32).tobytes())
print("exported", out, len(str(mod)) / 1e6, "MB")

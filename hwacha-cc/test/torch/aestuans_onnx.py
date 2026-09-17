#!/usr/bin/env python3
"""torch.onnx.export of the aestuans UNet (same static-shape wrapper as export_aestuans.py)"""
import sys, json, torch
repo, out = sys.argv[1:3]
sys.path.insert(0, repo)
from model import UNet, ModelConfig
cfg = json.load(open(f'{repo}/config.json'))
u = UNet(ModelConfig(**cfg['model_config']), 10)
u.load_state_dict(torch.load(f'{repo}/models/model_01.pt', map_location='cpu')); u.eval()
class Forward(torch.nn.Module):
    def __init__(self, u): super().__init__(); self.u = u
    def forward(self, x, t, c):
        u = self.u
        h, skip = u.encoder(x)
        t_e = u.time_mlp(t).view(1, -1, 1, 1).repeat(1, 1, h.size(2), h.size(3))
        c_e = u.context_mlp(c).view(1, -1, 1, 1).repeat(1, 1, h.size(2), h.size(3))
        h = u.decoder(torch.cat([h, t_e, c_e], dim=1))
        return u.final_conv(u.final_block(torch.cat([h, skip], dim=1)))
torch.onnx.export(Forward(u), (torch.randn(1, 1, 28, 28), torch.tensor([[0.6]]), torch.zeros(1, 10)), out,
                  input_names=['x', 't', 'c'], output_names=['eps'], opset_version=17, dynamo=False)
print("wrote", out)

# Export one basic PyTorch nn layer through torch-mlir to linalg-on-tensors, plus a PyTorch reference for
# one forward. Each module takes a single float tensor `net(x)` and returns a single float tensor, so one
# generic host (mod_main.c) drives them all.
#
#   python export_module.py <layer> <mlir_out> <check_out>
import sys, struct, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_mlir import fx

def rand_bn(m):   # give BatchNorm/GroupNorm non-degenerate eval stats + affine
    for mod in m.modules():
        if isinstance(mod,(nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
            mod.running_mean.normal_(0,0.3); mod.running_var.uniform_(0.5,1.5)
            mod.weight.data.uniform_(0.5,1.5); mod.bias.data.normal_(0,0.2)
    return m

class Attn(nn.Module):   # single-head-group multi-head self-attention (manual, so it lowers to matmul+softmax)
    def __init__(s,dim,heads):
        super().__init__(); s.h=heads; s.dh=dim//heads; s.sc=s.dh**-0.5
        s.qkv=nn.Linear(dim,dim*3); s.proj=nn.Linear(dim,dim)
    def forward(s,x):
        B,N,D=x.shape
        q,k,v=s.qkv(x).reshape(B,N,3,s.h,s.dh).permute(2,0,3,1,4)
        a=((q@k.transpose(-2,-1))*s.sc).softmax(-1)
        return s.proj((a@v).transpose(1,2).reshape(B,N,D))

# each entry: (module, input tensor). random seeds fixed for reproducibility.
def build(layer):
    torch.manual_seed(0)
    x4=torch.randn(1,4,8,8); x8=torch.randn(1,8,8,8)
    if layer=='linear':          return nn.Linear(16,8), torch.randn(1,16)
    if layer=='conv3x3':         return nn.Conv2d(4,8,3,padding=1), x4
    if layer=='conv1x1':         return nn.Conv2d(4,8,1), x4
    if layer=='conv_s2':         return nn.Conv2d(4,8,3,stride=2,padding=1), x4
    if layer=='conv5x5':         return nn.Conv2d(4,8,5,padding=2), x4
    if layer=='convtranspose2':  return nn.ConvTranspose2d(4,8,2,stride=2), x4
    if layer=='dwconv3x3':       return nn.Conv2d(8,8,3,padding=1,groups=8), x8
    if layer=='maxpool2':        return nn.MaxPool2d(2), x4
    if layer=='avgpool2':        return nn.AvgPool2d(2), x4
    if layer=='adaptiveavgpool': return nn.AdaptiveAvgPool2d(1), x4
    if layer=='batchnorm2d':     return rand_bn(nn.BatchNorm2d(4).eval()), x4
    if layer=='layernorm':       return nn.LayerNorm(16), torch.randn(1,4,16)
    if layer=='groupnorm':       return nn.GroupNorm(2,8), x8
    if layer=='relu':            return nn.ReLU(), torch.randn(1,16)
    if layer=='leakyrelu':       return nn.LeakyReLU(0.1), torch.randn(1,16)
    if layer=='gelu':            return nn.GELU(), torch.randn(1,16)
    if layer=='silu':            return nn.SiLU(), torch.randn(1,16)
    if layer=='sigmoid':         return nn.Sigmoid(), torch.randn(1,16)
    if layer=='tanh':            return nn.Tanh(), torch.randn(1,16)
    if layer=='hardswish':       return nn.Hardswish(), torch.randn(1,16)
    if layer=='hardsigmoid':     return nn.Hardsigmoid(), torch.randn(1,16)
    if layer=='elu':             return nn.ELU(), torch.randn(1,16)
    if layer=='softmax':         return nn.Softmax(-1), torch.randn(1,16)
    if layer=='flatten':         return nn.Flatten(), torch.randn(1,4,4,4)
    if layer=='upsample':        return nn.Upsample(scale_factor=2,mode='nearest'), torch.randn(1,4,4,4)
    if layer=='upsample_bilinear': return nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False), torch.randn(1,4,4,4)
    if layer=='pixelshuffle':    return nn.PixelShuffle(2), torch.randn(1,16,4,4)
    if layer=='attention':       return Attn(32,4), torch.randn(1,16,32)
    if layer=='mlp':             return nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16)), torch.randn(1,16)
    raise SystemExit("unknown layer "+layer)

if __name__=='__main__':
    layer, mlir_out, bin_out = sys.argv[1], sys.argv[2], sys.argv[3]
    m, x = build(layer); m=m.eval()
    with torch.no_grad(): y=m(x)
    print("%s: in %s -> out %s"%(layer,list(x.shape),list(y.shape)))
    mod=fx.export_and_import(m,x,output_type='linalg-on-tensors',func_name='net')
    open(mlir_out,'w').write(str(mod))
    xf=np.ascontiguousarray(x.numpy()).astype(np.float32).ravel()
    yf=np.ascontiguousarray(y.numpy()).astype(np.float32).ravel()
    with open(bin_out,'wb') as f:
        f.write(struct.pack('i',xf.size)); f.write(xf.tobytes())
        f.write(struct.pack('i',yf.size)); f.write(yf.tobytes())

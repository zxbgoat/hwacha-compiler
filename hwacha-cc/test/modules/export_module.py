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

class Fn(nn.Module):     # wrap an arbitrary torch.nn.functional call as a single-input module
    def __init__(s, fn): super().__init__(); s.fn=fn
    def forward(s, x): return s.fn(x)

class Seq0(nn.Module):   # wrap a module whose forward returns a tuple (e.g. RNN/LSTM/GRU) -> first element
    def __init__(s, m): super().__init__(); s.m=m
    def forward(s, x): return s.m(x)[0]

class TwoIn(nn.Module):  # wrap a two-input module (decoder tgt+memory, transformer src+tgt) as single-input
    def __init__(s, m, second_shape):
        super().__init__(); s.m=m; s.register_buffer('second', torch.randn(*second_shape))
    def forward(s, x): return s.m(x, s.second)

class SDPA(nn.Module):   # F.scaled_dot_product_attention (fused): does torch-mlir decompose it to matmul+softmax?
    def __init__(s,dim,heads):
        super().__init__(); s.h=heads; s.dh=dim//heads
        s.qkv=nn.Linear(dim,dim*3); s.proj=nn.Linear(dim,dim)
    def forward(s,x):
        B,N,D=x.shape
        q,k,v=s.qkv(x).reshape(B,N,3,s.h,s.dh).permute(2,0,3,1,4)
        o=F.scaled_dot_product_attention(q,k,v)
        return s.proj(o.transpose(1,2).reshape(B,N,D))

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
    # ---- torch.nn.functional ops not covered by the nn-module layers above ----
    # extra activations
    if layer=='softplus':        return nn.Softplus(), torch.randn(1,16)
    if layer=='mish':            return nn.Mish(), torch.randn(1,16)
    if layer=='hardtanh':        return nn.Hardtanh(), torch.randn(1,16)
    if layer=='relu6':           return nn.ReLU6(), torch.randn(1,16)
    if layer=='selu':            return nn.SELU(), torch.randn(1,16)
    if layer=='celu':            return nn.CELU(), torch.randn(1,16)
    if layer=='softsign':        return nn.Softsign(), torch.randn(1,16)
    if layer=='tanhshrink':      return nn.Tanhshrink(), torch.randn(1,16)
    if layer=='softshrink':      return nn.Softshrink(), torch.randn(1,16)
    if layer=='hardshrink':      return nn.Hardshrink(), torch.randn(1,16)
    if layer=='logsigmoid':      return nn.LogSigmoid(), torch.randn(1,16)
    if layer=='logsoftmax':      return nn.LogSoftmax(-1), torch.randn(1,16)
    if layer=='prelu':           return nn.PReLU(num_parameters=1), torch.randn(1,16)
    if layer=='glu':             return nn.GLU(dim=-1), torch.randn(1,16)
    if layer=='threshold':       return nn.Threshold(0.5,0.0), torch.randn(1,16)
    # other functional ops
    if layer=='normalize':       return Fn(lambda t: F.normalize(t,dim=-1)), torch.randn(1,16)
    if layer=='instancenorm':    return nn.InstanceNorm2d(4,affine=True).eval(), x4
    if layer=='lrn':             return nn.LocalResponseNorm(3), x4
    if layer=='pad':             return Fn(lambda t: F.pad(t,(1,1,1,1))), x4
    if layer=='pad_reflect':     return Fn(lambda t: F.pad(t,(1,1,1,1),mode='reflect')), x4
    if layer=='sdpa':            return SDPA(32,4), torch.randn(1,16,32)
    if layer=='unfold':          return nn.Unfold(kernel_size=2,stride=2), x4
    if layer=='interpolate':     return Fn(lambda t: F.interpolate(t,scale_factor=2,mode='bilinear',align_corners=False)), torch.randn(1,4,4,4)
    # ---- rest of the torch.nn layer catalogue (docs.pytorch.org/docs/2.14/nn.html) ----
    x1=torch.randn(1,4,16); x3=torch.randn(1,2,4,4,4)   # 1D (N,C,L) and 3D (N,C,D,H,W) tensors
    # convolution dimension variants
    if layer=='conv1d':          return nn.Conv1d(4,8,3,padding=1), x1
    if layer=='conv3d':          return nn.Conv3d(2,4,3,padding=1), x3
    if layer=='convtranspose1d': return nn.ConvTranspose1d(4,8,2,stride=2), torch.randn(1,4,8)
    if layer=='convtranspose3d': return nn.ConvTranspose3d(2,4,2,stride=2), x3
    # pooling dimension / kind variants
    if layer=='maxpool1d':       return nn.MaxPool1d(2), x1
    if layer=='maxpool3d':       return nn.MaxPool3d(2), x3
    if layer=='avgpool1d':       return nn.AvgPool1d(2), x1
    if layer=='avgpool3d':       return nn.AvgPool3d(2), x3
    if layer=='adaptiveavgpool1d': return nn.AdaptiveAvgPool1d(1), x1
    if layer=='adaptiveavgpool3d': return nn.AdaptiveAvgPool3d(1), x3
    if layer=='adaptivemaxpool1d': return nn.AdaptiveMaxPool1d(1), x1
    if layer=='adaptivemaxpool2d': return nn.AdaptiveMaxPool2d(1), x4
    if layer=='lppool2d':        return nn.LPPool2d(2,2), x4
    # padding
    if layer=='zeropad2d':       return nn.ZeroPad2d(1), x4
    if layer=='constantpad2d':   return nn.ConstantPad2d(1,0.5), x4
    if layer=='replicationpad2d':return nn.ReplicationPad2d(1), x4
    if layer=='circularpad2d':   return nn.CircularPad2d(1), x4
    if layer=='reflectionpad2d': return nn.ReflectionPad2d(1), x4
    if layer=='zeropad1d':       return nn.ConstantPad1d(1,0.0), x1
    # normalization variants
    if layer=='batchnorm1d':     return rand_bn(nn.BatchNorm1d(4).eval()), x1
    if layer=='batchnorm3d':     return rand_bn(nn.BatchNorm3d(2).eval()), x3
    if layer=='instancenorm1d':  return nn.InstanceNorm1d(4,affine=True).eval(), x1
    if layer=='rmsnorm':         return nn.RMSNorm(16), torch.randn(1,4,16)
    # more activations
    if layer=='rrelu':           return nn.RReLU().eval(), torch.randn(1,16)
    if layer=='softmin':         return nn.Softmin(-1), torch.randn(1,16)
    if layer=='softmax2d':       return nn.Softmax2d(), x4
    if layer=='identity':        return nn.Identity(), torch.randn(1,16)
    # dropout in eval mode = identity
    if layer=='dropout':         return nn.Dropout(0.5).eval(), torch.randn(1,16)
    if layer=='dropout2d':       return nn.Dropout2d(0.5).eval(), x4
    if layer=='alphadropout':    return nn.AlphaDropout(0.5).eval(), torch.randn(1,16)
    # shape / vision / shuffle
    if layer=='unflatten':       return nn.Unflatten(1,(2,8)), torch.randn(1,16)
    if layer=='pixelunshuffle':  return nn.PixelUnshuffle(2), x4
    if layer=='channelshuffle':  return nn.ChannelShuffle(2), x8
    if layer=='upsamplingnearest2d':  return nn.UpsamplingNearest2d(scale_factor=2), torch.randn(1,4,4,4)
    if layer=='upsamplingbilinear2d': return nn.UpsamplingBilinear2d(scale_factor=2), torch.randn(1,4,4,4)
    # recurrent (return only the output sequence / hidden, not the tuple)
    if layer=='rnn':             return Seq0(nn.RNN(8,16,batch_first=True).eval()), torch.randn(1,5,8)
    if layer=='lstm':            return Seq0(nn.LSTM(8,16,batch_first=True).eval()), torch.randn(1,5,8)
    if layer=='gru':             return Seq0(nn.GRU(8,16,batch_first=True).eval()), torch.randn(1,5,8)
    if layer=='rnncell':         return nn.RNNCell(8,16), torch.randn(1,8)
    # transformer
    if layer=='transformerencoderlayer':
        return nn.TransformerEncoderLayer(d_model=32,nhead=4,dim_feedforward=64,batch_first=True).eval(), torch.randn(1,8,32)
    if layer=='transformerencoder':
        el=nn.TransformerEncoderLayer(d_model=32,nhead=4,dim_feedforward=64,batch_first=True)
        return nn.TransformerEncoder(el,num_layers=2).eval(), torch.randn(1,8,32)
    if layer=='transformerdecoderlayer':
        dl=nn.TransformerDecoderLayer(d_model=32,nhead=4,dim_feedforward=64,batch_first=True).eval()
        return TwoIn(dl,(1,8,32)), torch.randn(1,8,32)                      # forward(tgt) with a fixed memory
    if layer=='transformerdecoder':
        dl=nn.TransformerDecoderLayer(d_model=32,nhead=4,dim_feedforward=64,batch_first=True)
        return TwoIn(nn.TransformerDecoder(dl,num_layers=2).eval(),(1,8,32)), torch.randn(1,8,32)
    if layer=='transformer':
        tr=nn.Transformer(d_model=32,nhead=4,num_encoder_layers=2,num_decoder_layers=2,dim_feedforward=64,batch_first=True).eval()
        return TwoIn(tr,(1,8,32)), torch.randn(1,8,32)                      # forward(src) with a fixed tgt
    if layer=='groupconv':       return nn.Conv2d(8,8,3,padding=1,groups=2), x8
    if layer=='groupconv_s2':    return nn.Conv2d(8,8,3,stride=2,padding=1,groups=2), x8
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

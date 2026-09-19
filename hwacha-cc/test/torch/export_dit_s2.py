# DiT-S/2 at the real config (Peebles & Xie): latent 4x32x32, patch 2 -> 256 tokens, depth 12,
# hidden 384, 6 heads, class-conditional. The sinusoidal timestep embedding is a fixed function of the
# scalar t, so it is precomputed on the host and fed in; the learned temb MLP and everything else run on
# Hwacha. The model returns the per-token output [1, 256, P*P*C]; unpatchify is done on the host.
#
#   python export_dit_s2.py dit_s2.mlir dit_s2_check.bin [depth hidden heads]
import sys, struct, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_mlir import fx
from export_dit import Block, modulate, unpatchify

FREQ = 256   # sinusoidal timestep embedding size

def timestep_embedding(t, dim=FREQ, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # [B, dim]

class DiTS2(nn.Module):
    def __init__(s, C=4, H=32, P=2, dim=384, depth=12, heads=6, nclass=10):
        super().__init__()
        s.P=P; s.C=C; s.gh=H//P; s.nt=s.gh*s.gh
        s.patch=nn.Conv2d(C,dim,P,stride=P)
        s.pos=nn.Parameter(torch.randn(1,s.nt,dim)*0.02)
        s.temb=nn.Sequential(nn.Linear(FREQ,dim), nn.SiLU(), nn.Linear(dim,dim))
        s.yemb=nn.Embedding(nclass,dim)
        s.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        s.nf=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6)
        s.adaf=nn.Sequential(nn.SiLU(),nn.Linear(dim,2*dim,bias=True))
        s.head=nn.Linear(dim,P*P*C)
    def forward(s, x, temb_sincos, y):     # temb_sincos: [1, FREQ] precomputed on host
        h=s.patch(x).flatten(2).transpose(1,2)+s.pos
        c=s.temb(temb_sincos)+s.yemb(y)
        for blk in s.blocks: h=blk(h,c)
        sf,bf=s.adaf(c).chunk(2,-1)
        return s.head(modulate(s.nf(h),bf,sf))

if __name__=='__main__':
    mlir_out, bin_out = sys.argv[1], sys.argv[2]
    a=sys.argv[3:]
    depth = int(a[0]) if len(a)>0 else 12
    dim   = int(a[1]) if len(a)>1 else 384
    heads = int(a[2]) if len(a)>2 else 6
    C,H,P = 4,32,2
    torch.manual_seed(0)
    m=DiTS2(C=C,H=H,P=P,dim=dim,depth=depth,heads=heads).eval()
    x=torch.randn(1,C,H,H); tval=torch.tensor([0.7]); y=torch.tensor([3])
    emb=timestep_embedding(tval)              # [1, FREQ] on host
    with torch.no_grad(): tok=m(x,emb,y)
    print("DiT-S/2 depth %d hidden %d heads %d: tokens %s max|tok| %.4f"%(depth,dim,heads,list(tok.shape),float(tok.abs().max())))
    mod=fx.export_and_import(m,x,emb,y,output_type='linalg-on-tensors',func_name='dit')
    open(mlir_out,'w').write(str(mod))
    xf=np.ascontiguousarray(x.numpy()).astype(np.float32).ravel()
    ef=np.ascontiguousarray(emb.numpy()).astype(np.float32).ravel()
    tf=np.ascontiguousarray(tok.numpy()).astype(np.float32).ravel()
    with open(bin_out,'wb') as f:
        f.write(struct.pack('i',xf.size)); f.write(xf.tobytes())
        f.write(struct.pack('i',ef.size)); f.write(ef.tobytes())
        f.write(struct.pack('i',int(y.item())))
        f.write(struct.pack('i',tf.size)); f.write(tf.tobytes())
    print("check.bin: nx %d nemb %d ntok %d"%(xf.size,ef.size,tf.size))

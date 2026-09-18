# Minimal DiT (Diffusion Transformer, Peebles & Xie) exported through torch-mlir to linalg-on-tensors,
# plus a PyTorch reference for one forward. The transformer runs whole on Hwacha (patchify, self-attention
# with softmax, adaLN modulation, MLP with GELU, final layer); the model returns the per-token output
# [1, nt, P*P*C] and the trivial unpatchify (a fixed pixel permutation) is left to the host.
#
#   python export_dit.py dit.mlir dit_check.bin [dim depth heads C H P]
import sys, struct, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch_mlir import fx

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class Attention(nn.Module):
    def __init__(s, dim, heads):
        super().__init__(); s.h=heads; s.dh=dim//heads; s.scale=s.dh**-0.5
        s.qkv=nn.Linear(dim,dim*3,bias=True); s.proj=nn.Linear(dim,dim)
    def forward(s,x):
        B,N,D=x.shape
        qkv=s.qkv(x).reshape(B,N,3,s.h,s.dh).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        att=((q@k.transpose(-2,-1))*s.scale).softmax(-1)
        return s.proj((att@v).transpose(1,2).reshape(B,N,D))

class Mlp(nn.Module):
    def __init__(s,dim,hid):
        super().__init__(); s.fc1=nn.Linear(dim,hid); s.fc2=nn.Linear(hid,dim)
    def forward(s,x): return s.fc2(F.gelu(s.fc1(x)))

class Block(nn.Module):
    def __init__(s,dim,heads,mlp=4):
        super().__init__()
        s.n1=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6); s.attn=Attention(dim,heads)
        s.n2=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6); s.mlp=Mlp(dim,dim*mlp)
        s.ada=nn.Sequential(nn.SiLU(), nn.Linear(dim,6*dim,bias=True))
    def forward(s,x,c):
        sa,ba,ga,sm,bm,gm=s.ada(c).chunk(6,-1)
        x=x+ga.unsqueeze(1)*s.attn(modulate(s.n1(x),ba,sa))
        x=x+gm.unsqueeze(1)*s.mlp(modulate(s.n2(x),bm,sm))
        return x

class DiT(nn.Module):
    def __init__(s,C=4,H=8,P=2,dim=64,depth=2,heads=4,nclass=10):
        super().__init__()
        s.P=P; s.C=C; s.gh=H//P; s.nt=s.gh*s.gh
        s.patch=nn.Conv2d(C,dim,P,stride=P)
        s.pos=nn.Parameter(torch.randn(1,s.nt,dim)*0.02)
        s.temb=nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
        s.yemb=nn.Embedding(nclass,dim)
        s.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        s.nf=nn.LayerNorm(dim,elementwise_affine=False,eps=1e-6)
        s.adaf=nn.Sequential(nn.SiLU(),nn.Linear(dim,2*dim,bias=True))
        s.head=nn.Linear(dim,P*P*C)
    def forward(s,x,t,y):
        h=s.patch(x).flatten(2).transpose(1,2)+s.pos
        c=s.temb(t)+s.yemb(y)
        for blk in s.blocks: h=blk(h,c)
        sf,bf=s.adaf(c).chunk(2,-1)
        return s.head(modulate(s.nf(h),bf,sf))      # [1, nt, P*P*C] tokens

def unpatchify(tok, C, P, gh):    # host-side: [1,nt,P*P*C] -> [1,C,gh*P,gh*P] in DiT ordering
    B=tok.shape[0]
    return tok.reshape(B,gh,gh,P,P,C).permute(0,5,1,3,2,4).reshape(B,C,gh*P,gh*P)

if __name__=='__main__':
    mlir_out, bin_out = sys.argv[1], sys.argv[2]
    a=sys.argv[3:]
    dim  = int(a[0]) if len(a)>0 else 64
    depth= int(a[1]) if len(a)>1 else 2
    heads= int(a[2]) if len(a)>2 else 4
    C    = int(a[3]) if len(a)>3 else 4
    H    = int(a[4]) if len(a)>4 else 8
    P    = int(a[5]) if len(a)>5 else 2
    torch.manual_seed(0)
    m=DiT(C=C,H=H,P=P,dim=dim,depth=depth,heads=heads).eval()
    x=torch.randn(1,C,H,H); t=torch.randn(1,1); y=torch.tensor([3])
    with torch.no_grad(): tok=m(x,t,y)
    print("DiT dim %d depth %d heads %d: tokens %s max|tok| %.4f"%(dim,depth,heads,list(tok.shape),float(tok.abs().max())))
    mod=fx.export_and_import(m,x,t,y,output_type='linalg-on-tensors',func_name='dit')
    open(mlir_out,'w').write(str(mod))
    xf=np.ascontiguousarray(x.numpy()).astype(np.float32).ravel()
    tf=np.ascontiguousarray(tok.numpy()).astype(np.float32).ravel()
    with open(bin_out,'wb') as f:
        f.write(struct.pack('i',xf.size)); f.write(xf.tobytes())
        f.write(struct.pack('f',float(t.item()))); f.write(struct.pack('i',int(y.item())))
        f.write(struct.pack('i',tf.size)); f.write(tf.tobytes())

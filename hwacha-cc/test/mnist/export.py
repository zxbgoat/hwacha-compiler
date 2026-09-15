#!/usr/bin/env python3
"""Export the three MNIST diffusion models to flat fp32 .bin files for mnist_main.c.
   usage: export.py <repo-parent-dir>   (expects MNISTDiffusion/, Conditional_Diffusion_MNIST/, mnist-diffusion/)
   Each file: int32 magic, then the model's config ints, then every parameter tensor in the fixed
   order the C loader expects (documented next to each export function)."""
import sys, os, struct, numpy as np, torch
root = sys.argv[1] if len(sys.argv) > 1 else '/tmp'
def w(f, t): f.write(np.ascontiguousarray(t.detach().cpu().float().numpy()).tobytes())
def ints(f, *v): f.write(struct.pack(f'{len(v)}i', *v))

# ---------------------------------------------------------------- bot66/MNISTDiffusion
# ConvBnSiLu = conv(w,b) + BN(gamma,beta,mean,var) + SiLU ; depthwise conv + BN ; Linear(w,b) ; Embedding
def bn(f, m): w(f, m.weight); w(f, m.bias); w(f, m.running_mean); w(f, m.running_var)
def conv(f, m): w(f, m.weight); w(f, m.bias) if m.bias is not None else None
def cbs(f, m): conv(f, m.module[0]); bn(f, m.module[1])
def bottleneck(f, m):   # branch1: dw conv, bn, cbs ; branch2: cbs, dw conv, bn, cbs
    conv(f, m.branch1[0]); bn(f, m.branch1[1]); cbs(f, m.branch1[2])
    cbs(f, m.branch2[0]); conv(f, m.branch2[1]); bn(f, m.branch2[2]); cbs(f, m.branch2[3])
def downsample(f, m):   # branch1: dw conv s2, bn, cbs ; branch2: cbs, dw conv s2, bn, cbs
    conv(f, m.branch1[0]); bn(f, m.branch1[1]); cbs(f, m.branch1[2])
    cbs(f, m.branch2[0]); conv(f, m.branch2[1]); bn(f, m.branch2[2]); cbs(f, m.branch2[3])
def timemlp(f, m): w(f, m.mlp[0].weight); w(f, m.mlp[0].bias); w(f, m.mlp[2].weight); w(f, m.mlp[2].bias)
def export_bot66():
    sys.modules.pop('model', None); sys.path.insert(0, f'{root}/MNISTDiffusion')
    from model import MNISTDiffusion
    import glob
    ck = sorted(glob.glob(f'{root}/MNISTDiffusion/results/steps_*.pt'))[-1]
    m = MNISTDiffusion(timesteps=1000, image_size=28, in_channels=1, base_dim=64, dim_mults=[2, 4])
    sd = torch.load(ck, map_location='cpu')['model_ema']
    sd = {k[len('module.'):]: v for k, v in sd.items() if k.startswith('module.')}
    m.load_state_dict(sd); m.eval()
    u = m.model
    with open('bot66.bin', 'wb') as f:
        ints(f, 0xB066, 1000, 256, 64, 2, 4)   # magic, timesteps, time_embedding_dim, base_dim, dim_mults
        w(f, m.betas)
        cbs(f, u.init_conv); w(f, u.time_embedding.weight)
        for e in u.encoder_blocks:
            for b in e.conv0: bottleneck(f, b)
            timemlp(f, e.time_mlp); downsample(f, e.conv1)
        for b in u.mid_block: bottleneck(f, b)
        for d in u.decoder_blocks:
            for b in d.conv0: bottleneck(f, b)
            timemlp(f, d.time_mlp); bottleneck(f, d.conv1)
        conv(f, u.final_conv)
    return m
# ---------------------------------------------------------------- TeaPearce/Conditional_Diffusion_MNIST
def export_teapearce():
    src = open(f'{root}/Conditional_Diffusion_MNIST/script.py').read().split('\ndef train_mnist')[0]
    ns = {}; exec(compile(src, 'script.py', 'exec'), ns)
    import zipfile; zipfile.ZipFile(f'{root}/Conditional_Diffusion_MNIST/pretrained_model.zip').extractall(f'{root}/Conditional_Diffusion_MNIST/pre')
    ddpm = ns['DDPM'](nn_model=ns['ContextUnet'](in_channels=1, n_feat=128, n_classes=10), betas=(1e-4, 0.02), n_T=400, device='cpu', drop_prob=0.1)
    ddpm.load_state_dict(torch.load(f'{root}/Conditional_Diffusion_MNIST/pre/model_39.pth', map_location='cpu')); ddpm.eval()
    u = ddpm.nn_model
    def cbg(f, seq): conv(f, seq[0]); bn(f, seq[1])           # Conv, BatchNorm (eval: running stats!), GELU
    def resblock(f, m): cbg(f, m.conv1); cbg(f, m.conv2)
    def embfc(f, m): w(f, m.model[0].weight); w(f, m.model[0].bias); w(f, m.model[2].weight); w(f, m.model[2].bias)
    def gn(f, m): w(f, m.weight); w(f, m.bias)
    with open('teapearce.bin', 'wb') as f:
        ints(f, 0x7EA9, 400, 128, 10)   # magic, n_T, n_feat, n_classes
        resblock(f, u.init_conv)
        resblock(f, u.down1.model[0]); resblock(f, u.down2.model[0])
        embfc(f, u.timeembed1); embfc(f, u.timeembed2); embfc(f, u.contextembed1); embfc(f, u.contextembed2)
        conv(f, u.up0[0]); gn(f, u.up0[1])
        for up in (u.up1, u.up2): conv(f, up.model[0]); resblock(f, up.model[1]); resblock(f, up.model[2])
        conv(f, u.out[0]); gn(f, u.out[1]); conv(f, u.out[3])
    return ddpm
# ---------------------------------------------------------------- aestuans/mnist-diffusion
def export_aestuans():
    sys.modules.pop('model', None); sys.path.insert(0, f'{root}/mnist-diffusion')
    from model import UNet, ModelConfig
    u = UNet(ModelConfig([64, 64, 128], [128, 64, 64], 16), 10)
    u.load_state_dict(torch.load(f'{root}/mnist-diffusion/models/model_01.pt', map_location='cpu')); u.eval()
    def res(f, m):
        w(f, m.conv1.weight); bn(f, m.bn1); w(f, m.conv2.weight); bn(f, m.bn2)
        if len(m.shortcut): w(f, m.shortcut[0].weight); bn(f, m.shortcut[1])
    def mlp(f, m): w(f, m.fc1.weight); w(f, m.fc1.bias); w(f, m.fc2.weight); w(f, m.fc2.bias)
    with open('aestuans.bin', 'wb') as f:
        ints(f, 0xAE57, 500, 64, 64, 128, 16, 10)   # magic, steps, enc feats, embedding feats, classes
        res(f, u.encoder.block1); res(f, u.encoder.block2); res(f, u.encoder.block3)
        mlp(f, u.time_mlp); mlp(f, u.context_mlp)
        res(f, u.decoder.up_block1); conv(f, u.decoder.up_conv1); res(f, u.decoder.up_block2); conv(f, u.decoder.up_conv2)
        res(f, u.final_block); conv(f, u.final_conv)
    return u


# ---------------------------------------------------------------- reference points for the C port: (x, t, c) -> eps
def write_check(name, x, extra_ints, extra_floats, eps):
    with open(name + '_check.bin', 'wb') as f:
        ints(f, *extra_ints); f.write(struct.pack(f'{len(extra_floats)}f', *extra_floats)); w(f, x); w(f, eps)
def checks():
    torch.manual_seed(1)
    x = torch.randn(1, 1, 28, 28)
    m = export_bot66()
    with torch.no_grad(): eps = m.model(x, torch.tensor([900]))
    write_check('bot66', x, [900], [], eps)
    d = export_teapearce()
    with torch.no_grad():
        eps1 = d.nn_model(x, torch.tensor([7]), torch.tensor([[[[0.75]]]]), torch.zeros(1))
        eps2 = d.nn_model(x, torch.tensor([7]), torch.tensor([[[[0.75]]]]), torch.ones(1))
    write_check('teapearce', x, [7], [0.75], torch.cat([eps1, eps2], 0))
    u = export_aestuans()
    with torch.no_grad(): eps = u(x, torch.tensor([[0.6]]), torch.tensor([3]))
    write_check('aestuans', x, [3], [0.6], eps)
if __name__ == '__main__':
    checks()
    for n in ('bot66', 'teapearce', 'aestuans'): print(n, os.path.getsize(n + '.bin') / 1e6, 'MB')

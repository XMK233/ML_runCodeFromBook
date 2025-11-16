## loss不降反升。。。

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler
from tqdm.auto import tqdm
import os
from torchvision import transforms, datasets
from PIL import Image, UnidentifiedImageError

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, include_prefixes=None, only_jpg_with_underscore=False):
        self.include_prefixes = include_prefixes
        self.only_jpg_with_underscore = only_jpg_with_underscore
        super().__init__(root=root, transform=transform)
        kept, skipped = [], 0
        for path, target in self.samples:
            fname = os.path.basename(path)
            dname = os.path.basename(os.path.dirname(path))
            prefix = dname.split()[0] if dname else ""
            if self.include_prefixes is not None and prefix not in self.include_prefixes:
                continue
            if self.only_jpg_with_underscore:
                if not fname.lower().endswith(".jpg") or "_" not in fname or fname.startswith("._"):
                    continue
            try:
                with Image.open(path) as im:
                    im.verify()
            except (UnidentifiedImageError, OSError):
                skipped += 1
                continue
            kept.append((path, target))
        present_classes = sorted({os.path.basename(os.path.dirname(p)) for p, _ in kept})
        class_to_idx = {cls: i for i, cls in enumerate(present_classes)}
        remapped = [(p, class_to_idx[os.path.basename(os.path.dirname(p))]) for p, _ in kept]
        self.classes = present_classes
        self.class_to_idx = class_to_idx
        self.samples = remapped
        self.imgs = remapped
        self.targets = [t for _, t in remapped]
    def find_classes(self, directory):
        classes = []
        for d in sorted(os.listdir(directory)):
            p = os.path.join(directory, d)
            if not os.path.isdir(p):
                continue
            prefix = d.split()[0] if d else ""
            if self.include_prefixes is None or prefix in self.include_prefixes:
                classes.append(d)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

root = "/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志"
img_sz = (64, 64)
tf = transforms.Compose([
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])
include = [str(i) for i in range(2, 23)]
dataset = SafeImageFolder(root=root, transform=tf, include_prefixes=include, only_jpg_with_underscore=True)
train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + self.gamma.view(1, -1, 1, 1) * x

class ViTBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        m = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, m), nn.GELU(), nn.Linear(m, dim))
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

def build_2d_sincos(h, w, dim):
    y = torch.arange(h).float()
    x = torch.arange(w).float()
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    omega = torch.arange(dim // 4).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    pos_x = xx.reshape(-1, 1) * omega.reshape(1, -1)
    pos_y = yy.reshape(-1, 1) * omega.reshape(1, -1)
    emb = torch.cat([torch.sin(pos_x), torch.cos(pos_x), torch.sin(pos_y), torch.cos(pos_y)], dim=1)
    if emb.shape[1] < dim:
        pad = torch.zeros(emb.shape[0], dim - emb.shape[1])
        emb = torch.cat([emb, pad], dim=1)
    return emb

class ClassConditionedConvNeXtTransformer(nn.Module):
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4, dims=(64, 128, 256), vit_dim=256, vit_layers=6, vit_heads=4):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        in_ch = 1 + class_emb_size
        self.stem = nn.Conv2d(in_ch, dims[0], 4, 4)
        self.stage1 = nn.Sequential(ConvNeXtBlock(dims[0]), ConvNeXtBlock(dims[0]))
        self.down1 = nn.Conv2d(dims[0], dims[1], 2, 2)
        self.stage2 = nn.Sequential(ConvNeXtBlock(dims[1]), ConvNeXtBlock(dims[1]))
        self.down2 = nn.Conv2d(dims[1], dims[2], 2, 2)
        self.stage3 = nn.Sequential(ConvNeXtBlock(dims[2]), ConvNeXtBlock(dims[2]))
        self.vproj_in = nn.Conv2d(dims[2], vit_dim, 1)
        gh = img_sz[0] // 16
        gw = img_sz[1] // 16
        pe = build_2d_sincos(gh, gw, vit_dim)
        self.register_buffer('pos_embed', pe.unsqueeze(0), persistent=False)
        self.vblocks = nn.ModuleList([ViTBlock(vit_dim, vit_heads) for _ in range(vit_layers)])
        self.vnorm = nn.LayerNorm(vit_dim)
        self.vproj_out = nn.Conv2d(vit_dim, dims[2], 1)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(dims[2], dims[1], 2, 2), ConvNeXtBlock(dims[1]))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(dims[1], dims[0], 2, 2), ConvNeXtBlock(dims[0]))
        self.out = nn.Sequential(nn.ConvTranspose2d(dims[0], 64, 4, 4), nn.Conv2d(64, 1, 1))
    def forward(self, x, t, class_labels):
        bs, _, h, w = x.shape
        c = self.class_emb(class_labels).view(bs, -1, 1, 1).expand(bs, -1, h, w)
        z = torch.cat([x, c], 1)
        y = self.stem(z)
        y = self.stage1(y)
        y = self.down1(y)
        y = self.stage2(y)
        y = self.down2(y)
        y = self.stage3(y)
        v = self.vproj_in(y)
        gh, gw = v.shape[2], v.shape[3]
        s = v.flatten(2).transpose(1, 2)
        s = s + self.pos_embed[:, : gh * gw, :].to(s.device)
        for blk in self.vblocks:
            s = blk(s)
        s = self.vnorm(s)
        s = s.transpose(1, 2).reshape(bs, -1, gh, gw)
        v = self.vproj_out(s)
        v = self.up1(v)
        v = self.up2(v)
        return self.out(v)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
n_epochs = 20
net = ClassConditionedConvNeXtTransformer().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
net_ema = ClassConditionedConvNeXtTransformer().to(device)
net_ema.load_state_dict(net.state_dict())
ema_decay = 0.999

losses = []
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        x = x.to(device) * 2 - 1
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        pred = net(noisy_x, timesteps, y)
        velocity = noise_scheduler.get_velocity(x, noise, timesteps)
        alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
        snr = alphas / (1 - alphas)
        weight = torch.sqrt(torch.clamp(snr, max=5.0))
        loss = (weight * (pred - velocity) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for p, p_ema in zip(net.parameters(), net_ema.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        losses.append(loss.item())
    avg_loss = sum(losses[-100:]) / 100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
    if (epoch + 1) % 5 == 0:
        xg = torch.randn(8 * 2, 1, img_sz[0], img_sz[1]).to(device)
        yg = torch.tensor([[i] * 8 for i in range(2)]).flatten().to(device)
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(50)
        for i, t in tqdm(enumerate(ddim.timesteps)):
            with torch.no_grad():
                net_ema.eval()
                residual = net_ema(xg, t.to(xg.device), yg)
            xg = ddim.step(residual, t, xg).prev_sample
        out_name = os.path.join(os.path.dirname(__file__), f'samples_convnext_transformer_epoch{epoch+1}.png')
        torchvision.utils.save_image(
            xg.detach().cpu().clip(-1, 1),
            out_name,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )

x = torch.randn(8 * 2, 1, img_sz[0], img_sz[1]).to(device)
y = torch.tensor([[i] * 8 for i in range(2)]).flatten().to(device)
ddim = DDIMScheduler.from_config(noise_scheduler.config)
ddim.set_timesteps(30)
for i, t in tqdm(enumerate(ddim.timesteps)):
    with torch.no_grad():
        residual = net(x, t.to(x.device), y)
    x = ddim.step(residual, t, x).prev_sample
torchvision.utils.save_image(
    x.detach().cpu().clip(-1, 1),
    os.path.join(os.path.dirname(__file__), 'samples_convnext_transformer_bw.png'),
    nrow=8,
    normalize=True,
    value_range=(-1, 1)
)
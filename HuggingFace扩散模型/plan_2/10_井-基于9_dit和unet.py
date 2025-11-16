## 探索的一些线路：
### 一开始的时候是unet，loss下降到0.03上下就不降了。
### 后来探索dit。几乎不会下降。搞了一下，还是不怎么降低。
### 后来求助了Qwen，尝试了本次的Uvit(unet+dit)。loss能下降了。
### 但是又遇上了一些问题，就是生成的图片有的比较清晰，有的比较糊。
### 改进方案：【】训练损失切换为 v-pred + SNR 加权： plan_2/10_井-基于9_dit和unet.py:166-171
## 采样步数 60，并加入简化 CFG（条件与“轻度无条件”预测结合，提高清晰度）： plan_2/10_井-基于9_dit和unet.py:182-191
# - 随机 10% 条件 dropout 生成 yg_uncond
# - residual = residual_uncond + guidance_scale * (residual_cond - residual_uncond) ， guidance_scale=2.0

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler
from matplotlib import pyplot as plt
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
train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class ViTBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
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

class ClassConditionedUViT(nn.Module):
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4, vit_dim=256, vit_heads=4, vit_layers=6):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        in_ch = 1 + class_emb_size
        self.down1 = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.pool1 = nn.Conv2d(64, 64, 3, 2, 1)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(), nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU())
        self.pool2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(), nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU())
        self.pool3 = nn.Conv2d(128, 128, 3, 2, 1)
        self.vproj_in = nn.Conv2d(128, vit_dim, 1)
        self.vblocks = nn.ModuleList([ViTBlock(vit_dim, vit_heads) for _ in range(vit_layers)])
        self.vnorm = nn.LayerNorm(vit_dim)
        self.vproj_out = nn.Conv2d(vit_dim, 128, 1)
        self.up1 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(), nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU())
        self.up2 = nn.Sequential(nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(), nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU())
        self.up3 = nn.Sequential(nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(), nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU())
        self.out = nn.Conv2d(64, 1, 1)
        gh = img_sz[0] // 8
        gw = img_sz[1] // 8
        pe = build_2d_sincos(gh, gw, vit_dim)
        self.register_buffer('pos_embed', pe.unsqueeze(0), persistent=False)
    def forward(self, x, t, class_labels):
        bs, _, h, w = x.shape
        c = self.class_emb(class_labels).view(bs, -1, 1, 1).expand(bs, -1, h, w)
        z = torch.cat([x, c], 1)
        d1 = self.down1(z)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        vb = self.vproj_in(p3)
        gh = vb.shape[2]
        gw = vb.shape[3]
        s = vb.flatten(2).transpose(1, 2)
        s = s + self.pos_embed[:, : gh * gw, :].to(s.device)
        for blk in self.vblocks:
            s = blk(s)
        s = self.vnorm(s)
        s = s.transpose(1, 2).reshape(bs, -1, gh, gw)
        yb = self.vproj_out(s)
        u1 = F.interpolate(yb, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = torch.cat([u1, d3], 1)
        u1 = self.up1(u1)
        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=False)
        u2 = torch.cat([u2, d2], 1)
        u2 = self.up2(u2)
        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=False)
        u3 = torch.cat([u3, d1], 1)
        u3 = self.up3(u3)
        return self.out(u3)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
n_epochs = 100
net = ClassConditionedUViT().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
net_ema = ClassConditionedUViT().to(device)
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
        ddim.set_timesteps(60)
        for i, t in tqdm(enumerate(ddim.timesteps)):
            with torch.no_grad():
                net_ema.eval()
                residual_cond = net_ema(xg, t.to(xg.device), yg)
                drop_mask = torch.rand_like(yg.float()) < 0.1
                yg_uncond = yg.clone()
                yg_uncond[drop_mask] = 0
                residual_uncond = net_ema(xg, t.to(xg.device), yg_uncond)
                guidance_scale = 2.0
                residual = residual_uncond + guidance_scale * (residual_cond - residual_uncond)
            xg = ddim.step(residual, t, xg).prev_sample
        out_name = os.path.join(os.path.dirname(__file__), f'samples_uvit_bw_epoch{epoch+1}.png')
        torchvision.utils.save_image(
            xg.detach().cpu().clip(-1, 1),
            out_name,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
plt.plot(losses)

# x = torch.randn(8 * 2, 1, img_sz[0], img_sz[1]).to(device)
# y = torch.tensor([[i] * 8 for i in range(2)]).flatten().to(device)
# ddim = DDIMScheduler.from_config(noise_scheduler.config)
# ddim.set_timesteps(30)
# for i, t in tqdm(enumerate(ddim.timesteps)):
#     with torch.no_grad():
#         residual = net(x, t.to(x.device), y)
#     x = ddim.step(residual, t, x).prev_sample
# fig, ax = plt.subplots(1, 1, figsize=(144, 144))
# imgg = torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)
# ax.imshow(imgg[0], cmap='Greys')
# torchvision.utils.save_image(
#     x.detach().cpu().clip(-1, 1),
#     os.path.join(os.path.dirname(__file__), 'samples_uvit_bw.png'),
#     nrow=8,
#     normalize=True,
#     value_range=(-1, 1)
# )
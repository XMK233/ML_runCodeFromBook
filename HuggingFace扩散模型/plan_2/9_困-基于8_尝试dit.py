import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import math
import numpy as np

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

import os
# import torch
from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

import os
from torchvision import datasets, transforms
import copy
from PIL import Image, UnidentifiedImageError

class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None,
                 include_prefixes=None,             # 例如 ['2','3',...,'22']
                 only_jpg_with_underscore=False):   # 仅文件名含 '_' 的 .jpg
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

        # 可选：只保留有样本的类别并重映射 targets（保证 classes 与实际样本一致）
        present_classes = sorted({os.path.basename(os.path.dirname(p)) for p, _ in kept})
        class_to_idx = {cls: i for i, cls in enumerate(present_classes)}
        remapped = [(p, class_to_idx[os.path.basename(os.path.dirname(p))]) for p, _ in kept]

        self.classes = present_classes
        self.class_to_idx = class_to_idx
        self.samples = remapped
        self.imgs = remapped
        self.targets = [t for _, t in remapped]
        print(f"SafeImageFolder: kept={len(remapped)} skipped={skipped} classes={len(self.classes)}")

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

# 路径
root = "/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志"
img_sz = (128, 128)

# 与 MNIST 类似的变换（返回 C×H×W 的 float 张量）
crop_x, crop_y, crop_w, crop_h = 0, 0, 800, 800 ## 400, 550 
## 如果不想裁切，后面两个设置很大的数就行了。
# 0, 0, 256, 256
def crop_im(im):
    W, H = im.size
    x0 = max(0, min(W - 1, crop_x))
    y0 = max(0, min(H - 1, crop_y))
    x1 = max(x0 + 1, min(W, x0 + crop_w))
    y1 = max(y0 + 1, min(H, y0 + crop_h))
    return im.crop((x0, y0, x1, y1))

tf = transforms.Compose([
    transforms.Lambda(crop_im),
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# 仅加载 2..22 目录、带下划线的 .jpg，且自动跳过坏图
include = [str(i) for i in range(2, 23)]
dataset = SafeImageFolder(
    root=root,
    transform=tf,
    include_prefixes=include,
    only_jpg_with_underscore=True
)

p0 = dataset.samples[0][0]
im = Image.open(p0)
print('Original size:', im.size)
im_c = crop_im(im)
print('Cropped original size:', im_c.size)
im.save(os.path.join(os.path.dirname(__file__), 'dataset_sample_original.png'))
im_c.save(os.path.join(os.path.dirname(__file__), 'dataset_sample_cropped_original_size.png'))

# 取一个样本（形式与 MNIST 一样）
img, label = dataset[0]
print(img.shape, label)  # 例如 torch.Size([3, 128, 128]), label 为 int
print("classes:", dataset.classes)
torchvision.utils.save_image(
    img.unsqueeze(0),
    os.path.join(os.path.dirname(__file__), 'dataset_sample_after_transform.png'),
    normalize=True,
    value_range=(-1, 1)
)

# 创建数据加载器
train_dataloader = DataLoader(dataset, batch_size=4,shuffle=True)

x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        m = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, m), nn.GELU(), nn.Linear(m, hidden_size))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ClassConditionedDiT(nn.Module):
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4, hidden_size=256, num_layers=6, num_heads=4, patch_size=8):
        super().__init__()
        self.sample_size = (img_sz[0], img_sz[1])
        self.in_channels = 1
        self.out_channels = 1
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.class_proj = nn.Linear(class_emb_size, hidden_size)
        self.patch_embed = nn.Conv2d(self.in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        gh = self.sample_size[0] // patch_size
        gw = self.sample_size[1] // patch_size
        self.register_buffer('pos_embed', self._build_2d_sincos(gh, gw, hidden_size).unsqueeze(0), persistent=False)
        self.time_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 4), nn.GELU(), nn.Linear(hidden_size * 4, hidden_size))
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Conv2d(hidden_size, self.out_channels, kernel_size=1)
        self._init_weights()

    def _build_2d_sincos(self, h, w, dim):
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

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_time_embedding(self, timesteps):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = torch.tensor(timesteps, device=self.pos_embed.device)
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        half = self.hidden_size // 2
        emb = np.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.hidden_size % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, x, t, class_labels):
        t = t.to(x.device)
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)
        c_emb = self.class_proj(self.class_emb(class_labels.to(x.device)))
        h = self.patch_embed(x)
        h = h.flatten(2).transpose(1, 2)
        h = h + self.pos_embed
        h = h + t_emb.unsqueeze(1)
        h = h + c_emb.unsqueeze(1)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        H = self.sample_size[0] // self.patch_size
        W = self.sample_size[1] // self.patch_size
        h = h.transpose(1, 2).reshape(-1, self.hidden_size, H, W)
        h = F.interpolate(h, size=self.sample_size, mode='bilinear', align_corners=False)
        y = self.proj_out(h)
        return y

# 创建一个调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
n_epochs = 150 
net = ClassConditionedDiT().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
net_ema = copy.deepcopy(net).to(device)
ema_decay = 0.999
steps_per_epoch = len(train_dataloader)
total_steps = max(1, steps_per_epoch * n_epochs)
warmup = max(1, int(0.1 * total_steps))
min_ratio = 0.1
def _lr_lambda(step):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total_steps - warmup)
    c = 0.5 * (1 + math.cos(math.pi * p))
    return min_ratio + (1 - min_ratio) * c
scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
best_loss = float('inf')
epochs_no_improve = 0

losses = [] # 记录损失值
# 训练开始
for epoch in range(n_epochs):
    epoch_loss_sum = 0.0
    epoch_batches = 0
    for x, y in tqdm(train_dataloader):
        # 获取数据并添加噪声
        x = x.to(device) * 2 - 1 # 数据被归一化到区间(-1, 1)
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999,(x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise,timesteps)
        # 预测
        pred = net(noisy_x, timesteps, y)
        # v-预测 + SNR 加权损失
        velocity = noise_scheduler.get_velocity(x, noise, timesteps)
        alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
        snr = alphas / (1 - alphas)
        weight = torch.sqrt(torch.clamp(snr, max=5.0))
        loss = (weight * (pred - velocity) ** 2).mean()
        # 梯度回传，参数更新
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for p, p_ema in zip(net.parameters(), net_ema.parameters()):
                p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
        scheduler.step()
        # 保存损失值
        losses.append(loss.item())
        epoch_loss_sum += loss.item()
        epoch_batches += 1
    # 输出损失值
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
    print(f'Current LR: {opt.param_groups[0]["lr"]:.6f}')
    epoch_avg = epoch_loss_sum / max(1, epoch_batches)
    if epoch_avg >= best_loss - 1e-6:
        epochs_no_improve += 1
    else:
        best_loss = epoch_avg
        epochs_no_improve = 0
    if epochs_no_improve >= 3:
        for g in opt.param_groups:
            g['lr'] *= 0.7
            print("loss shrinked")
        epochs_no_improve = 0
    if (epoch + 1) % 5 == 0:
        xg = torch.randn(8*2, 1, img_sz[0], img_sz[1]).to(device)
        yg = torch.tensor([[i]*8 for i in range(2)]).flatten().to(device)
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(30)
        for i, t in tqdm(enumerate(ddim.timesteps)):
            with torch.no_grad():
                net_ema.eval()
                residual = net_ema(xg, t.to(xg.device), yg)
            xg = ddim.step(residual, t, xg).prev_sample
        fig, ax = plt.subplots(1, 1, figsize=(144, 144))
        imgg = torchvision.utils.make_grid(xg.detach().cpu().clip(-1,1),nrow=8)
        ax.imshow(imgg[0], cmap='Greys')
        out_name = os.path.join(os.path.dirname(__file__), f'samples_ddim_bw_epoch{epoch+1}-copy1.png')
        torchvision.utils.save_image(
            xg.detach().cpu().clip(-1,1),
            out_name,
            nrow=8,
            normalize=True,
            value_range=(-1, 1)
        )
# 可视化训练损失，效果如图5-17所示
plt.plot(losses)

pass

## 不裁剪/n_epochs = 50轮/g['lr'] *= 0.7
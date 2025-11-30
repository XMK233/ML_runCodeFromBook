import argparse
_arg_parser = argparse.ArgumentParser(add_help=False)
_arg_parser.add_argument('--n_epochs', type=int, default=100)
_args, _unknown = _arg_parser.parse_known_args()

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
_base = os.path.splitext(os.path.basename(__file__))[0]
_save_path = os.path.join(os.path.dirname(__file__), f"{_base}_model.pth")
n_epochs = _args.n_epochs

# 与 MNIST 类似的变换（返回 C×H×W 的 float 张量）
crop_x, crop_y, crop_w, crop_h = 50, 100, 400, 550 
## 如果不想裁切，后面两个设置很大的数就行了。
# 0, 0, 256, 256
def crop_im(im):
    W, H = im.size
    x0 = max(0, min(W - 1, crop_x))
    y0 = max(0, min(H - 1, crop_y))
    x1 = max(x0 + 1, min(W, x0 + crop_w))
    y1 = max(y0 + 1, min(H, y0 + crop_h))
    return im.crop((x0, y0, x1, y1))

img_sz = (256, 256)
tf = transforms.Compose([
    transforms.Lambda(crop_im),
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
    """
    MoE 版本的 ClassConditionedUViT：
    - 下采样编码器共享。
    - 在瓶颈的 ViT 序列处使用多个专家，每个专家处理不同的内容风格。
    - 门控网络使用类别嵌入、时间步嵌入和全局特征决定专家权重。
    其他接口保持不变。
    """
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4, vit_dim=256, vit_heads=4, vit_layers=6, num_experts=4, time_emb_size=32):
        super().__init__()
        # 条件嵌入
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.t_emb = nn.Embedding(1000, time_emb_size)

        # 编码器（共享）
        in_ch = 1 + class_emb_size
        self.down1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU()
        )
        self.pool1 = nn.Conv2d(64, 64, 3, 2, 1)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.pool2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.pool3 = nn.Conv2d(128, 128, 3, 2, 1)

        # ViT 输入投影（共享）
        self.vproj_in = nn.Conv2d(128, vit_dim, 1)

        # MoE 专家集合：每个专家一套 ViTBlock + LayerNorm + 输出投影
        self.num_experts = num_experts
        experts = []
        for _ in range(num_experts):
            expert = nn.ModuleDict({
                'blocks': nn.ModuleList([ViTBlock(vit_dim, vit_heads) for _ in range(vit_layers)]),
                'norm': nn.LayerNorm(vit_dim),
                'proj_out': nn.Conv2d(vit_dim, 128, 1)
            })
            experts.append(expert)
        self.experts = nn.ModuleList(experts)

        # 门控网络：基于类别嵌入 + 时间步嵌入 + 全局特征
        gate_in_dim = 128 + class_emb_size + time_emb_size
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, 128), nn.GELU(),
            nn.Linear(128, num_experts)
        )

        # 解码器（共享）
        self.up1 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.GroupNorm(16, 128), nn.GELU()
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GroupNorm(8, 64), nn.GELU()
        )
        self.out = nn.Conv2d(64, 1, 1)

        # 预计算位置编码（与原实现保持一致）
        gh = img_sz[0] // 8
        gw = img_sz[1] // 8
        pe = build_2d_sincos(gh, gw, vit_dim)
        self.register_buffer('pos_embed', pe.unsqueeze(0), persistent=False)

    def forward(self, x, t, class_labels):
        bs, _, h, w = x.shape
        # 类别条件拼接到输入
        c2d = self.class_emb(class_labels).view(bs, -1, 1, 1).expand(bs, -1, h, w)
        z = torch.cat([x, c2d], 1)

        # 编码器
        d1 = self.down1(z)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # ViT tokens
        vb = self.vproj_in(p3)
        gh = vb.shape[2]
        gw = vb.shape[3]
        s = vb.flatten(2).transpose(1, 2)
        s = s + self.pos_embed[:, : gh * gw, :].to(s.device)

        # 门控权重（per-sample）
        if t.dim() == 0:
            t = t.view(1).repeat(bs)
        t = t.long().clamp(0, 999)
        t_feat = self.t_emb(t)  # [bs, time_emb_size]
        c_feat = self.class_emb(class_labels)  # [bs, class_emb_size]
        p3_pool = F.adaptive_avg_pool2d(p3, 1).view(bs, 128)  # [bs, 128]
        gate_in = torch.cat([p3_pool, c_feat, t_feat], dim=1)
        gate_logits = self.gate(gate_in)
        gate_w = torch.softmax(gate_logits, dim=-1)  # [bs, num_experts]
        # 记录门控权重供训练阶段的负载均衡辅助损失使用
        # 注意：不 detach，让该辅助损失反向传播到门控网络，促进均衡路由
        self.last_gate_w = gate_w

        # 专家前向并加权融合
        yb_sum = None
        for e, expert in enumerate(self.experts):
            se = s
            for blk in expert['blocks']:
                se = blk(se)
            se = expert['norm'](se)
            se = se.transpose(1, 2).reshape(bs, -1, gh, gw)
            yb_e = expert['proj_out'](se)  # [bs, 128, gh, gw]
            w_e = gate_w[:, e].view(bs, 1, 1, 1)
            yb_e = yb_e * w_e
            yb_sum = yb_e if yb_sum is None else (yb_sum + yb_e)

        # 解码器
        u1 = F.interpolate(yb_sum, scale_factor=2, mode='bilinear', align_corners=False)
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
net = ClassConditionedUViT().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
net_ema = ClassConditionedUViT().to(device)
# 若存在已保存的模型文件，则加载并继续训练
if os.path.exists(_save_path):
    ckpt = torch.load(_save_path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            res = net.load_state_dict(ckpt['model_state_dict'], strict=False)
        if 'ema_state_dict' in ckpt:
            net_ema.load_state_dict(ckpt['ema_state_dict'], strict=False)
        else:
            net_ema.load_state_dict(net.state_dict())
        print(f"已检测到现有模型，加载并续训: {_save_path}")
    else:
        # 兼容直接保存整个 state_dict 的情况
        try:
            res = net.load_state_dict(ckpt, strict=False)
            net_ema.load_state_dict(ckpt, strict=False)
            print(f"已从原始 state_dict 加载并续训: {_save_path}")
        except Exception as e:
            print(f"加载模型失败，改为从头开始训练: {e}")
            net_ema.load_state_dict(net.state_dict())
else:
    net_ema.load_state_dict(net.state_dict())
ema_decay = 0.999

# losses = []
# for epoch in range(n_epochs):
#     for x, y in tqdm(train_dataloader):
#         x = x.to(device) * 2 - 1
#         y = y.to(device)
#         noise = torch.randn_like(x)
#         timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
#         noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
#         pred = net(noisy_x, timesteps, y)
#         velocity = noise_scheduler.get_velocity(x, noise, timesteps)
#         alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
#         snr = alphas / (1 - alphas)
#         weight = torch.sqrt(torch.clamp(snr, max=5.0))
#         # 基础损失：v-pred + SNR 加权
#         loss = (weight * (pred - velocity) ** 2).mean()

#         # 负载均衡 Auxiliary Loss（门控的均衡正则）
#         # 原理：统计一个 batch 中各专家的平均路由概率分布 mean_w，
#         #      通过将 mean_w 与均匀分布 U 之间的 KL 距离作为正则项，
#         #      逼近 mean_w → U，从而让专家使用更均衡，避免某些专家过载或长期闲置。
#         # 功效：提升训练稳定性与泛化能力，减少单专家过拟合。
#         lb_coef = 0.01  # 负载均衡损失权重，可按需调整
#         if hasattr(net, 'last_gate_w') and net.last_gate_w is not None:
#             mean_w = net.last_gate_w.mean(dim=0)  # [num_experts]
#             uniform = torch.full_like(mean_w, 1.0 / net.num_experts)
#             aux_loss = F.kl_div(mean_w.log(), uniform, reduction='batchmean')
#             loss = loss + lb_coef * aux_loss
#         opt.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
#         opt.step()
#         with torch.no_grad():
#             for p, p_ema in zip(net.parameters(), net_ema.parameters()):
#                 p_ema.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)
#         losses.append(loss.item())
#     avg_loss = sum(losses[-100:]) / 100
#     print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
#     if (epoch + 1) % 5 == 0:
#         xg = torch.randn(3 * 2, 1, img_sz[0], img_sz[1]).to(device)
#         yg = torch.tensor([[i] * 3 for i in range(2)]).flatten().to(device)
#         ddim = DDIMScheduler.from_config(noise_scheduler.config)
#         ddim.set_timesteps(60)
#         for i, t in tqdm(enumerate(ddim.timesteps)):
#             with torch.no_grad():
#                 net_ema.eval()
#                 residual_cond = net_ema(xg, t.to(xg.device), yg)
#                 drop_mask = torch.rand_like(yg.float()) < 0.1
#                 yg_uncond = yg.clone()
#                 yg_uncond[drop_mask] = 0
#                 residual_uncond = net_ema(xg, t.to(xg.device), yg_uncond)
#                 guidance_scale = 2.0
#                 residual = residual_uncond + guidance_scale * (residual_cond - residual_uncond)
#             xg = ddim.step(residual, t, xg).prev_sample
#         out_name = os.path.join(os.path.dirname(__file__), f'samples_uvit_bw_epoch{epoch+1}.png')
#         torchvision.utils.save_image(
#             xg.detach().cpu().clip(-1, 1),
#             out_name,
#             nrow=8,
#             normalize=True,
#             value_range=(-1, 1)
#         )
# plt.plot(losses)

# 保存模型，文件名基于 __file__
torch.save({
    "model_state_dict": net.state_dict(),
    "ema_state_dict": net_ema.state_dict()
}, _save_path)
print(f"模型已保存到: {_save_path}")

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
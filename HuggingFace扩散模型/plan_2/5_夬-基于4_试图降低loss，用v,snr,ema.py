import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

import os
# import torch
from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError

import os
from torchvision import datasets, transforms
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
img_sz = (64, 64)

# 与 MNIST 类似的变换（返回 C×H×W 的 float 张量）
tf = transforms.Compose([
    transforms.Resize((img_sz[0], img_sz[1])),
    transforms.Grayscale(num_output_channels=1),

    ## 【改进】数据增广（灰度）
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05)),

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

# 取一个样本（形式与 MNIST 一样）
img, label = dataset[0]
print(img.shape, label)  # 例如 torch.Size([3, 128, 128]), label 为 int
print("classes:", dataset.classes)

# 创建数据加载器
train_dataloader = DataLoader(dataset, batch_size=2,shuffle=True)

x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4):
        super().__init__()
        # 这个网络层会把数字所属的类别映射到一个长度为class_emb_size的特征向量上
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # self.model是一个不带生成条件的UNet模型，在这里，我们给它添加了额外的
        # 输入通道，用于接收条件信息
        self.model = UNet2DModel(
            sample_size=img_sz[0],
            in_channels=1 + class_emb_size,
            out_channels=1,
            layers_per_block=2, # 设置一个UNet模块有多少个残差连接层
            block_out_channels=(
                32, 
                64, 
                64
            ),
            down_block_types=(
                "DownBlock2D", # 常规的ResNet下采样模块
                "AttnDownBlock2D", # 含有spatial self-attention的
                "AttnDownBlock2D", # ResNet下采样模块
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D", # 含有spatial self-attention的ResNet上采样模块
                "UpBlock2D", # 常规的ResNet下采样模块
            ),
        )
    
    # 此时扩散模型的前向计算就会含有额外的类别标签作为输入了
    def forward(self, x, t, class_labels):
        bs, ch, w, h = x.shape
        # 类别条件将会以额外通道的形式输入
        class_cond = self.class_emb(class_labels) # 将类别映射为向量形式，
        # 并扩展成类似于(bs, 4, 28, 28)的张量形状
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # 将原始输入和类别条件信息拼接到一起
        net_input = torch.cat((x, class_cond), 1)
        # 使用模型进行预测
        # print("xxxxxxxxxyy", x.shape, net_input.shape, class_labels.shape)
        return self.model(net_input, t).sample # (bs, 1, 28, 28)

# 创建一个调度器
# v-prediction：将模型目标改为速度（v），配合 SNR 加权提升稳定性
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2', prediction_type='v_prediction')
n_epochs = 10
net = ClassConditionedUnet().to(device)
# EMA 初始化：训练中维护指数滑动平均的权重，用于更稳定的采样
ema_decay = 0.9999
ema_params = [p.detach().clone() for p in net.parameters()]
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
# opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

losses = [] # 记录损失值
# 训练开始
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        # 获取数据并添加噪声
        x = x.to(device) * 2 - 1 # 数据被归一化到区间(-1, 1)
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999,(x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise,timesteps)
        # 预测
        # v-预测 + SNR 加权损失
        pred = net(noisy_x, timesteps, y)
        # 速度目标（v）：与 v-prediction 调度对应
        velocity = noise_scheduler.get_velocity(x, noise, timesteps)
        # 计算 SNR 并进行加权，抑制高噪声步的主导
        alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
        snr = alphas / (1 - alphas)
        weight = torch.sqrt(torch.clamp(snr, max=5.0))
        loss = (weight * (pred - velocity) ** 2).mean()
        # 梯度回传，参数更新
        opt.zero_grad()
        loss.backward()
        opt.step()
        # EMA 权重更新：ep = decay * ep + (1 - decay) * p
        with torch.no_grad():
            for p, ep in zip(net.parameters(), ema_params):
                ep.mul_(ema_decay).add_(p.detach(), alpha=1 - ema_decay)
        # 保存损失值
        losses.append(loss.item())
    # 输出损失值
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
# 可视化训练损失，效果如图5-17所示
plt.plot(losses)

# 准备一个随机噪声作为起点，并准备我们想要的图片的标签
x = torch.randn(
    8*2, ## 就两个类别就好了，不要多。 len(dataset.classes)
    1, img_sz[0], img_sz[1]
).to(device)
y = torch.tensor(
    [[i]*8 for i in range(2)] ## 就两类，不要多了。
).flatten().to(device)
# 使用 DDIM 加速采样
with torch.no_grad():
    # 采样前加载 EMA 权重，提升生成稳定性与质量
    for p, ep in zip(net.parameters(), ema_params):
        p.copy_(ep)
# 使用 DDIM 加速采样（30 步）
ddim = DDIMScheduler.from_config(noise_scheduler.config)
ddim.set_timesteps(30)
for i, t in tqdm(enumerate(ddim.timesteps)):
    with torch.no_grad():
        residual = net(x, t, y)
    x = ddim.step(residual, t, x).prev_sample
# 显示结果，如图5-18所示
fig, ax = plt.subplots(1, 1, figsize=(144, 144))
imgg = torchvision.utils.make_grid(x.detach().cpu().clip(-1,1),nrow=8)
ax.imshow(imgg[0], cmap='Greys')
torchvision.utils.save_image(
    x.detach().cpu().clip(-1,1),
    os.path.join(os.path.dirname(__file__), 'samples_ddim_bw.png'),
    nrow=8,
    normalize=True,
    value_range=(-1, 1)
)
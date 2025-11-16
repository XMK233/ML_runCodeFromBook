## 基于裁剪过的图片。

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DConditionModel, DDIMScheduler
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
train_dataloader = DataLoader(dataset, batch_size=2,shuffle=True)

x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=len(dataset.classes), class_emb_size=4):
        super().__init__()
        self.class_emb = nn.Embedding(num_classes, class_emb_size)
        self.model = UNet2DConditionModel(
            sample_size=img_sz[0],
            in_channels=1,
            out_channels=1,
            layers_per_block=3,
            block_out_channels=(64, 128, 128),
            down_block_types=(
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D",
                "CrossAttnUpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=class_emb_size,
        )
    
    # 此时扩散模型的前向计算就会含有额外的类别标签作为输入了
    def forward(self, x, t, class_labels):
        cond = self.class_emb(class_labels).unsqueeze(1)
        return self.model(x, t, encoder_hidden_states=cond).sample

# 创建一个调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000,beta_schedule='squaredcos_cap_v2')
n_epochs = 50 
net = ClassConditionedUnet().to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
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
        pred = net(noisy_x, timesteps, y) # 注意这里也输入了类别标签y
        # 计算损失值
        loss = loss_fn(pred, noise) # 判断预测结果和实际的噪声有多接近
        # 梯度回传，参数更新
        opt.zero_grad()
        loss.backward()
        opt.step()
        # 保存损失值
        losses.append(loss.item())
        epoch_loss_sum += loss.item()
        epoch_batches += 1
    # 输出损失值
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
    epoch_avg = epoch_loss_sum / max(1, epoch_batches)
    if epoch_avg >= best_loss - 1e-6:
        epochs_no_improve += 1
    else:
        best_loss = epoch_avg
        epochs_no_improve = 0
    if epochs_no_improve >= 3:
        for g in opt.param_groups:
            g['lr'] *= 0.5 
            print("loss shrinked")
        epochs_no_improve = 0
    if (epoch + 1) % 5 == 0:
        xg = torch.randn(8*2, 1, img_sz[0], img_sz[1]).to(device)
        yg = torch.tensor([[i]*8 for i in range(2)]).flatten().to(device)
        ddim = DDIMScheduler.from_config(noise_scheduler.config)
        ddim.set_timesteps(30)
        for i, t in tqdm(enumerate(ddim.timesteps)):
            with torch.no_grad():
                residual = net(xg, t, yg)
            xg = ddim.step(residual, t, xg).prev_sample
        fig, ax = plt.subplots(1, 1, figsize=(144, 144))
        imgg = torchvision.utils.make_grid(xg.detach().cpu().clip(-1,1),nrow=8)
        ax.imshow(imgg[0], cmap='Greys')
        out_name = os.path.join(os.path.dirname(__file__), f'samples_ddim_bw_epoch{epoch+1}.png')
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
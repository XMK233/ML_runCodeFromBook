import torch
import torchvision
from torch import nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, DDPMPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HanfuDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 预训练模型分辨率为 128x128，这里将你的数据适配到 128x128
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = HanfuDataset("/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志/14 士庶巾服/", transform)
train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 从 HuggingFace 加载预训练动漫 DDPM 管线
pretrained_id = "mrm8488/ddpm-ema-anime-v2-128"
pipe = DDPMPipeline.from_pretrained(pretrained_id, cache_dir="/mnt/d/HuggingFaceModels/")
pipe = pipe.to(device)

# 取出预训练的 UNet 与调度器作为初始化
net: UNet2DModel = pipe.unet
net.to(device)

noise_scheduler: DDPMScheduler = pipe.scheduler

# 训练微调：保持 scheduler 配置，微调 UNet 参数
loss_fn = nn.MSELoss()
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(5):
    for x in train_dataloader:
        x = x.to(device)
        noise = torch.randn_like(x)
        bs = x.shape[0]
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=x.device).long()
        noisy_images = noise_scheduler.add_noise(x, noise, timesteps)
        noise_pred = net(noisy_images, timesteps, return_dict=False)[0]
        loss = loss_fn(noise_pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()

# 采样用 DDIM 加速
ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
pipe.scheduler = ddim_scheduler

gen = torch.Generator(device=device).manual_seed(42)
images = pipe(num_inference_steps=30, generator=gen).images
images[0].save("sample_ddim_ft.png")
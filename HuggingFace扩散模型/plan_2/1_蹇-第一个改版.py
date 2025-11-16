import torch
import torchvision
from torch import nn
import random, os, tqdm, time, json
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import timm

device = torch.device("cuda")
print(f'Using device: {device}')

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    plt.imshow(grid_im)
    
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

transform = transforms.Compose([  
    transforms.Resize((160, 100)), 
    transforms.CenterCrop((128, 64)),  # 原先是(128, 64)
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])

dataset = HanfuDataset("/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志/14 士庶巾服/", transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)  

## 我们自己实现一遍corrupt：
def corrupt(x, amount):
    '''
    实际上是不同的图加的噪音程度不一样。
    就是说，给x加上amount程度的【正态分布noise】。
    '''
    ## 生成一个形状和x一样的正态分布随机数。
    noise = torch.randn_like(x)
    ## 接下来的部分是照抄的。
    amount = amount.view(-1, 1, 1, 1)
    return (1-amount) * x + amount * noise

# DDPM

from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler
from datetime import datetime

train_dataloader = data_loader

for x in train_dataloader:
    break

show_images(x)

# How many runs through the data should we do?
n_epochs = 3

net = UNet2DModel(
    sample_size=(128, 64),  # 提高分辨率 原来是(128,64)其实就已经有一定的效果了。
    in_channels=3,  # 输入通道数，RGB 图像为 3
    out_channels=3,  # 输出通道数
    layers_per_block=2,  # 每个 UNet 块使用的 ResNet 层数
    block_out_channels=(64, 128, 256, 512),  # 增加通道数以增加参数
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",  # 增加注意力块
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # 增加注意力块
        "UpBlock2D",
    ),
)
net.to(device)

convnext = timm.create_model(
                "convnext_tiny", pretrained=True, num_classes=0,
                cache_dir="/mnt/d/HuggingFaceModels/",
                pretrained_cfg_overlay={'file': '/mnt/d/HuggingFaceModels/models--timm--convnext_tiny.in12k_ft_in1k/snapshots/aa096f03029c7f0ec052013f64c819b34f8ad790/model.safetensors'}
            )
convnext.to(device)
convnext.eval()
for p in convnext.parameters():
    p.requires_grad = False

sum([p.numel() for p in net.parameters()])

from diffusers import DDPMScheduler
    
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# Our loss finction
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-5) 

def to_imagenet_norm(x):
    x = x * 0.5 + 0.5
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std

lambda_perc = 0.05

# Keeping a record of the losses for later viewing
losses = []

# The training loop
# with TimerContext():
for epoch in range(100):
    for x in train_dataloader:
        # Get some data and prepare the corrupted version
        x = x.to(device) # Data on the GPU
        noise = torch.randn(x.shape).to(x.device)
        bs = x.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, 
            noise_scheduler.num_train_timesteps, ## 前两个参数指的是生成数组的取值范围？
            (bs,), ## 这个是数组的长度？
            device=x.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(x, noise, timesteps)

        noise_pred = net(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss_noise = loss_fn(noise_pred, noise)
        alphas = noise_scheduler.alphas_cumprod[timesteps].to(x.device).view(-1, 1, 1, 1)
        x0_hat = (noisy_images - torch.sqrt(1 - alphas) * noise_pred) / torch.sqrt(alphas)
        feat_real = convnext(to_imagenet_norm(x))
        feat_fake = convnext(to_imagenet_norm(x0_hat))
        loss_perc = loss_fn(feat_fake, feat_real.detach())
        loss = loss_noise + lambda_perc * loss_perc

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the loss values for this epoch:
    avg_loss = sum(losses[-len(train_dataloader):])/len(train_dataloader)
    print(f'Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}')

from diffusers import DDPMPipeline
ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
image_pipe = DDPMPipeline(unet=net, scheduler=ddim_scheduler)

# 保存DDPM模型
model_save_path = "./saved_ddpm_model"
image_pipe.save_pretrained(model_save_path)
print(f"DDPM模型已成功保存到{model_save_path}目录")

# # 加载DDPM模型（示例代码）
# print("\n加载已保存的DDPM模型...")
# loaded_pipe = DDPMPipeline.from_pretrained(model_save_path)
# print(f"DDPM模型已成功从{model_save_path}目录加载")

image_pipe.to(device)
image_pipe.set_progress_bar_config(disable=True)
gen = torch.Generator(device=device).manual_seed(42)
pipeline_output = image_pipe(num_inference_steps=30, generator=gen)
pipeline_output.images[0]
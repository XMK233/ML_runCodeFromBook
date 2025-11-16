import torch
import torchvision
from torch import nn
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import DDPMScheduler, UNet2DModel, DDIMScheduler, DDPMPipeline
import tqdm

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

transform = transforms.Compose([
    transforms.Resize((160, 100)),
    transforms.CenterCrop((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = HanfuDataset("/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志/14 士庶巾服/", transform)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

net = UNet2DModel(
    sample_size=(128, 64),
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(64, 128, 256, 512),
    down_block_types=(
        "DownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
    ),
)
net.to(device)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-4)

for epoch in tqdm.tqdm(range(1000)):
    for x in train_dataloader:
        x = x.to(device)
        noise = torch.randn(x.shape, device=x.device)
        bs = x.shape[0]
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=x.device).long()
        noisy_images = noise_scheduler.add_noise(x, noise, timesteps)
        noise_pred = net(noisy_images, timesteps, return_dict=False)[0]
        loss = loss_fn(noise_pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()

ddim_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
image_pipe = DDPMPipeline(unet=net, scheduler=ddim_scheduler)
image_pipe.to(device)
gen = torch.Generator(device=device).manual_seed(42)
images = image_pipe(num_inference_steps=30, generator=gen).images
images[0].save("sample_ddim.png")
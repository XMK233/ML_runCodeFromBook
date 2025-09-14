import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

# 设置随机种子以确保可重复性
random.seed(618)
np.random.seed(907)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义数据路径
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "segmentation_data")
OUTPUT_PATH = os.path.join(BASE_PATH, "segmentation_output")
MODEL_PATH = os.path.join(BASE_PATH, "segmentation_models")

# 创建必要的文件夹
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# 定义UNet架构中的基本构建块
class DoubleConv(nn.Module):
    """两次卷积操作（Conv2d -> BatchNorm -> ReLU）的组合"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

# 定义UNet架构
class UNet(nn.Module):
    """基于经典UNet架构的图像分割模型"""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 下采样路径（编码器）
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
        
        # 上采样路径（解码器）
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # 瓶颈层
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # 输出层
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []
        
        # 下采样过程，保存跳跃连接
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 反转跳跃连接列表，以便在解码器中使用
        skip_connections = skip_connections[::-1]
        
        # 上采样过程，结合跳跃连接
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            
            # 如果输入尺寸不匹配，调整跳跃连接的尺寸
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)
            
            # 连接特征图
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        
        # 输出分割图
        return self.final_conv(x)

# 自定义数据集类
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace("image", "mask"))
        
        # 加载图像和掩码
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 转为灰度图
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            mask = (mask > 0.5).float()  # 二值化
        
        return image, mask

# 定义数据变换
def get_transforms(img_size=(256, 256)):
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

# 定义IoU（Intersection over Union）指标计算函数
def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    
    if union == 0:
        return 0.0  # 避免除以零
    
    return intersection / union

# 定义训练函数
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=50, patience=5):
    best_val_iou = 0.0
    early_stopping_counter = 0
    
    train_losses = []
    val_losses = []
    val_ious = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        running_val_loss = 0.0
        running_val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, masks = images.to(device), masks.to(device)
                
                # 前向传播
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                # 计算IoU
                iou = calculate_iou(outputs, masks)
                
                running_val_loss += loss.item() * images.size(0)
                running_val_iou += iou * images.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_iou = running_val_iou / len(val_loader.dataset)
        
        val_losses.append(epoch_val_loss)
        val_ious.append(epoch_val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {epoch_train_loss:.4f}")
        print(f"  验证损失: {epoch_val_loss:.4f}")
        print(f"  验证IoU: {epoch_val_iou:.4f}")
        
        # 早停机制
        if epoch_val_iou > best_val_iou:
            best_val_iou = epoch_val_iou
            torch.save(model.state_dict(), os.path.join(MODEL_PATH, "best_unet_model.pth"))
            early_stopping_counter = 0
            print(f"  模型已保存，最佳IoU: {best_val_iou:.4f}")
        else:
            early_stopping_counter += 1
            print(f"  早停计数器: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print("早停机制触发，停止训练")
                break
    
    # 绘制训练过程图表
    plot_training_history(train_losses, val_losses, val_ious)
    
    return model

# 绘制训练历史图表
def plot_training_history(train_losses, val_losses, val_ious):
    plt.figure(figsize=(12, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # 绘制IoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_ious, label='验证IoU')
    plt.title('验证IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'training_history.png'))
    plt.close()

# 显示分割结果
def visualize_predictions(model, test_loader, num_samples=5):
    model.eval()
    samples_shown = 0
    
    plt.figure(figsize=(15, num_samples * 5))
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            # 前向传播获取预测结果
            outputs = model(images)
            preds = (outputs > 0.5).float()
            
            # 显示结果
            for i in range(images.size(0)):
                if samples_shown >= num_samples:
                    break
                
                # 反标准化图像以正确显示
                image = images[i].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = std * image + mean
                image = np.clip(image, 0, 1)
                
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()
                
                # 绘制原图、真实掩码和预测掩码
                plt.subplot(num_samples, 3, samples_shown * 3 + 1)
                plt.imshow(image)
                plt.title('原图')
                plt.axis('off')
                
                plt.subplot(num_samples, 3, samples_shown * 3 + 2)
                plt.imshow(mask, cmap='gray')
                plt.title('真实掩码')
                plt.axis('off')
                
                plt.subplot(num_samples, 3, samples_shown * 3 + 3)
                plt.imshow(pred, cmap='gray')
                plt.title('预测掩码')
                plt.axis('off')
                
                samples_shown += 1
            
            if samples_shown >= num_samples:
                break
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'segmentation_results.png'))
    plt.close()

# 创建一个简单的演示数据集（用于测试）
def create_demo_dataset():
    """创建一个简单的演示数据集用于测试图像分割算法"""
    # 创建数据集文件夹
    demo_image_dir = os.path.join(DATA_PATH, "demo_images")
    demo_mask_dir = os.path.join(DATA_PATH, "demo_masks")
    os.makedirs(demo_image_dir, exist_ok=True)
    os.makedirs(demo_mask_dir, exist_ok=True)
    
    # 创建一些简单的图像和对应的掩码
    for i in range(20):
        # 创建随机彩色图像
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # 创建对应的掩码（圆形）
        mask = np.zeros((256, 256), dtype=np.uint8)
        center_x, center_y = np.random.randint(64, 192, 2)
        radius = np.random.randint(32, 64)
        
        y, x = np.ogrid[:256, :256]
        distance = np.sqrt((x - center_x)**2 + (y - center_y)** 2)
        mask[distance <= radius] = 255
        
        # 保存图像和掩码
        Image.fromarray(image).save(os.path.join(demo_image_dir, f"image_{i}.png"))
        Image.fromarray(mask).save(os.path.join(demo_mask_dir, f"mask_{i}.png"))
    
    print(f"演示数据集已创建，包含20张图像和对应的掩码")
    return demo_image_dir, demo_mask_dir

# 主函数
def main():
    print("=== 图像分割算法实现 ===")
    
    # 创建演示数据集
    print("1. 创建演示数据集...")
    image_dir, mask_dir = create_demo_dataset()
    
    # 获取数据变换
    print("2. 准备数据变换...")
    train_transforms, val_transforms = get_transforms(img_size=(256, 256))
    
    # 创建数据集和数据加载器
    print("3. 创建数据集和数据加载器...")
    # 简单划分训练集和验证集（80%训练，20%验证）
    all_images = os.listdir(image_dir)
    train_size = int(0.8 * len(all_images))
    
    train_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transforms
    )
    
    val_dataset = SegmentationDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=val_transforms
    )
    
    # 使用随机采样器划分数据集
    train_indices = torch.randperm(len(train_dataset))[:train_size]
    val_indices = torch.randperm(len(val_dataset))[train_size:]
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        sampler=train_sampler,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        sampler=val_sampler,
        num_workers=2
    )
    
    # 创建模型
    print("4. 创建UNet模型...")
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    
    # 定义损失函数和优化器
    print("5. 定义损失函数和优化器...")
    criterion = nn.BCEWithLogitsLoss()  # 二元交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练模型
    print("6. 开始训练模型...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=20,
        patience=3
    )
    
    # 加载最佳模型权重
    print("7. 加载最佳模型权重...")
    best_model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512]).to(device)
    best_model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "best_unet_model.pth")))
    
    # 可视化分割结果
    print("8. 可视化分割结果...")
    visualize_predictions(best_model, val_loader, num_samples=5)
    
    print("=== 图像分割算法实现完成 ===")
    print(f"结果保存在: {OUTPUT_PATH}")

# 如果作为主程序运行，则执行main函数
if __name__ == "__main__":
    main()
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, UnidentifiedImageError
import timm
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root = "/mnt/d/forCoding_data/ML_runCodeFromBook/HuggingFace扩散模型/originalData/Q版大明衣冠图志"

included = set(str(i) for i in range(2, 23))

def is_included_dir(d):
    name = os.path.basename(d)
    return name.split()[0] in included

class FilteredImageFolder(datasets.ImageFolder):
    def collect_samples(self):
        samples = []
        for cls, idx in self.class_to_idx.items():
            folder = os.path.join(self.root, cls)
            for root, _, files in os.walk(folder):
                for f in files:
                    if f.lower().endswith('.jpg') and '_' in f and not f.startswith('._'):
                        path = os.path.join(root, f)
                        try:
                            with Image.open(path) as im:
                                im.verify()
                            samples.append((path, idx))
                        except (UnidentifiedImageError, OSError):
                            continue
        return samples
    def find_classes(self, directory):
        classes = []
        for d in sorted(os.listdir(directory)):
            p = os.path.join(directory, d)
            if os.path.isdir(p) and is_included_dir(p):
                classes.append(d)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __init__(self, root, transform=None):
        super().__init__(root=root, transform=transform)
        # Override samples to only include jpg with underscore
        sel = self.collect_samples()
        self.samples = sel
        self.imgs = sel

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

dataset = FilteredImageFolder(root=root, transform=train_tf)

num_classes = len(dataset.classes)

indices = list(range(len(dataset)))
split = int(0.8 * len(indices))
train_idx, val_idx = indices[:split], indices[split:]
train_ds = torch.utils.data.Subset(dataset, train_idx)
val_ds = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

model = timm.create_model(
    "convnext_tiny", pretrained=True, num_classes=num_classes,
    cache_dir="/mnt/d/HuggingFaceModels/",
    pretrained_cfg_overlay={'file': '/mnt/d/HuggingFaceModels/models--timm--convnext_tiny.in12k_ft_in1k/snapshots/aa096f03029c7f0ec052013f64c819b34f8ad790/model.safetensors'}
)
model.to(device)

for p in model.layers[:-1] if hasattr(model, 'layers') else []:
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

def evaluate():
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().tolist())
            trues.extend(y.cpu().tolist())
    if len(trues) == 0:
        print("No validation samples.")
        return
    labels_sorted = sorted(set(trues))
    target_names = [dataset.classes[i] for i in labels_sorted]
    print(classification_report(trues, preds, labels=labels_sorted, target_names=target_names, digits=4, zero_division=0))

best_acc = 0.0
for epoch in range(10):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += x.size(0)
    train_acc = correct / total
    print(f"Epoch {epoch}: train_loss={loss_sum/total:.4f} train_acc={train_acc:.4f}")
    # val
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += x.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch}: val_acc={val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'convnext_tiny_best.pt'))

print("Final evaluation:")
evaluate()
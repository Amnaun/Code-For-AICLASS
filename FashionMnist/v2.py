# -*- coding: utf-8 -*-
"""
FashionMNIST ResNet-18 (PyTorch) - 最终版
风格重构版：按 MNIST 示例，采用平铺式脚本风格
- [模型] 改造后的 ResNet-18 架构
- [增强] 强数据增强 (Rotation, Erasing)
- [策略] AdamW + CosineAnnealingLR
- [修复] 指定 macOS 系统中文字体 (STHeiti)
"""

import os
import random
import numpy as np
import multiprocessing as mp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import datasets, transforms
from torchvision.models import resnet18

# ===================== 1. 全局设置 =====================

# --- 【修复】设置中文字体 ---
# 解决 Matplotlib 中文显示为方框的问题 (使用 macOS 自带字体)
try:
    # "华文黑体" (STHeiti) 在 macOS 上通常可用
    matplotlib.rcParams["font.family"] = "STHeiti"
    matplotlib.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
    print("字体 'STHeiti' 设置成功。")
except Exception as e:
    print(f"字体 'STHeiti' 设置失败: {e}. 尝试备选 'PingFang SC'...")
    try:
        matplotlib.rcParams["font.family"] = "PingFang SC"
        matplotlib.rcParams["axes.unicode_minus"] = False
        print("字体 'PingFang SC' 设置成功。")
    except Exception as e2:
        print(f"备选字体 'PingFang SC' 亦失败: {e2}.")
        print("所有备选字体均失败。plots 可能无法正确显示中文。")
        print("请尝试 '【终极方案】' (见代码底部注释)。")
        matplotlib.rcParams["axes.unicode_minus"] = False

# --- 随机种子 ---
torch.manual_seed(17)
random.seed(17)
np.random.seed(17)

# --- 设备 ---
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"使用设备: {device}\n")

# --- 类别 ---
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# --- 数据增强 ---
train_transform = transforms.Compose(
    [
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(15),  # 随机旋转 ±15 度
        transforms.ToTensor(),
        # 随机擦除
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0),
        transforms.Normalize((0.2861,), (0.3530,)),  # 标准化
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.2861,), (0.3530,)),
    ]
)

# --- 数据集 ---
train_dataset = datasets.FashionMNIST(
    "./dataset_fashion", train=True, download=True, transform=train_transform
)
test_dataset = datasets.FashionMNIST(
    "./dataset_fashion", train=False, download=True, transform=test_transform
)

# --- 数据加载器 ---
if device.type == "cuda":
    num_workers = 2
    pin_memory = True
    BATCH_SIZE = 128
else:
    # MPS/CPU：最稳的是 workers=0，pin_memory=False
    num_workers = 0
    pin_memory = False
    BATCH_SIZE = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=True,  # 配合 Scheduler 步数对齐
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
)

# ===================== 2. 模型定义 =====================


class ResNet18ForFashionMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载 ResNet-18 结构, weights=None 表示随机初始化
        self.model = resnet18(weights=None)

        # 1. 改造输入层: 1通道, 3x3 kernel, stride 1
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        # 2. 移除 MaxPool (对 28x28 太猛了)
        self.model.maxpool = nn.Identity()
        # 3. 改造输出层: 512 -> 10
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.model(x)


# ===================== 3. 全局实例和超参 =====================

model = ResNet18ForFashionMNIST().to(device)

print("模型结构: (ResNet18-FashionMNIST)")
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,} (ResNet-18)\n")

# --- 损失函数 (带标签平滑), 优化器, 周期 ---
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
EPOCHS = 3

# --- 调度器: 余弦退火 ---
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6
)

# ===================== 4. 训练/评估函数 (依赖全局变量) =====================


def train(epoch, train_losses, train_accs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        predicted = output.argmax(1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    train_losses.append(avg_loss)
    train_accs.append(accuracy)

    print(
        f"Epoch {epoch + 1:2d} | Train Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%",
        end="",
    )
    return avg_loss, accuracy


def evaluate(loader, test_losses, test_accs):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)

    test_losses.append(test_loss)
    test_accs.append(accuracy)

    print(f" | Test Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%")
    return accuracy


# ===================== 5. 训练主循环 =====================

best_acc = 0.0
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("开始训练 (使用 ResNet-18)...\n")

for epoch in range(EPOCHS):
    train(epoch, train_losses, train_accs)
    acc = evaluate(test_loader, test_losses, test_accs)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "fashionmnist_best_resnet18.pth")

print(f"\n训练完成, 最佳准确率: {best_acc:.2f}%\n")

# ===================== 6. 结果可视化 (全局执行) =====================

# --- 6.1 绘制学习曲线 ---
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(train_losses, label="Train Loss", linewidth=2)
ax[0].plot(test_losses, label="Test Loss", linewidth=2)
ax[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
ax[0].set_xlabel("Epoch", fontsize=12)
ax[0].set_ylabel("Loss", fontsize=12)
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot(train_accs, label="Train Acc", linewidth=2)
ax[1].plot(test_accs, label="Test Acc", linewidth=2)
ax[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
ax[1].set_xlabel("Epoch", fontsize=12)
ax[1].set_ylabel("Accuracy (%)", fontsize=12)
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# --- 6.2 载入最佳权重 ---
print("载入最佳权重 (ResNet-18) 进行评估...")
model.load_state_dict(torch.load("fashionmnist_best_resnet18.pth", map_location=device))
model.eval()

# --- 6.3 随机抽样 15 张可视化 ---
num_samples = 15
indices = random.sample(range(len(test_dataset)), num_samples)

imgs_list = [test_dataset[i][0] for i in indices]
lbls_list = [test_dataset[i][1] for i in indices]

images = torch.stack(imgs_list, dim=0).to(device)
labels = torch.tensor(lbls_list)

with torch.no_grad():
    outputs = model(images)
    pred = outputs.argmax(1).cpu()

rows, cols = 3, 5
fig, axes = plt.subplots(rows, cols, figsize=(14, 8))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= num_samples:
        ax.axis("off")
        continue
    img = images[i].detach().cpu().squeeze().numpy()
    p, t = pred[i].item(), labels[i].item()
    ok = p == t
    color = "green" if ok else "red"

    ax.imshow(img, cmap="gray")
    ax.set_title(
        f"Pred: {CLASS_NAMES[p]}\nTrue: {CLASS_NAMES[t]}",
        fontsize=11,
        color=color,
        fontweight="bold",
    )
    ax.set_xticks([])
    ax.set_yticks([])

    for side in ["left", "right", "top", "bottom"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(2.2)
        ax.spines[side].set_color(color)

plt.suptitle("随机抽样 15 张：预测 vs 真实", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# --- 6.4 混淆矩阵 & 分类报告 ---
print("\n生成混淆矩阵...")
all_preds, all_labels = [], []
model.eval()

for imgs, lbls in test_loader:
    imgs = imgs.to(device)
    out = model(imgs)
    all_preds.extend(out.argmax(1).cpu().numpy())
    all_labels.extend(lbls.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.title("混淆矩阵", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n分类报告:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

# --- 6.5 你要的“抽样对比” (来自第一个 Paddle 脚本) ---
print("\n" + "=" * 20 + " 抽样展示 (类 Paddle) " + "=" * 20)

model.eval()
indexs = [37, 24, 51, 181, 262, 388]

for idx in indexs:
    image, label = test_dataset[idx]
    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_batch)

    pred_idx = output.argmax(1).item()

    true_label = CLASS_NAMES[label]
    pred_label = CLASS_NAMES[pred_idx]

    print(
        f"第 {idx:<3} 条记录 | 真实值: {label} ({true_label:<11}) | 预测值: {pred_idx} ({pred_label:<11})"
    )

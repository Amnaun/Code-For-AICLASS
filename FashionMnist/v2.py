# -*- coding: utf-8 -*-
"""
FashionMNIST CNN (PyTorch, macOS/MPS 友好版)
- [升级] 更强的数据增强 (Rotation, Erasing)
- [升级] 使用 CosineAnnealingLR 调度器
- 训练/评估/最佳权重保存
- 学习曲线
- 随机抽测 15 张（3×5）带方框可视化
- 混淆矩阵 & 分类报告
- 指定索引抽样对比
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


# ===================== 随机种子 =====================
def set_seed(seed: int = 24):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ===================== 字体（Mac 默认 Helvetica） =====================
def setup_matplotlib_font():
    try:
        matplotlib.rcParams["font.family"] = "Helvetica"  # 如报错可改为 'Arial'
    except Exception as e:
        print(f"设置字体失败: {e}. 使用默认字体。")
    matplotlib.rcParams["axes.unicode_minus"] = False


# ===================== 模型 =====================
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28->14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14->7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7->3
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ===================== 数据增强/标准化 =====================
def build_transforms():
    # 【修改 1】: 增加 RandomRotation 和 RandomErasing
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomRotation(15),  # 随机旋转 ±15 度
            transforms.ToTensor(),  # 转为 Tensor
            # ToTensor() 之后才能用 RandomErasing
            # p=0.5: 50%概率擦除
            # scale: 擦除面积占总面积的 2% 到 20%
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0
            ),
            transforms.Normalize((0.2861,), (0.3530,)),  # 标准化
        ]
    )

    # 测试集不做增强
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2861,), (0.3530,)),
        ]
    )
    return train_transform, test_transform


# ===================== DataLoader（按设备自适配） =====================
def create_loaders(train_dataset, test_dataset, device):
    if device.type == "cuda":
        num_workers = 2
        pin_memory = True
        batch_size = 128
    else:
        # MPS/CPU：最稳的是 workers=0，pin_memory=False
        num_workers = 0
        pin_memory = False
        batch_size = 128

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1000,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader


# ===================== 训练/评估函数 =====================
def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epoch_idx,
    train_losses,
    train_accs,
):
    model.train()
    total, correct, run_loss = 0, 0, 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # CosineAnnealingLR 需要在每个 batch step 之后调用
        if scheduler is not None:
            scheduler.step()

        run_loss += loss.item()
        pred = out.argmax(1)
        total += target.size(0)
        correct += (pred == target).sum().item()

    avg = run_loss / len(train_loader)
    acc = 100.0 * correct / total
    train_losses.append(avg)
    train_accs.append(acc)
    print(f"Epoch {epoch_idx + 1:02d} | Train Loss: {avg:.4f}, Acc: {acc:.2f}%", end="")
    return avg, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, test_losses, test_accs):
    model.eval()
    total, correct, run_loss = 0, 0, 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        run_loss += criterion(out, target).item()
        pred = out.argmax(1)
        total += target.size(0)
        correct += (pred == target).sum().item()
    avg = run_loss / len(loader)
    acc = 100.0 * correct / total
    test_losses.append(avg)
    test_accs.append(acc)
    print(f" | Test Loss: {avg:.4f}, Acc: {acc:.2f}%")
    return acc


# ===================== 可视化：学习曲线 =====================
def plot_curves(train_losses, test_losses, train_accs, test_accs):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses, label="Train Loss", linewidth=2)
    ax[0].plot(test_losses, label="Test Loss", linewidth=2)
    ax[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(train_accs, label="Train Acc", linewidth=2)
    ax[1].plot(test_accs, label="Test Acc", linewidth=2)
    ax[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================== 可视化：随机抽样 15 张（3×5）带方框 =====================
def visualize_random_15(model, test_dataset, device, class_names):
    model.eval()
    num_samples = 15
    indices = random.sample(range(len(test_dataset)), num_samples)

    imgs_list = [test_dataset[i][0] for i in indices]  # Tensor [1,28,28]
    lbls_list = [test_dataset[i][1] for i in indices]  # int

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
            f"Pred: {class_names[p]}\nTrue: {class_names[t]}",
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


# ===================== 混淆矩阵 & 分类报告 =====================
@torch.no_grad()
def plot_confusion_and_report(model, test_loader, device, class_names):
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
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("预测标签", fontsize=12)
    plt.ylabel("真实标签", fontsize=12)
    plt.title("混淆矩阵", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    print("\n分类报告:")
    print(
        classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    )


# ===================================================================
# 主执行流程
# ===================================================================

set_seed(24)
setup_matplotlib_font()

# 设备优先级：MPS > CUDA > CPU
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"使用设备: {device}\n")

# 类别名称
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

# 数据与加载器
train_tf, test_tf = build_transforms()
train_dataset = datasets.FashionMNIST(
    "./dataset_fashion", train=True, download=True, transform=train_tf
)
test_dataset = datasets.FashionMNIST(
    "./dataset_fashion", train=False, download=True, transform=test_tf
)
train_loader, test_loader = create_loaders(train_dataset, test_dataset, device)

# 模型、损失、优化器
model = ImprovedCNN().to(device)
print("模型结构:\n", model)
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}\n")

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

EPOCHS = 30

# 【修改 2】: 启用 CosineAnnealingLR 调度器
# T_max: 学习率降到最低所需的总步数。
# 我们设置为总训练步数 (epochs * steps_per_epoch)
# eta_min: 学习率的最小值
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6
)

# 训练循环
best_acc = 0.0
train_losses, train_accs, test_losses, test_accs = [], [], [], []
print("开始训练\n")
for ep in range(EPOCHS):
    train_one_epoch(
        model,
        train_loader,
        optimizer,
        scheduler,  # 传入调度器
        criterion,
        device,
        ep,
        train_losses,
        train_accs,
    )
    acc = evaluate(model, test_loader, criterion, device, test_losses, test_accs)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "fashionmnist_best_improved.pth")
print(f"\n训练完成, 最佳准确率: {best_acc:.2f}%\n")

# 学习曲线
plot_curves(train_losses, test_losses, train_accs, test_accs)

# 载入最佳权重再做可视化与报告
print("载入最佳权重进行评估...")
model.load_state_dict(torch.load("fashionmnist_best_improved.pth", map_location=device))
model.eval()

# 随机抽样 15 张（3×5）带方框可视化
visualize_random_15(model, test_dataset, device, CLASS_NAMES)

# 混淆矩阵 & 分类报告
plot_confusion_and_report(model, test_loader, device, CLASS_NAMES)


# ===================== 脚本 1 的抽样对比 =====================
print("\n" + "=" * 20 + " 抽样展示 (类 Paddle) " + "=" * 20)

model.eval()

# 脚本 1 中使用的索引
indexs = [2, 15, 38, 211, 222, 323]

# CLASS_NAMES 已经在 main 函数里定义过了

for idx in indexs:
    # 从 test_dataset 中获取数据
    image, label = test_dataset[idx]

    # 1. 增加 batch 维度: [1, 28, 28] -> [1, 1, 28, 28]
    # 2. 发送到设备
    image_batch = image.unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(image_batch)

    # 4. 获取预测结果
    pred_idx = output.argmax(1).item()

    # 5. 获取真实标签和预测标签的名称
    true_label = CLASS_NAMES[label]
    pred_label = CLASS_NAMES[pred_idx]

    # 按照脚本 1 的格式打印
    print(
        f"第 {idx:<3} 条记录 | 真实值: {label} ({true_label:<11}) | 预测值: {pred_idx} ({pred_label:<11})"
    )
# ==========================================================

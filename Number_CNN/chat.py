# =========================================
# MNIST（MPS 优化版，EPOCHS=40，基宽=64）
# 修正版：按 key 的 EMA（参数做 EMA，BN buffers 直接 copy）
# - 训练期评估/保存：以当前模型为准，同时跟踪/保存 EMA
# - EMA 衰减热身渐进（0.99 -> 0.999）
# - 可选 BN 重校准与 BN 同步检查
# =========================================

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ---- 设备 & 随机种子
torch.manual_seed(42)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}\n")

# ---- 数据
train_transform = T.Compose(
    [
        T.RandomRotation(12),
        T.RandomAffine(degrees=0, translate=(0.15, 0.15), shear=10),
        T.RandomPerspective(distortion_scale=0.03, p=0.25),
        T.ToTensor(),
        T.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
        T.Normalize((0.1307,), (0.3081,)),
    ]
)
test_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ]
)

train_dataset = datasets.MNIST(
    "./dataset", train=True, download=True, transform=train_transform
)
test_dataset = datasets.MNIST("./dataset", train=False, transform=test_transform)

BATCH_TRAIN, BATCH_TEST = 128, 1000
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_TRAIN, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_TEST, shuffle=False, num_workers=0
)


# ---- 模型（Small ResNet + SE + GAP）
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch, max(1, ch // r), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, ch // r), ch, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class BasicBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.skip is not None:
            identity = self.skip(identity)
        return self.relu(out + identity)


class SmallResNetSE(nn.Module):
    def __init__(self, num_classes=10, base=64, use_se=True):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 4
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            BasicBlockSE(c1, c1, stride=1, use_se=use_se),
            BasicBlockSE(c1, c1, stride=1, use_se=use_se),
        )
        self.layer2 = nn.Sequential(
            BasicBlockSE(c1, c2, stride=2, use_se=use_se),
            BasicBlockSE(c2, c2, stride=1, use_se=use_se),
        )
        self.layer3 = nn.Sequential(
            BasicBlockSE(c2, c3, stride=2, use_se=use_se),
            BasicBlockSE(c3, c3, stride=1, use_se=use_se),
        )
        self.layer4 = nn.Sequential(
            BasicBlockSE(c3, c4, stride=2, use_se=use_se),
            BasicBlockSE(c4, c4, stride=1, use_se=use_se),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c4, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


model = SmallResNetSE(base=64).to(device)
print("模型结构:")
print(model)
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}\n")

# ---- 损失 / 优化器 / 调度器
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.SGD(
    model.parameters(), lr=0.05, momentum=0.9, nesterov=True, weight_decay=5e-4
)

EPOCHS = 40
steps_per_epoch = len(train_loader)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.35,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.25,
    div_factor=10.0,
    final_div_factor=1e4,
)


# ---- EMA（按 key 同步：参数做 EMA，BN buffers 直接 copy）
class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = deepcopy(model)
        if device is not None:
            self.ema.to(device)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def set_decay(self, decay: float):
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k in esd.keys():
            src = msd[k]
            dst = esd[k]
            if not torch.is_floating_point(dst):
                dst.copy_(src)
                continue
            if (
                ("running_mean" in k)
                or ("running_var" in k)
                or ("num_batches_tracked" in k)
            ):
                dst.copy_(src)  # BN 缓冲区直接同步
            else:
                dst.copy_(dst * d + src * (1.0 - d))  # 其他浮点参数做 EMA
        self.ema.load_state_dict(esd, strict=True)


ema = ModelEMA(model, decay=0.999, device=device)
ema.ema.load_state_dict(model.state_dict())  # 立即对齐


# EMA 衰减热身渐进：前 ~2 个 epoch 用 0.99 -> 渐进到 0.999
def ema_update_with_ramp(
    ema_obj,
    model,
    global_step,
    ramp_steps=2 * steps_per_epoch,
    base_decay=0.99,
    final_decay=0.999,
):
    if global_step < ramp_steps:
        t = global_step / max(1, ramp_steps)
        cur = base_decay + (final_decay - base_decay) * t
    else:
        cur = final_decay
    ema_obj.set_decay(cur)
    ema_obj.update(model)


# ---- 可选：BN 同步检查 & BN 重校准
def check_bn_sync(model, ema_model):
    m_bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    e_bns = [m for m in ema_model.modules() if isinstance(m, nn.BatchNorm2d)]
    ok = True
    for i, (mb, eb) in enumerate(zip(m_bns, e_bns)):
        if not torch.allclose(mb.running_mean, eb.running_mean, atol=1e-4, rtol=1e-4):
            ok = False
        if not torch.allclose(mb.running_var, eb.running_var, atol=1e-3, rtol=1e-3):
            ok = False
    print("BN buffers synced? ", ok)


@torch.no_grad()
def bn_recalibrate(model_to_calib, loader, max_batches=50):
    model_to_calib.train()
    it = 0
    for x, _ in loader:
        x = x.to(device)
        _ = model_to_calib(x)
        it += 1
        if it >= max_batches:
            break
    model_to_calib.eval()


# ---- 训练 / 评估 / TTA
def train_one_epoch(epoch, log_interval=100):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 如需可开启
        optimizer.step()
        scheduler.step()

        global_step = epoch * steps_per_epoch + batch_idx
        ema_update_with_ramp(ema, model, global_step)

        running_loss += loss.item()
        pred = output.argmax(1)
        total += target.size(0)
        correct += (pred == target).sum().item()

        if (batch_idx + 1) % log_interval == 0:
            print(
                f"Epoch {epoch + 1:02d} | Step {batch_idx + 1:04d}/{steps_per_epoch} "
                f"| LR {scheduler.get_last_lr()[0]:.5f} "
                f"| Loss {running_loss / (batch_idx + 1):.4f} "
                f"| Acc {100.0 * correct / total:.2f}%"
            )
    return running_loss / steps_per_epoch, 100.0 * correct / total


@torch.no_grad()
def evaluate(model_to_eval):
    model_to_eval.eval()
    test_loss, correct, total = 0.0, 0, 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        out = model_to_eval(data)
        test_loss += criterion(out, target).item()
        pred = out.argmax(1)
        total += target.size(0)
        correct += (pred == target).sum().item()
    return test_loss / len(test_loader), 100.0 * correct / total


@torch.no_grad()
def evaluate_tta(model_to_eval):
    import torchvision.transforms.functional as F

    model_to_eval.eval()
    total, correct, total_loss = 0, 0, 0.0
    angles = [0, +5, -5]
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits_sum = None
        for a in angles:
            aug = torch.stack(
                [
                    F.rotate(
                        img,
                        a,
                        interpolation=T.InterpolationMode.BILINEAR,
                        expand=False,
                        center=None,
                        fill=0,
                    )
                    for img in imgs
                ]
            )
            out = model_to_eval(aug)
            logits_sum = out if logits_sum is None else (logits_sum + out)
        out = logits_sum / len(angles)
        total_loss += criterion(out, labels).item()
        pred = out.argmax(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    return total_loss / len(test_loader), 100.0 * correct / total


# ---- 主循环：训练期以“当前模型”评估/保存，同时跟踪 EMA
best_model_acc = 0.0
best_ema_acc = 0.0
tr_losses, tr_accs, te_losses, te_accs = [], [], [], []

print("开始训练...\n")
printed_bn_check = False

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_one_epoch(epoch)

    # 评估（稳定起见，用当前模型为准）
    te_loss, te_acc = evaluate(model)

    # 仅前几轮对 EMA 做一次 BN 重校准（可注释掉）
    if epoch in (0, 1, 2):
        bn_recalibrate(ema.ema, train_loader, max_batches=50)

    # 可选：第一轮打印一次 BN 同步情况
    if not printed_bn_check:
        check_bn_sync(model, ema.ema)
        printed_bn_check = True

    ema_loss, ema_acc = evaluate(ema.ema)

    tr_losses.append(tr_loss)
    tr_accs.append(tr_acc)
    te_losses.append(te_loss)
    te_accs.append(te_acc)

    print(
        f"===> Epoch {epoch + 1:02d} | Train Loss {tr_loss:.4f} Acc {tr_acc:.2f}% "
        f"| Test(Model) Loss {te_loss:.4f} Acc {te_acc:.2f}% "
        f"| Test(EMA) Loss {ema_loss:.4f} Acc {ema_acc:.2f}%"
    )

    # 分别保存两份 best
    if te_acc > best_model_acc:
        best_model_acc = te_acc
        torch.save(model.state_dict(), "mnist_best_model.pth")
    if ema_acc > best_ema_acc:
        best_ema_acc = ema_acc
        torch.save(ema.ema.state_dict(), "mnist_best_ema.pth")

print(
    f"\n训练完成! 最佳(Model) Acc: {best_model_acc:.2f}% | 最佳(EMA) Acc: {best_ema_acc:.2f}%\n"
)

# ---- 训练曲线
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(tr_losses, label="Train Loss", linewidth=2)
axes[0].plot(te_losses, label="Test Loss (Model)", linewidth=2)
axes[0].set_title("Loss Curve")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(tr_accs, label="Train Acc", linewidth=2)
axes[1].plot(te_accs, label="Test Acc (Model)", linewidth=2)
axes[1].set_title("Accuracy Curve")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy (%)")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ---- 最终对比评估（best-EMA vs best-Model）
print("载入 best_ema.pth 并做 TTA 评估...")
model.load_state_dict(torch.load("mnist_best_ema.pth", map_location=device))
ema_tta_loss, ema_tta_acc = evaluate_tta(model)
print(f"[Best EMA]   TTA Loss: {ema_tta_loss:.4f} | TTA Acc: {ema_tta_acc:.2f}%")

print("载入 best_model.pth 并做 TTA 评估...")
model.load_state_dict(torch.load("mnist_best_model.pth", map_location=device))
mdl_tta_loss, mdl_tta_acc = evaluate_tta(model)
print(f"[Best Model] TTA Loss: {mdl_tta_loss:.4f} | TTA Acc: {mdl_tta_acc:.2f}%")

# ---- 可视化 & 报告（用 best_model）
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)
with torch.no_grad():
    out = model(images)
    preds = out.argmax(1)

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat[:16]):
    img = images[i].detach().cpu().squeeze().numpy()
    ax.imshow(img, cmap="gray")
    p, t = int(preds[i].item()), int(labels[i].item())
    ax.set_title(
        f"Pred: {p}, True: {t}", color=("green" if p == t else "red"), fontweight="bold"
    )
    ax.axis("off")
plt.suptitle("预测结果（Best Model）", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

print("\n生成混淆矩阵与分类报告（Best Model）...")
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, lbls in test_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        all_preds.extend(out.argmax(1).detach().cpu().numpy())
        all_labels.extend(lbls.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10)
)
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title("混淆矩阵 (Best Model)")
plt.tight_layout()
plt.show()

print("\n分类报告:")
print(
    classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(10)]
    )
)

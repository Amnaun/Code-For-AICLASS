import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# 设置随机种子
torch.manual_seed(24)

# 数据预处理
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# 数据加载
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, transform=transform)

BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}\n")


class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # 卷积块1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 卷积块2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 新增: 双卷积
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


model = ImprovedCNN().to(device)
print("模型结构:")
print(model)
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}\n")

# 优化器和学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 新增


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
        optimizer.step()

        running_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(output.data, 1)
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


# ========== 评估函数 (修复Bug) ==========
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

    # 修复: 在循环外计算
    test_loss /= len(loader)
    accuracy = 100.0 * correct / len(loader.dataset)

    test_losses.append(test_loss)
    test_accs.append(accuracy)

    print(f" | Test Loss: {test_loss:.4f}, Acc: {accuracy:.2f}%")
    return accuracy


# ========== 训练循环 ==========
EPOCHS = 15  # 增加训练轮次
best_acc = 0.0

# 记录训练过程
train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("开始训练...\n")

for epoch in range(EPOCHS):
    train(epoch, train_losses, train_accs)
    acc = evaluate(test_loader, test_losses, test_accs)

    # 保存最佳模型
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mnist_best.pth")

    # 更新学习率
    scheduler.step()

print(f"\n训练完成! 最佳准确率: {best_acc:.2f}%\n")

# ========== 可视化训练过程 ==========
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 损失曲线
axes[0].plot(train_losses, label="Train Loss", linewidth=2)
axes[0].plot(test_losses, label="Test Loss", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Training and Test Loss", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

# 准确率曲线
axes[1].plot(train_accs, label="Train Accuracy", linewidth=2)
axes[1].plot(test_accs, label="Test Accuracy", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Accuracy (%)", fontsize=12)
axes[1].set_title("Training and Test Accuracy", fontsize=14, fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ========== 预测展示 ==========
model.load_state_dict(torch.load("mnist_best.pth"))
model.eval()

# 获取一批数据
dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

# 显示前16个
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    if i >= 16:
        break

    img = images[i].cpu().squeeze().numpy()
    ax.imshow(img, cmap="gray")

    pred = predicted[i].item()
    true = labels[i].item()
    color = "green" if pred == true else "red"

    ax.set_title(f"Pred: {pred}, True: {true}", color=color, fontweight="bold")
    ax.axis("off")

plt.suptitle("预测结果 (绿色=正确, 红色=错误)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# ========== 混淆矩阵 ==========
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("\n生成混淆矩阵...")

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10)
)
plt.xlabel("预测标签", fontsize=12)
plt.ylabel("真实标签", fontsize=12)
plt.title("混淆矩阵", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# 分类报告
print("\n分类报告:")
print(
    classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(10)]
    )
)

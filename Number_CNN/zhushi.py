# =========================================
# 0) 基本库导入 & 随机种子
# =========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 为了结果可复现，固定随机数种子（训练每次更稳定）
torch.manual_seed(24)


# =========================================
# 1) 数据预处理与数据集
#    - 训练集做数据增强：轻微旋转/平移
#    - 统一标准化到MNIST的均值/方差
# =========================================
train_transform = transforms.Compose(
    [
        transforms.RandomRotation(10),                 # 随机旋转±10°
        transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移10%
        transforms.ToTensor(),                        # 转成张量，范围[0,1]
        transforms.Normalize((0.1307,), (0.3081,)),   # 按MNIST统计值做标准化
    ]
)

# 测试集不做增强，只做同样的标准化
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# 下载/加载MNIST
train_dataset = datasets.MNIST(
    "./dataset", train=True, download=True, transform=train_transform
)
test_dataset = datasets.MNIST("./dataset", train=False, transform=test_transform)

# 按批组织数据；训练集打乱，测试集不打乱
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# =========================================
# 2) 设备选择
#    - Apple 芯片上优先用 MPS；否则退回 CPU
# =========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"使用设备: {device}\n")


# =========================================
# 3) 模型结构：改进版CNN
#    思路：
#      - 卷积块：Conv→BN→ReLU（提特征）×若干，再接最大池化（降采样）
#      - 三个卷积块后接全连接层做分类
#      - Dropout在全连接层，减轻过拟合
# =========================================
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()

        # 第一块：输入1通道→32通道，堆两次3x3卷积 + BN + ReLU，再池化
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
        )

        # 第二块：32→64通道，同样结构
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )

        # 第三块：64→128通道，单卷积 + 池化
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7 -> 3x3（向下取整）
        )

        # 全连接层：先Dropout，再两层Linear输出10类
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.conv1(x)        # [B,1,28,28] -> [B,32,14,14]
        x = self.conv2(x)        # -> [B,64,7,7]
        x = self.conv3(x)        # -> [B,128,3,3]
        x = torch.flatten(x, 1)  # 展平成[B, 128*3*3]
        x = self.fc(x)           # 输出[B,10]，未过softmax（交叉熵里自带）
        return x


# 实例化模型并查看结构与参数量
model = ImprovedCNN().to(device)
print("模型结构:")
print(model)
print(f"\n参数量: {sum(p.numel() for p in model.parameters()):,}\n")


# =========================================
# 4) 损失函数 / 优化器 / 学习率调度
#    - CrossEntropyLoss：多分类常用
#    - Adam：收敛快，调参相对省心
#    - OneCycleLR：前期升学习率再稳步降，有助于更快/更稳收敛
# =========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,                 # 学习率最高到0.01，然后再降
    epochs=20,                   # 与总轮数一致
    steps_per_epoch=len(train_loader),  # 每个step更新一次调度器
)


# =========================================
# 5) 训练与评估函数
#    - train(): 单轮训练，记录平均损失/准确率
#      细节：梯度清零→前向→计算loss→反向传播→梯度裁剪→优化器/调度器步进
#    - evaluate(): 在验证/测试集上评估
# =========================================
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

        # 梯度裁剪：限制梯度范数，避免梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()  # 每个batch都更新学习率

        running_loss += loss.item()
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


def evaluate(loader, test_losses, test_accs):
    model.eval()
    test_loss = 0
    correct = 0

    # 评估阶段不需要梯度，速度更快、显存更省
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


# =========================================
# 6) 训练主循环
#    - 保存测试集上表现最好的权重
#    - 同时记录曲线所需指标
# =========================================
EPOCHS = 20
best_acc = 0.0

train_losses = []
train_accs = []
test_losses = []
test_accs = []

print("开始训练...\n")

for epoch in range(EPOCHS):
    train(epoch, train_losses, train_accs)
    acc = evaluate(test_loader, test_losses, test_accs)

    # 只要当前准确率更高，就覆盖保存
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "mnist_best_improved.pth")

print(f"\n训练完成! 最佳准确率: {best_acc:.2f}%\n")


# =========================================
# 7) 可视化训练过程：损失/准确率曲线
# =========================================
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(train_losses, label="Train Loss", linewidth=2)
axes[0].plot(test_losses, label="Test Loss", linewidth=2)
axes[0].set_xlabel("Epoch", fontsize=12)
axes[0].set_ylabel("Loss", fontsize=12)
axes[0].set_title("Loss Curve", fontsize=14, fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(train_accs, label="Train Accuracy", linewidth=2)
axes[1].plot(test_accs, label="Test Accuracy", linewidth=2)
axes[1].set_xlabel("Epoch", fontsize=12)
axes[1].set_ylabel("Accuracy (%)", fontsize=12)
axes[1].set_title("Accuracy Curve", fontsize=14, fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================
# 8) 加载最佳模型 & 可视化部分预测
#    - 随机取一批测试图像，展示预测 vs 真实标签
#    - 预测正确为绿色，错误为红色
# =========================================
model.load_state_dict(torch.load("mnist_best_improved.pth"))
model.eval()

dataiter = iter(test_loader)
images, labels = next(dataiter)
images = images.to(device)

with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

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

plt.suptitle("预测结果", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()


# =========================================
# 9) 混淆矩阵 & 分类报告
#    - 混淆矩阵能看出每一类被错分到哪一类
#    - 分类报告包含Precision/Recall/F1
# =========================================
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

print("\n分类报告:")
print(
    classification_report(
        all_labels, all_preds, target_names=[str(i) for i in range(10)]
    )
)

# 核心数据处理
import numpy as np  # 导入 numpy ,用于数值和数组操作
from sklearn.model_selection import train_test_split  # 导入scikit-learn 的数据分割工具
from sklearn.preprocessing import StandardScaler  # 导入 scikit-learn 的特征标准化工具

# PyTorch 核心库
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
from torch.utils.data import (
    TensorDataset,
    DataLoader,
)  # 数据加载工具 TensorDataset 封装数据， DataLoader 批量加载数据
from sklearn.datasets import load_breast_cancer  # 加载 scikit-learn 内置数据集

# 数据加载(scikit-learn 辅助)
data = load_breast_cancer(as_frame=False)  # 加载乳腺癌数据集
X = data.data.astype(np.float32)  # 获取特征数据 X ,并确保数据类型为 float32
y = data.target.astype(np.float32)  # 获取目标变量 y ，并确保数据类型为 float32

# 数据集分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=24,  # 将数据按 80% / 20% 比例划分为训练集和测试集 random_state 为定值确保可以复现
)

# 特征缩放
scaler = StandardScaler()  # 实例化 StandardScaler 对象
X_train_scaled = scaler.fit_transform(
    X_train
)  # 用训练集数据拟合 (fit) 缩放器并进行转换 (transform)
X_test_scaled = scaler.transform(
    X_test
)  # 用训练集学到的参数直接转换 (transform) 数据集

# 转换为PyTorch Tensor
X_train_tensor = torch.from_numpy(
    X_train_scaled
)  # 将训练集特征的 numpy 数组转化为 PyTorch Tensor
y_train_tensor = torch.from_numpy(y_train).view(
    -1, 1
)  # 将训练集标签转换为 Tensor ,并调整形状为 (N,1)
X_test_tensor = torch.from_numpy(X_test_scaled)  # 将测试集特征转化为 Tensor
y_test_tensor = torch.from_numpy(y_test).view(
    -1, 1
)  # 将测试集标签转化为 Tensor ,并调整形状为 (N,1)

# 创建DataLoader
# TensorDataset 绑定数据和标签
train_dataset = TensorDataset(
    X_train_tensor, y_train_tensor
)  # 创建训练集 TensorDataset , 绑定数据和标签
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)  # 创建测试集 TensorDataset

# DataLoader 负责按批量加载数据
train_loader = DataLoader(
    dataset=train_dataset, batch_size=32, shuffle=True
)  # 创建训练集 DataLoader , 批量大小 32 ，打乱数据
test_loader = DataLoader(
    dataset=test_dataset, batch_size=32, shuffle=False
)  # 创建测试集 DataLoader , 批量大小 32 , 不打乱数据

input_size = X_train_tensor.shape[1]  # 获取输入特征的数量
print(f"输入特征数量{input_size}")  # 打印特征数量


# 调整网络层和激活函数
class SimpleNN(nn.Module):  # 定义一个继承自 nn.Module 的神经网络类
    def __init__(self, input_size):  # 构造函数
        super(SimpleNN, self).__init__()  # 调用父类函数

        # 定义网络层
        self.layer_1 = nn.Linear(
            input_size, 64
        )  # 第一层：全连接层，输入维度 input_size , 输出 64
        self.relu = nn.ReLU()  # ReLU 激活函数
        self.dropout = nn.Dropout(0.2)  # Dropout 层，失活率为 20% ，用于防止过拟合
        self.layer_2 = nn.Linear(64, 32)  # 第二层：全连接层，输入 64 ，输出 32

        # 输出层：二分类 units=1 , sigmoid
        self.output_layer = nn.Linear(32, 1)  # 输出层：全连接层，输入 32 ，输出 1
        self.sigmoid = nn.Sigmoid()  # Sigmoid 激活函数 , 将输出压缩到 (0,1) 之间

    def forward(self, x):  # 定义数据前向传播的方法
        # 定义数据前向传播的路径
        out = self.layer_1(x)  # 数据通过第一层
        out = self.relu(out)  # 应用 ReLU 激活
        out = self.dropout(out)  # 应用 Dropout
        out = self.layer_2(out)  # 数据通过第二层
        out = self.relu(out)  # 应用 Relu 激活
        out = self.output_layer(out)  # 数据通过输出层
        out = self.sigmoid(out)  # 应用 Sigmoid 激活
        return out  # 返回最终输出


# 实例化模型
model = SimpleNN(input_size)  # 使用前面获取的 input_size 实例化模型

# 设置设备
device = torch.device(
    "mps"
)  # 这里因为用的是 Mac 所以是 mps ，如果用 Windows 或者 Linux 就用 cuda
model.to(device)  # 将整个模型移动到选定的设备上
print(f"模型部署到{device}")  # 打印模型部署的设备

# 调整超参数

# 定义损失函数和优化器
# 二分类用 BCELoss
criterion = nn.BCELoss()  # 定义损失函数:二分类交叉熵损失
optimizer = optim.Adam(
    model.parameters(), lr=0.001
)  # 定义优化器为 Adam , 学习率 lr 设定为 0.001

num_epochs = 50  # 设置训练的总 Epoch 数量
print("\n--- 开始模型训练 ---")  # 打印训练开始的信息

loss = torch.tensor(0.0)  # 初始化 loss

# 训练循环
for epoch in range(num_epochs):  # 循环 N 次 epoch
    model.train()  # 将模型设置为训练模式
    for inputs, targets in train_loader:  # 遍历训练集 DataLoader 中的每一个数据批次
        inputs, targets = (
            inputs.to(device),
            targets.to(device),
        )  # 将输入数据和标签移动到设备上

        # 前向传播
        outputs = model(inputs)  # 模型进行前向传播，得到预测输出
        loss = criterion(outputs, targets)  # 计算损失值

        # 反向传播与优化
        optimizer.zero_grad()  # 清楚上一步计算的梯度
        loss.backward()  # 反向传播，计算当前批次的梯度
        optimizer.step()  # 根据梯度更新模型的权重

    # 打印训练状态 （每 10 个 epoch ）
    if (epoch + 1) % 10 == 0:  # 每 10 个 epoch 打印一次状态
        print(
            f"Epoch [{epoch + 1}/{num_epochs}],Loss:{loss.item():.4f}"
        )  # 打印当前的 epoch 数和损失值

# 模型评估
model.eval()  # 设置为评估模式
with torch.no_grad():  # 禁用梯度计算,节省内存并加速
    correct = 0  # 初始化正确预测数
    total = 0  # 初始化总样本数
    for inputs, targets in test_loader:  # 遍历测试集 DataLoader 中的每一个批次
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据转移到设备上

        outputs = model(inputs)  # 模型进行预测

        # 将概率转换为预测类别
        predicted = (
            outputs > 0.5
        ).float()  # 将 Sigmoid 输出的概率值转化为 0 或 1 的预测标签

        total += targets.size(0)  # 累加批次中的样本数到总数
        correct += (predicted == targets).sum().item()  # 统计预测正确的数量

    accuracy = 100 * correct / total  # 计算最终的准确率
    print(f"\n测试集准确率：{accuracy:.2f}")  # 打印测试集准确率

# 保存模型
torch.save(model.state_dict(), "pytorch_nn_model.pth")  # 保存模型的参数
print(f"模型已保存到 pytorch_nn_model ")  # 打印保存成功的消息

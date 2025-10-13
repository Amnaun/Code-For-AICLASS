"""
鲍鱼年龄预测 - 纯线性回归 + SGD优化
使用PyTorch和M4 MPS加速
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
torch.manual_seed(24)
np.random.seed(24)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

data_X = []
data_Y = []
sex_map = {'I': 0, 'M': 1, 'F': 2}

with open('dataset/AbaloneAgePrediction.txt') as f:
    for line in f.readlines():
        line = line.split(',')
        line[0] = sex_map[line[0]]
        data_X.append(line[:-1])
        data_Y.append(line[-1:])

data_X = np.array(data_X, dtype='float32')
data_Y = np.array(data_Y, dtype='float32')

print(f'数据形状: X={data_X.shape}, Y={data_Y.shape}')
print(f'年龄范围: [{data_Y.min():.1f}, {data_Y.max():.1f}]')

scaler_X = StandardScaler()
data_X_scaled = scaler_X.fit_transform(data_X)
scaler_y = StandardScaler()
data_Y_scaled = scaler_y.fit_transform(data_Y)

# ========== 3. 数据分割 ==========
X_train, X_test, y_train, y_test = train_test_split(
    data_X_scaled, 
    data_Y_scaled,
    test_size=0.2,
    random_state=42
)

print(f'\n训练集: {X_train.shape}')
print(f'测试集: {X_test.shape}')

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f'Batch大小: {batch_size}')

# ========== 4. 线性回归模型 ==========
class LinearRegressor(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
model = LinearRegressor(input_dim).to(device)

learning_rate = 0.1  
momentum = 0.9       
weight_decay = 1e-5  
optimizer = optim.SGD(
    model.parameters(), 
    lr=learning_rate,
    momentum=momentum,
    weight_decay=weight_decay
)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=20,
    min_lr=1e-6
)

criterion = nn.MSELoss()

# ========== 6. 训练函数 ==========
def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(data_loader)


# ========== 7. 训练模型 ==========

epochs = 200
train_losses = []
test_losses = []
best_test_loss = float('inf')

for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    
    test_loss = evaluate(model, test_loader, criterion)
    test_losses.append(test_loss)
    
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(test_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), 'best_linear_model.pth')
    
    if (epoch + 1) % 20 == 0 or epoch < 5:
        print(f"epoch [{epoch+1:3d}/{epochs}] | "
              f"train loss: {train_loss:.5f} | "
              f"test loss: {test_loss:.5f} | "
              f"lr: {new_lr:.6f}")
        
        if new_lr != old_lr:
            print(f"  >>> 学习率降低: {old_lr:.6f} -> {new_lr:.6f}")


model.load_state_dict(torch.load('best_linear_model.pth'))

# ========== 8. 模型评估 ==========
model.eval()

with torch.no_grad():
    predictions_scaled = model(X_test_tensor).cpu().numpy()
    actuals_scaled = y_test_tensor.cpu().numpy()

predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
actuals = scaler_y.inverse_transform(actuals_scaled).flatten()

rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

print(f"  - RMSE (均方根误差): {rmse:.4f} 岁")
print(f"  - R²   (决定系数):    {r2:.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_title('Training History', fontsize=14)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (MSE)')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.scatter(actuals, predictions, alpha=0.6, edgecolors='w', linewidth=0.5)
ax2.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', linewidth=2, label='Perfect Line')
ax2.set_title('Prediction vs. Ground Truth', fontsize=14)
ax2.set_xlabel('Ground Truth (真实年龄)')
ax2.set_ylabel('Prediction (预测年龄)')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

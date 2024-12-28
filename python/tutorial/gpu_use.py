import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 定义一个简单的神经网络模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 4096)  # 输入层：784维，输出层：128维
        self.fc2 = nn.Linear(4096, 1024)   # 隐藏层：128维，64维
        self.fc3 = nn.Linear(1024, 64)    # 隐藏层：64维，10维
        self.fc4 = nn.Linear(64, 10)    # 输出层：10维（假设是 10 类分类问题）

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# 创建一个简单的数据集（例如，MNIST 手写数字数据）
# 在这里我们用随机数据来模拟训练过程
# 你可以用真实的数据集，如 MNIST 或 CIFAR10，来替换它

X_train = torch.randn(1000, 784)  # 1000 个样本，每个样本 784 特征
y_train = torch.randint(0, 10, (1000,))  # 1000 个标签，值范围从 0 到 9

# 将数据转换为 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 初始化模型并将其移动到 GPU
model = SimpleNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 1000
epoch = 0
while True:
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 将数据移动到 GPU
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播并优化
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # 每个 epoch 打印一次损失值
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")
    epoch += 1

print("Training finished.")

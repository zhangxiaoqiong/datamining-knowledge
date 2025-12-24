---
tags: [算法, 深度学习, 神经网络, 监督学习, 梯度下降]
math: true
difficulty: 困难
---

# 人工神经网络 (Artificial Neural Network, ANN)

## 💡 核心直觉

- **一句话定义**：通过多层神经元的非线性组合，学习输入到输出的复杂映射函数 $f(x)$ 的通用近似器。

- **解决问题**：解决了线性模型（LR、SVM）无法捕捉数据中复杂非线性关系的问题。单层感知机无法表示异或(XOR)，多层网络可以。

- **核心逻辑**：ANN = 输入层 → 隐层（多个）→ 输出层，通过**反向传播**优化参数，使预测逼近真实值。

- **几何意义**：每一层的非线性激活函数都是一次**特征空间的非线性变换**，深层网络通过叠加多个这样的变换，将原始特征空间扭曲、折叠、分割，最终在高维特征空间中线性可分。

- **杀手锏 (Killer Feature)**：通用近似定理（Universal Approximation Theorem）保证足够宽的单隐层网络可以近似任意连续函数。实际应用中，深网络（深层）比宽网络更高效（样本复杂度低）。

> [!TIP] 核心架构图解
>
> ```
> 输入层          隐层1         隐层2         输出层
>   x₁  ─┐                                    ŷ₁
>        ├──→ h₁⁽¹⁾  ─┐                    ┌─→ ŷ₂
>   x₂  ─┤            ├──→ h₁⁽²⁾  ──→  σ  ┤
>   x₃  ─┴──→ h₂⁽¹⁾  ─┤          (输出)    └─→ ŷ₃
>                      └──→ h₂⁽²⁾
>
> 前向传播：计算 ŷ = σ(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)
> 反向传播：计算 ∂L/∂W，∂L/∂b 用于梯度下降
> 核心机制：每层的非线性激活函数（ReLU、sigmoid）打破线性性
> ```

---

## 📐 数学原理

### 1. 前向传播 (Forward Propagation)

对于 $L$ 层网络，第 $l$ 层的计算为：

**线性变换**：
$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

**非线性激活**：
$$a^{(l)} = \sigma(z^{(l)})$$

其中：
- $a^{(l)}$：第 $l$ 层的激活向量（输出），$a^{(0)} = x$（输入）
- $W^{(l)}$：权重矩阵，形状 $(n^{(l)}, n^{(l-1)})$
- $b^{(l)}$：偏置向量，形状 $(n^{(l)}, 1)$
- $\sigma$：激活函数（ReLU、sigmoid、tanh等）
- $z^{(l)}$：线性组合结果（未激活）

**完整前向传播**：从输入递推到输出
$$\hat{y} = a^{(L)} = \sigma^{(L)}(W^{(L)} \sigma^{(L-1)}(\cdots \sigma^{(1)}(W^{(1)} x + b^{(1)}) \cdots) + b^{(L)})$$

> [!ABSTRACT] 激活函数的必要性
>
> 不使用激活函数时，多层网络退化为线性变换：
> $$a^{(L)} = W^{(L)} W^{(L-1)} \cdots W^{(1)} x + \text{(bias terms)}$$
> 其中 $W^{(L)} W^{(L-1)} \cdots W^{(1)}$ 仍是矩阵，无法表示非线性。激活函数引入了非线性，使深层网络真正有表达能力。

### 2. 损失函数与反向传播

**回归任务**（MSE 损失）：
$$L = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$

**分类任务**（交叉熵损失）：
$$L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$

其中：
- $m$：样本数
- $K$：类别数
- $y_{i,k}$：one-hot 编码的真实标签
- $\hat{y}_{i,k}$：softmax 预测的概率

### 3. 反向传播 (Backpropagation)

反向传播的核心是**链式法则**。对于第 $l$ 层，计算梯度：

**输出层梯度**（以 MSE 为例）：
$$\delta^{(L)} = \frac{\partial L}{\partial z^{(L)}} = (\hat{y} - y) \odot \sigma'(z^{(L)})$$

其中：
- $\odot$：element-wise 乘积（Hadamard积）
- $\sigma'$：激活函数的导数

**隐层梯度**（链式法则递推）：
$$\delta^{(l)} = (W^{(l+1)})^T \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

这从第 $L$ 层逆向传播到第 1 层。

**参数梯度**：
$$\frac{\partial L}{\partial W^{(l)}} = \frac{1}{m} \delta^{(l)} (a^{(l-1)})^T$$

$$\frac{\partial L}{\partial b^{(l)}} = \frac{1}{m} \sum_{i=1}^{m} \delta^{(l)}_i$$

**参数更新**（梯度下降）：
$$W^{(l)} \leftarrow W^{(l)} - \eta \frac{\partial L}{\partial W^{(l)}}$$

$$b^{(l)} \leftarrow b^{(l)} - \eta \frac{\partial L}{\partial b^{(l)}}$$

其中 $\eta$ 是学习率。

> [!TIP] 反向传播的几何意义
>
> 梯度 $\nabla L$ 指向损失函数增加最快的方向。梯度下降就是沿着 $-\nabla L$ 方向走一步 $\eta$，目标是找到局部最小值。
>
> **关键计算**：链式法则 $\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial z^{(L-1)}} \cdots \frac{\partial z^{(l+1)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$
>
> 每一步都包含激活函数导数，这是导致**梯度消失**的源头（ReLU 解决了这个问题）。

### 4. 常用激活函数

| 激活函数 | 公式 | 导数 | 优缺点 |
|---|---|---|---|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ | 输出范围 [0,1]，但导数最大 0.25，易梯度消失 |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $\tanh'(z) = 1 - \tanh^2(z)$ | 输出范围 [-1,1]，导数最大 1，比 sigmoid 好 |
| **ReLU** | $\text{ReLU}(z) = \max(0, z)$ | $\text{ReLU}'(z) = \begin{cases}1 & z>0 \\ 0 & z\leq0 \end{cases}$ | 计算快，导数恒为 0 或 1，解决梯度消失。但有 dead ReLU 问题 |
| **Leaky ReLU** | $\text{LReLU}(z) = \max(\alpha z, z)$ | $\text{LReLU}'(z) = \begin{cases}1 & z>0 \\ \alpha & z\leq0 \end{cases}$ | 改进 ReLU，负数段有小梯度，避免完全死亡 |
| **Softmax** | $\text{softmax}_i(z) = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | - | 仅用于输出层（多分类），输出概率分布 |

---

## 💻 算法实现

### PyTorch 完整实现（2层网络）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleNeuralNetwork(nn.Module):
    """PyTorch 实现的 2 层神经网络"""

    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNeuralNetwork, self).__init__()

        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 隐层（输入 → 隐层）
        self.relu = nn.ReLU()                           # ReLU 激活
        self.fc2 = nn.Linear(hidden_size, output_size) # 输出层

        # He 初始化（适配 ReLU）
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """前向传播"""
        z1 = self.fc1(x)           # 线性变换
        a1 = self.relu(z1)         # 隐层激活
        z2 = self.fc2(a1)          # 输出层线性
        return z2  # 返回 logits，由 loss 函数内部应用 softmax

class NeuralNetworkTrainer:
    """训练器，封装训练循环"""

    def __init__(self, model, learning_rate=0.01, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # CrossEntropyLoss 内部包含 softmax + 交叉熵
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """训练网络"""
        # 转换为 PyTorch 张量
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

        # 创建验证集
        val_size = int(len(X_train) * validation_split)
        indices = torch.randperm(len(X_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train_split, y_train_split = X_tensor[train_indices], y_tensor[train_indices]
        X_val, y_val = X_tensor[val_indices], y_tensor[val_indices]

        # 创建 DataLoader（自动 shuffle 和 batch）
        train_dataset = TensorDataset(X_train_split, y_train_split)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # ===== 训练阶段 =====
            self.model.train()  # 设置为训练模式（Dropout、BatchNorm 会激活）
            epoch_loss = 0
            num_batches = 0

            for X_batch, y_batch in train_loader:
                # 前向传播
                outputs = self.model(X_batch)

                # 计算损失
                loss = self.criterion(outputs, y_batch)

                # 反向传播
                self.optimizer.zero_grad()  # 清空上一步梯度
                loss.backward()              # 计算梯度（链式法则自动化）
                self.optimizer.step()        # 参数更新

                epoch_loss += loss.item()
                num_batches += 1

            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)

            # ===== 验证阶段 =====
            self.model.eval()  # 设置为评估模式
            with torch.no_grad():  # 不计算梯度，节省内存
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val)
                val_losses.append(val_loss.item())

            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_loss.item():.4f}")

        return train_losses, val_losses

    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()

# ===== 使用示例 =====
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    # 加载数据
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 创建模型和训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleNeuralNetwork(input_size=4, hidden_size=64, output_size=3)
    trainer = NeuralNetworkTrainer(model, learning_rate=0.01, device=device)

    # 训练
    train_losses, val_losses = trainer.train(
        X_train, y_train, epochs=100, batch_size=16, validation_split=0.2
    )

    # 评估
    y_pred = trainer.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.3f}")
```

### PyTorch 进阶实现（带 Dropout 和 BatchNorm）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

class AdvancedNeuralNetwork(nn.Module):
    """更复杂的网络，包含 Dropout 和 BatchNormalization"""

    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(AdvancedNeuralNetwork, self).__init__()

        layers = []
        prev_size = input_size

        # 动态构建隐层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))  # 批量归一化
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))     # Dropout 正则化
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # He 初始化
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)

# ===== 完整训练管道 =====
class AdvancedTrainer:
    """支持 Early Stopping 的训练器"""

    def __init__(self, model, learning_rate=0.001, device='cpu', patience=20):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.patience = patience  # Early stopping 耐心值
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        return val_loss / len(val_loader), correct / total

    def train(self, X_train, y_train, epochs=200, batch_size=32, validation_split=0.2):
        """训练，支持 Early Stopping"""
        # 数据准备
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

        val_size = int(len(X_train) * validation_split)
        indices = torch.randperm(len(X_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        train_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # Early Stopping 逻辑
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # 加载最佳模型
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break

            if (epoch + 1) % 30 == 0:
                print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}")

        return train_losses, val_losses, val_accs

    def predict(self, X):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1)
        return predictions.cpu().numpy()

# ===== 使用示例 =====
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report

    # 加载数据
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 创建模型和训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AdvancedNeuralNetwork(
        input_size=4,
        hidden_sizes=[64, 32],  # 2 个隐层
        output_size=3,
        dropout_rate=0.3
    )
    trainer = AdvancedTrainer(model, learning_rate=0.001, device=device, patience=30)

    # 训练
    train_losses, val_losses, val_accs = trainer.train(
        X_train, y_train, epochs=200, batch_size=16, validation_split=0.2
    )

    # 绘制训练曲线
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curve (Loss)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 评估
    y_pred = trainer.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 获取中间层特征（特征提取）
    print("\n===== 特征提取示例 =====")
    # 删除输出层，获取倒数第二层特征
    feature_extractor = nn.Sequential(*list(model.network.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        hidden_features = feature_extractor(X_test_tensor)
    print(f"Hidden features shape: {hidden_features.shape}")  # (n_samples, 32)
```

---

## 🔧 超参数调优

### 关键参数详解

| 参数 | 含义 | 对决策边界的影响 | 推荐范围 |
|---|---|---|---|
| **隐层数 (Depth)** | 网络的深度 | 更深 → 可学习更复杂的分层特征，但训练难（梯度消失）。根据数据复杂度选择 | 2-5 层（通常 2-3 足够） |
| **隐层宽度 (Width)** | 每层神经元数 | 更宽 → 更强的表达能力（每层特征更丰富），但参数增多，过拟合风险大 | 64-512（根据输入维度调整） |
| **learning_rate** | 梯度下降的步长 | 过大 → 震荡不收敛；过小 → 收敛慢，陷入局部最小值 | 0.001-0.1（通常 0.01） |
| **batch_size** | Mini-batch 大小 | 小 → 梯度噪声大，震荡更新但可能跳出局部最小值；大 → 平稳但可能陷入尖锐最小值 | 16-128 |
| **activation (隐层)** | 激活函数 | **关键！** ReLU → 快速收敛、避免梯度消失；sigmoid/tanh → 梯度消失、训练慢 | ReLU 或 Leaky ReLU ✅ |
| **dropout_rate** | Dropout 比例 | 增大 → 正则化强，防过拟合但可能欠拟合；减小 → 过拟合风险 | 0.2-0.5 |
| **epochs** | 训练轮数 | 更多 → 可能过拟合；太少 → 欠拟合 | Early stopping（监测验证损失） |

> [!TIP] learning_rate 与梯度下降的动态性
>
> - **固定 lr**：$W \leftarrow W - \eta \nabla L$，容易在平坦区域卡住（小梯度）或在陡峭区域震荡
> - **自适应 lr**（Adam, RMSprop）：
>   - 记录梯度的一阶矩（动量）和二阶矩（方差）
>   - 自动调整学习率：坡陡时降速，坡平时加速
>   - 公式：$W \leftarrow W - \frac{\eta}{\sqrt{v + \epsilon}} m$（其中 $m$ 是动量，$v$ 是方差）
> - **推荐**：使用 Adam（默认 $\eta=0.001$）或 RMSprop，而非朴素 SGD

### 调优实践（方法1：网格搜索）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import itertools

class HyperparameterGrid:
    """网格搜索超参数"""

    def __init__(self, X_train, y_train, device='cpu'):
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.LongTensor(y_train).to(device)
        self.device = device
        self.results = []

    def build_model(self, hidden_size, dropout_rate):
        """构建模型"""
        model = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 3)
        )
        return model

    def train_and_evaluate(self, model, learning_rate, hidden_size, dropout_rate, epochs=50):
        """训练单个模型并返回验证精度"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # 验证集分割
        val_size = int(len(self.X_train) * 0.2)
        indices = torch.randperm(len(self.X_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train_split = self.X_train[train_indices]
        y_train_split = self.y_train[train_indices]
        X_val = self.X_train[val_indices]
        y_val = self.y_train[val_indices]

        train_loader = DataLoader(
            TensorDataset(X_train_split, y_train_split),
            batch_size=16,
            shuffle=True
        )

        best_val_acc = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 验证
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_val).float().mean().item()
                best_val_acc = max(best_val_acc, val_acc)

        return best_val_acc

    def grid_search(self, param_grid, epochs=50):
        """执行网格搜索"""
        hidden_sizes = param_grid['hidden_size']
        dropout_rates = param_grid['dropout_rate']
        learning_rates = param_grid['learning_rate']

        total_trials = len(hidden_sizes) * len(dropout_rates) * len(learning_rates)
        trial = 0

        for hidden_size, dropout_rate, learning_rate in itertools.product(
            hidden_sizes, dropout_rates, learning_rates
        ):
            trial += 1
            print(f"Trial {trial}/{total_trials}: "
                  f"hidden={hidden_size}, dropout={dropout_rate:.2f}, lr={learning_rate:.1e}")

            model = self.build_model(hidden_size, dropout_rate)
            val_acc = self.train_and_evaluate(
                model, learning_rate, hidden_size, dropout_rate, epochs
            )

            self.results.append({
                'hidden_size': hidden_size,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'val_accuracy': val_acc
            })

            print(f"  → Val Accuracy: {val_acc:.4f}\n")

        # 返回最佳超参数
        best_result = max(self.results, key=lambda x: x['val_accuracy'])
        return best_result

# ===== 使用示例 =====
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 定义参数网格
param_grid = {
    'hidden_size': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'learning_rate': [1e-4, 1e-3, 1e-2]
}

searcher = HyperparameterGrid(X_train, y_train, device=device)
best_params = searcher.grid_search(param_grid, epochs=50)

print("\n===== 最佳超参数 =====")
print(f"Hidden Size: {best_params['hidden_size']}")
print(f"Dropout Rate: {best_params['dropout_rate']}")
print(f"Learning Rate: {best_params['learning_rate']:.1e}")
print(f"Validation Accuracy: {best_params['val_accuracy']:.4f}")

# 用最佳参数训练最终模型
final_model = nn.Sequential(
    nn.Linear(4, best_params['hidden_size']),
    nn.BatchNorm1d(best_params['hidden_size']),
    nn.ReLU(),
    nn.Dropout(best_params['dropout_rate']),
    nn.Linear(best_params['hidden_size'], 3)
)
trainer = AdvancedTrainer(final_model, learning_rate=best_params['learning_rate'], device=device)
```

### 调优实践（方法2：随机搜索 - 更高效）

```python
import random
from scipy.stats import loguniform

class RandomSearch:
    """随机搜索超参数（比网格搜索更高效）"""

    def __init__(self, X_train, y_train, device='cpu'):
        self.X_train = torch.FloatTensor(X_train).to(device)
        self.y_train = torch.LongTensor(y_train).to(device)
        self.device = device
        self.results = []

    def build_model(self, hidden_size, dropout_rate):
        model = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 3)
        )
        return model

    def train_and_evaluate(self, model, learning_rate, epochs=50):
        """训练单个模型"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        val_size = int(len(self.X_train) * 0.2)
        indices = torch.randperm(len(self.X_train))
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]

        X_train_split = self.X_train[train_indices]
        y_train_split = self.y_train[train_indices]
        X_val = self.X_train[val_indices]
        y_val = self.y_train[val_indices]

        train_loader = DataLoader(
            TensorDataset(X_train_split, y_train_split),
            batch_size=16,
            shuffle=True
        )

        best_val_acc = 0

        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                _, val_preds = torch.max(val_outputs, 1)
                val_acc = (val_preds == y_val).float().mean().item()
                best_val_acc = max(best_val_acc, val_acc)

        return best_val_acc

    def random_search(self, n_trials=20, epochs=50):
        """执行随机搜索"""
        best_result = None
        best_acc = 0

        for trial in range(n_trials):
            # 随机采样超参数
            hidden_size = random.choice([32, 64, 128, 256, 512])
            dropout_rate = random.uniform(0.1, 0.5)
            learning_rate = float(loguniform.rvs(1e-4, 1e-2))

            print(f"Trial {trial+1}/{n_trials}: "
                  f"hidden={hidden_size}, dropout={dropout_rate:.2f}, lr={learning_rate:.1e}")

            model = self.build_model(hidden_size, dropout_rate)
            val_acc = self.train_and_evaluate(model, learning_rate, epochs)

            result = {
                'hidden_size': hidden_size,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'val_accuracy': val_acc
            }
            self.results.append(result)

            if val_acc > best_acc:
                best_acc = val_acc
                best_result = result

            print(f"  → Val Accuracy: {val_acc:.4f}\n")

        return best_result

# ===== 使用示例 =====
searcher = RandomSearch(X_train, y_train, device=device)
best_params = searcher.random_search(n_trials=20, epochs=50)

print("\n===== 最佳超参数（随机搜索）=====")
print(f"Hidden Size: {best_params['hidden_size']}")
print(f"Dropout Rate: {best_params['dropout_rate']:.4f}")
print(f"Learning Rate: {best_params['learning_rate']:.1e}")
print(f"Validation Accuracy: {best_params['val_accuracy']:.4f}")
```

> [!WARNING] 常见陷阱
>
> 1. **learning_rate 过大或过小**：学习率是最敏感的超参数。监测训练损失曲线，应平稳下降而非震荡或停滞
> 2. **隐层数过深但未用 BatchNorm**：深网络易梯度消失，用 BatchNormalization 缓解。不用会卡在欠拟合
> 3. **激活函数用 sigmoid**：在隐层使用 sigmoid 会导致梯度消失（导数最大 0.25），改用 ReLU
> 4. **Dropout 比例过高**：太高的 dropout（如 0.8）会让网络完全随机化，无法学习
> 5. **不用验证集监测**：容易严重过拟合。需要设 `validation_split` 或 early stopping

---

## ⚖️ 优缺点与场景

### ✅ 优势 (Pros)

1. **通用近似器**：足够宽的单隐层可近似任意连续函数（Universal Approximation Theorem）
2. **自动特征学习**：无需手工特征工程，网络自动学习分层特征
3. **非线性表达能力强**：可捕捉复杂的非线性关系，远优于线性模型
4. **端到端可微**：反向传播允许优化任意可微的损失函数
5. **并行计算友好**：矩阵运算易于 GPU 加速

### ❌ 劣势 (Cons)

1. **可解释性差**：隐层特征无明确含义，难以理解网络的决策逻辑
2. **训练复杂**：超参数众多（深度、宽度、lr、batch_size 等），调参困难；易陷入局部最小值
3. **容易过拟合**：参数多，若缺乏正则化易严重过拟合
4. **需要大量数据**：相比树模型，ANN 需要更多样本才能泛化
5. **梯度消失/爆炸**：深网络训练不稳定（虽然 ReLU 缓解了梯度消失）
6. **训练时间长**：需要多个 epoch，单次反向传播计算量大

### 🎯 适用场景

| 场景 | 适用度 | 原因 |
|---|---|---|
| 图像识别（CNN） | ⭐⭐⭐⭐⭐ | 卷积利用空间局部性，参数共享；深层堆叠捕捉多尺度特征 |
| NLP（序列模型） | ⭐⭐⭐⭐⭐ | Transformer、LSTM 是序列建模的标准方案 |
| 非线性分类（中等规模） | ⭐⭐⭐⭐ | 优于 SVM，比随机森林更灵活 |
| 回归（连续值预测） | ⭐⭐⭐⭐ | 可用，效果与树模型接近 |
| 小数据集（<10K） | ⭐⭐ | 易过拟合，不如树模型稳定；需要正则化或预训练 |
| 结构化表格数据（大）| ⭐⭐⭐ | 可用，但通常不如 XGBoost；除非需要特殊的端到端学习 |
| 异构数据融合 | ⭐⭐⭐⭐⭐ | ANN 自然处理多模态数据，可融合图像+文本+结构化特征 |

---

## 💬 面试必问

> [!question] Q1: 推导反向传播的链式法则，为什么梯度消失在深网络中是关键问题？
>
> **答案框架**：
>
> **链式法则推导**：
>
> 对于 $L$ 层网络，计算 $\frac{\partial L}{\partial W^{(1)}}$ 涉及从输出层逆向链式相乘：
>
> $$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial a^{(L-1)}} \cdot \frac{\partial a^{(L-1)}}{\partial z^{(L-1)}} \cdots \frac{\partial z^{(2)}}{\partial a^{(1)}} \cdot \frac{\partial a^{(1)}}{\partial z^{(1)}} \cdot \frac{\partial z^{(1)}}{\partial W^{(1)}}$$
>
> **梯度消失分析**：
>
> 每一项包含激活函数导数 $\sigma'(z^{(l)})$。对 sigmoid：
> - $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$
> - 链式乘积：$\prod_{l=1}^{L-1} \sigma'(z^{(l)}) \leq 0.25^{L-1}$
> - 当 $L$ 很大时，如 $L=10$，乘积 $\leq 0.25^9 \approx 10^{-6}$，梯度接近 0
> - 第 1 层的权重几乎无法更新，网络无法学习早期特征
>
> **ReLU 的救赎**：
> - ReLU 导数：$\text{ReLU}'(z) = 1$（当 $z > 0$）
> - 链式乘积：$\prod_{l} \text{ReLU}'(z^{(l)}) = 1$，梯度不衰减
> - 虽然有 dead ReLU 问题（$z < 0$ 时导数为 0），但 Leaky ReLU 或 ELU 改进

> [!question] Q2: 如何区分过拟合和欠拟合？ANN 中的正则化方法有哪些？
>
> **答案核心**：
>
> **诊断方法**：
> - **过拟合**：训练损失 → 0，验证损失 → 高。特征：训练准确度高，测试低
> - **欠拟合**：训练损失仍很高，验证损失也高，差距不大
> - **工具**：绘制 loss 曲线（epochs vs train_loss/val_loss）
>
> **正则化技术对比**：
>
> | 方法 | 原理 | 何时用 | 效果 |
> |---|---|---|---|
> | **L2 正则化** | $L' = L + \lambda \sum W^2$，惩罚大权重 | 总是推荐 | 轻度过拟合，稳定 |
> | **L1 正则化** | $L' = L + \lambda \sum \|W\|$，产生稀疏权重 | 需要特征选择 | 中度过拟合 |
> | **Dropout** | 随机关闭神经元，强制冗余学习 | 中大型网络（>100 参数） | 中-重度过拟合 |
> | **Early Stopping** | 监测验证损失，验证不降时停止 | 总是用 | 简单有效 |
> | **BatchNormalization** | 每层归一化，稳定训练 | 深网络（>3 层） | 加速收敛+正则化效果 |
> | **数据增强** | 扩展训练数据 | 小数据集 | 实质性改进 |
>
> **推荐组合**：L2 正则化 + Dropout + Early Stopping + BatchNorm

> [!question] Q3: 解释批量归一化（BatchNormalization）的作用，为什么它既加速训练又防过拟合？
>
> **答案核心**：
>
> **公式**：
>
> $$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
> $$y = \gamma \hat{x} + \beta$$
>
> 其中：
> - $\mu_B, \sigma_B$：mini-batch 的均值和方差
> - $\gamma, \beta$：可学习的缩放和偏移参数
>
> **加速训练的原因**：
> 1. **解决 Internal Covariate Shift**：深网络中，每层输入分布随参数更新而变化（输入时不稳定），BN 稳定分布
> 2. **增大学习率上界**：分布稳定，梯度爆炸风险降低，可用更大的 lr
> 3. **减少对初始化的敏感性**：BN 使网络对初始化鲁棒
>
> **防过拟合的原因**：
> 1. **正则化效果**：使用 mini-batch 统计而非全局统计，引入噪声（类似 Dropout）
> 2. **简化优化景观**：使损失曲面更平滑，陷入尖锐最小值（容易过拟合）的概率降低
>
> **实战建议**：
> - 总是在隐层加 BN（在激活函数前）：`Dense → BatchNorm → ReLU`
> - 注意：BN 的参数化改变了后续层的输入，可能影响权重初始化策略
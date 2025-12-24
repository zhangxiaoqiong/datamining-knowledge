---
tags: [算法, 深度学习, 序列模型, RNN, LSTM, GRU]
math: true
difficulty: 困难
---

# 循环神经网络 (RNN, LSTM, GRU)

## 💡 核心直觉

- **一句话定义**：通过在时间步间共享参数，使网络具有"记忆"，能处理任意长度的序列，捕捉序列中的长期依赖。

- **解决问题**：解决了全连接神经网络无法处理变长序列、丢失时间信息的问题。RNN 推动了 NLP 从 n-gram、特征工程时代进入深度学习时代。

- **核心逻辑**：RNN = 参数共享 + 隐状态传递。每个时刻都用相同的参数集，隐状态 $h_t$ 既是输出，也是下一时刻的输入，形成"记忆"。

- **几何意义**：
  - **全连接网络**：每个输出独立计算，无时间依赖
  - **RNN**：隐状态沿时间链传递，每层可"看到"历史信息（通过 $h_{t-1}$）
  - **LSTM**：增加了控制信息流的"闸门"，能选择记住或遗忘信息

- **杀手锏 (Killer Feature)**：**参数共享** + **权重递归** 使序列学习成为可能。RNN 是所有序列建模的经典范式：机器翻译（seq2seq）、语言模型、时间序列预测等。

> [!TIP] RNN vs LSTM vs Transformer 的对比
>
> ```
> RNN:          x₀ → RNN → h₀ → RNN → h₁ → RNN → h₂ → ... (逐步处理，梯度消失)
>               ↓            ↓            ↓
>               y₀           y₁           y₂
>
> LSTM:         x₀ → [输入门|遗忘门|输出门] → h₀ → ... (门控制信息流)
>               ↓           ↓
>               y₀      (细粒度控制)
>
> Transformer:  [x₀, x₁, x₂, ...] → Self-Attention → [h₀, h₁, h₂, ...] (完全并行)
>               ↓                                      ↓
>               同时处理所有 token，无序列依赖
> ```

---

## 📐 数学原理

### 1. 基础 RNN 的前向传播

在时刻 $t$，RNN 的计算为：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$y_t = W_{hy} h_t + b_y$$

其中：
- $x_t$：时刻 $t$ 的输入，形状 $(batch, input\_size)$
- $h_t$：时刻 $t$ 的隐状态，形状 $(batch, hidden\_size)$，也是"记忆"
- $y_t$：时刻 $t$ 的输出，形状 $(batch, output\_size)$
- $W_{hh}$：隐状态权重，形状 $(hidden\_size, hidden\_size)$，**跨时间共享**
- $W_{xh}$：输入权重，形状 $(input\_size, hidden\_size)$，**跨时间共享**
- $W_{hy}$：输出权重，形状 $(hidden\_size, output\_size)$，**跨时间共享**

**关键**：参数 $W_{hh}, W_{xh}, W_{hy}$ 在所有时刻 $t$ 共享，这就是 RNN 的精髓。

> [!ABSTRACT] 参数共享的含义
>
> - **优势**：参数数量与序列长度无关，能处理任意长序列
> - **劣势**：同一参数在多个时间步重复使用，梯度在反向传播时累积相乘，易梯度消失/爆炸

### 2. 反向传播通过时间 (BPTT - BackPropagation Through Time)

从时刻 $T$ 反向传播到时刻 1，计算 $\frac{\partial L}{\partial W_{hh}}$ 需链式法则：

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L}{\partial h_t} \frac{\partial h_t}{\partial W_{hh}}$$

但关键是 $\frac{\partial h_t}{\partial W_{hh}}$ 依赖所有之前的时刻：

$$\frac{\partial h_t}{\partial W_{hh}} = \sum_{k=1}^{t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

其中链式涉及：

$$\frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(1 - h_t^2) W_{hh}$$

（tanh 的导数为 $1 - h^2$）

**梯度消失问题**：当 $t$ 很大时，梯度需要乘以 $t-1$ 次的 $\frac{\partial h_{t'}}{\partial h_{t'-1}}$，若其绝对值 $< 1$，梯度指数衰减：

$$\left|\frac{\partial h_T}{\partial h_1}\right| \approx \left|\text{tanh}'(z) \cdot W_{hh}\right|^{T-1}$$

若 $|\text{tanh}'| \leq 1$ 且 $\|W_{hh}\| < 1$，则 $|\frac{\partial h_T}{\partial h_1}| \to 0$。

### 3. LSTM (Long Short-Term Memory) 的门机制

LSTM 通过三个门控制信息流，解决梯度消失问题：

**输入门**（决定保留多少新信息）：
$$i_t = \sigma(W_{ii} x_t + W_{hi} h_{t-1} + b_i)$$

**遗忘门**（决定丢弃多少旧信息）：
$$f_t = \sigma(W_{if} x_t + W_{hf} h_{t-1} + b_f)$$

**候选隐状态**（新的信息）：
$$\tilde{h}_t = \tanh(W_{ih} x_t + W_{hh} h_{t-1} + b_h)$$

**单元状态更新**（长期记忆，核心创新）：
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{h}_t$$

**输出门**（决定输出多少信息）：
$$o_t = \sigma(W_{io} x_t + W_{ho} h_{t-1} + b_o)$$

**隐状态更新**：
$$h_t = o_t \odot \tanh(c_t)$$

其中：
- $\sigma$：sigmoid 函数，输出范围 $[0, 1]$（门的"开度"）
- $\odot$：element-wise 乘法（Hadamard积）
- $c_t$：单元状态（cell state），独立于隐状态，传递长期依赖

> [!TIP] LSTM 为什么解决梯度消失？
>
> 关键是**单元状态的直接传递**：
>
> $$c_t = f_t \odot c_{t-1} + \ldots$$
>
> 反向传播时：
> $$\frac{\partial L}{\partial c_{t-1}} = \frac{\partial L}{\partial c_t} \cdot f_t$$
>
> 若 $f_t$ （遗忘门）接近 1，梯度不会衰减，可传递很多时刻。
> 与传统 RNN 的 $\text{tanh}' \cdot W_{hh}$ 不同，这里是**相乘而非链式**，梯度可长期保持。

### 4. GRU (Gated Recurrent Unit) - LSTM 的简化版

GRU 合并了遗忘门和输入门，参数更少：

**重置门**：
$$r_t = \sigma(W_{ir} x_t + W_{hr} h_{t-1} + b_r)$$

**更新门**：
$$z_t = \sigma(W_{iz} x_t + W_{hz} h_{t-1} + b_z)$$

**候选隐状态**：
$$\tilde{h}_t = \tanh(W_{ih} x_t + W_{hh} (r_t \odot h_{t-1}) + b_h)$$

**隐状态更新**：
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

**对比**：
| 特性 | LSTM | GRU |
|---|---|---|
| 参数数量 | 多（3 个门 + cell state） | 少（2 个门，无 cell state） |
| 梯度流 | 通过 cell state 直接传递 | 通过隐状态加权平均 |
| 效果 | 略优，尤其长序列 | 计算快，中等序列足够 |
| 推荐 | 需要精度时 | 计算资源紧张时 |

---

## 💻 算法实现

### PyTorch 完整实现（LSTM 编码器-解码器）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class SimpleRNNCell(nn.Module):
    """基础 RNN 单元（演示用）"""

    def __init__(self, input_size, hidden_size):
        super(SimpleRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 参数共享
        self.W_ih = nn.Linear(input_size, hidden_size)   # 输入 → 隐状态
        self.W_hh = nn.Linear(hidden_size, hidden_size)  # 隐状态 → 隐状态
        self.tanh = nn.Tanh()

    def forward(self, x, h_prev):
        """
        Args:
            x: (batch_size, input_size)
            h_prev: (batch_size, hidden_size)
        Returns:
            h: (batch_size, hidden_size)
        """
        h = self.tanh(self.W_ih(x) + self.W_hh(h_prev))
        return h

class LSTMModel(nn.Module):
    """基于 LSTM 的序列模型"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(LSTMModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状：(batch, seq_len, input_size)
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch_size, seq_len, input_size)
            hidden: tuple of (h_0, c_0)，若为 None 则自动初始化为 0
        Returns:
            output: (batch_size, seq_len, output_size) 或 (batch_size, output_size)
            hidden: (h_n, c_n)
        """
        lstm_out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden_size)

        # 仅用最后时刻的输出进行分类
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        output = self.fc(last_output)     # (batch_size, output_size)

        return output, hidden

class BidirectionalLSTM(nn.Module):
    """双向 LSTM（可捕捉前向和后向信息）"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.5):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # 关键：双向
            dropout=dropout if num_layers > 1 else 0
        )

        # 输出层：双向后隐状态维度加倍
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, hidden = self.lstm(x)  # (batch, seq_len, hidden_size * 2)

        # 最后时刻的输出（包含前向和后向信息）
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size * 2)
        output = self.fc(last_output)

        return output

class Seq2SeqModel(nn.Module):
    """序列到序列模型（编码器-解码器）"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=2):
        super(Seq2SeqModel, self).__init__()

        # 编码器：读取输入序列，生成上下文向量
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # 解码器：从上下文向量生成输出序列
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)
        self.output_size = output_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_len, input_size) - 源序列
            tgt: (batch_size, tgt_len, output_size) - 目标序列（用于 teacher forcing）
            teacher_forcing_ratio: 使用真实标签的概率
        Returns:
            output: (batch_size, tgt_len, output_size) - 预测的目标序列
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]

        # 编码器：处理输入序列，获得上下文向量（最后的隐状态）
        _, (hidden, cell) = self.encoder(src)  # hidden: (num_layers, batch, hidden_size)

        # 解码器：逐时刻生成输出
        decoder_input = tgt[:, 0, :].unsqueeze(1)  # 第一个时刻的目标（batch, 1, output_size）
        outputs = []

        for t in range(1, tgt_len):
            # 解码一步
            decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            # (1, batch, hidden_size)

            # 输出层
            output = self.fc(decoder_output.squeeze(1))  # (batch, output_size)
            outputs.append(output)

            # 决定下一时刻的输入：teacher forcing 或 自回归
            if np.random.random() < teacher_forcing_ratio:
                decoder_input = tgt[:, t, :].unsqueeze(1)  # 真实下一时刻
            else:
                decoder_input = output.unsqueeze(1)  # 模型预测的下一时刻

        outputs = torch.stack(outputs, dim=1)  # (batch, tgt_len-1, output_size)
        return outputs

# ===== 训练器 =====
class RNNTrainer:
    """RNN 训练器"""

    def __init__(self, model, learning_rate=1e-3, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)  # (batch, seq_len, input_size)
            y_batch = y_batch.to(self.device)

            output, _ = self.model(X_batch)
            loss = self.criterion(output, y_batch)

            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                output, _ = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(output, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

        return val_loss / len(val_loader), correct / total

    def train(self, train_loader, val_loader, epochs=10):
        """完整训练循环"""
        train_losses = []
        val_losses = []
        val_accs = []

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val Acc: {val_acc:.4f}")

        return train_losses, val_losses, val_accs

# ===== 使用示例 =====
if __name__ == "__main__":
    # 生成模拟时间序列数据
    batch_size = 32
    seq_len = 50
    input_size = 10
    hidden_size = 128
    num_classes = 5
    num_samples = 500

    # 随机生成序列和标签
    X = torch.randn(num_samples, seq_len, input_size)
    y = torch.randint(0, num_classes, (num_samples,))

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=2,
        output_size=num_classes,
        dropout=0.5
    )

    # 训练
    trainer = RNNTrainer(model, learning_rate=1e-3, device=device)
    train_losses, val_losses, val_accs = trainer.train(train_loader, val_loader, epochs=10)

    print("\n训练完成！")
    print(f"最终验证精度：{val_accs[-1]:.4f}")

    # ===== 对比不同模型 =====
    print("\n===== 模型对比 =====")

    # 模型 1：单层 LSTM
    model1 = LSTMModel(input_size=input_size, hidden_size=64, num_layers=1, output_size=num_classes)
    print(f"单层 LSTM 参数数：{sum(p.numel() for p in model1.parameters())}")

    # 模型 2：双向 LSTM
    model2 = BidirectionalLSTM(input_size=input_size, hidden_size=64, num_layers=2, output_size=num_classes)
    print(f"双向 LSTM 参数数：{sum(p.numel() for p in model2.parameters())}")

    # 模型 3：Seq2Seq
    model3 = Seq2SeqModel(input_size=input_size, hidden_size=hidden_size, output_size=input_size)
    print(f"Seq2Seq 参数数：{sum(p.numel() for p in model3.parameters())}")
```

---

## 🔧 超参数调优

### 关键参数详解

| 参数 | 含义 | 对性能的影响 | 推荐值 |
|---|---|---|---|
| **hidden_size** | LSTM 隐状态维度 | 越大 → 表达能力强但参数多；需平衡 | 128-512 |
| **num_layers** | LSTM 堆叠层数 | 越深 → 更复杂的特征，但难以训练（梯度消失）；通常 2-3 足够 | 2-3 |
| **dropout** | Dropout 比例 | 防过拟合；多层 LSTM 才有效（层间 dropout）| 0.3-0.5 |
| **learning_rate** | 学习率 | RNN 对 lr 敏感；太大易梯度爆炸，太小收敛慢 | 1e-3-1e-4（需梯度裁剪） |
| **batch_size** | 批量大小 | 小 batch → 梯度噪声大但逃离局部最小值；大 batch → 稳定但可能欠拟合 | 32-64 |
| **seq_len** | 序列长度 | 长序列 → BPTT 难度大，梯度消失/爆炸风险高；可用截断 | 50-100（可截断） |
| **gradient_clip** | 梯度裁剪阈值 | RNN 必须，防止梯度爆炸；通常 1.0 | 1.0 |
| **bidirectional** | 是否双向 | 双向 → 参数翻倍，精度略高；只能用于非生成任务 | True（分类）/ False（生成） |

> [!TIP] RNN 特有的调优技巧
>
> 1. **梯度裁剪（必须）**：RNN 易梯度爆炸（不仅消失）
>    ```python
>    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
>    ```
> 2. **序列长度截断**：长序列 BPTT 计算量爆炸，可截断为 32-50
> 3. **门机制选择**：LSTM（精度）vs GRU（速度）
> 4. **双向限制**：双向只适合编码阶段（分类、NER），解码（生成）必须单向
> 5. **初始化**：正交初始化比高斯更稳定
>    ```python
>    nn.init.orthogonal_(lstm.weight_hh_l0)
>    ```

---

## ⚖️ 优缺点与场景

### ✅ 优势 (Pros)

1. **天然处理变长序列**：参数共享使模型对任意长度序列适用
2. **捕捉长期依赖**（LSTM/GRU）：门机制直接解决梯度消失
3. **双向建模**：可同时利用前向和后向信息
4. **参数高效**：相比同等深度的全连接网络少很多参数
5. **可解释的隐状态**：$h_t$ 显式传递，可视化理解

### ❌ 劣势 (Cons)

1. **串行计算**：无法并行处理序列，训练慢（相比 Transformer）
2. **长序列困难**：即使 LSTM，太长序列（>1000）仍梯度消失
3. **BPTT 计算复杂**：需从输出反向计算到输入，内存占用大
4. **梯度爆炸**：RNN 特有（CNN/Transformer 无），需梯度裁剪
5. **已被 Transformer 超越**：同等参数下 Transformer 精度更高

### 🎯 适用场景

| 场景 | 适用度 | 原因 |
|---|---|---|
| **语言建模** | ⭐⭐⭐⭐ | 经典用途，虽 Transformer 更优但仍用 |
| **机器翻译** | ⭐⭐⭐ | seq2seq 标配，但现在 Transformer 更佳 |
| **序列标注（NER、POS）** | ⭐⭐⭐⭐ | 双向 LSTM 标准，预训练 Transformer 超越 |
| **情感分析** | ⭐⭐⭐⭐ | 可用，但 BERT 更优 |
| **时间序列预测** | ⭐⭐⭐⭐⭐ | 擅长，捕捉时间依赖；有些时序任务 LSTM 仍最优 |
| **文本生成** | ⭐⭐⭐ | 可用，但 Transformer/GPT 更强 |
| **文本分类** | ⭐⭐⭐ | 可用，但不如 BERT/Transformer |
| **对话系统** | ⭐⭐⭐ | seq2seq 曾经主流，现 Transformer + 微调更优 |

---

## 💬 面试必问

> [!question] Q1: 推导 RNN 的梯度消失和梯度爆炸问题，LSTM 是如何解决的？
>
> **答案框架**：
>
> **梯度消失分析**：
>
> RNN 的参数 $W_{hh}$ 在所有时刻共享。从时刻 1 到时刻 $T$ 的梯度链式：
>
> $$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} = \prod_{t=2}^{T} \text{tanh}'(z_t) \cdot W_{hh}$$
>
> 其中 $\text{tanh}'(z) = 1 - \tanh^2(z) \leq 1$，所以：
>
> $$\left|\frac{\partial h_T}{\partial h_1}\right| \leq \|W_{hh}\|^{T-1} \cdot \prod_t (1 - \tanh^2(z_t))$$
>
> 当 $T$ 很大时（如 100 步），若 $\|W_{hh}\| < 1$，梯度呈指数衰减：$0.9^{99} \approx 10^{-5}$
>
> **梯度爆炸**：若 $\|W_{hh}\| > 1$，反而梯度指数增长，导致数值溢出。
>
> **LSTM 的解决**：
>
> LSTM 的单元状态 $c_t$ 有专门的"高速通道"：
>
> $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{h}_t$$
>
> 反向传播时的梯度：
>
> $$\frac{\partial L}{\partial c_{t-1}} = \frac{\partial L}{\partial c_t} \odot f_t$$
>
> 这里是 **element-wise 乘法而非矩阵乘法**，且 $f_t$ 通常接近 1（遗忘门），所以梯度不会指数衰减。可证明LSTM 梯度 $\leq 1 + O(\text{gate变化})$，不随 $T$ 指数衰减。
>
> **对比**：RNN 梯度 $\propto 0.9^T$，LSTM 梯度保持 $O(1)$

> [!question] Q2: LSTM 的三个门（输入门、遗忘门、输出门）各自的作用是什么？能否用更少门数代替？
>
> **答案核心**：
>
> **三个门的角色分工**：
> - **遗忘门** $f_t$：控制过去信息的保留程度（$\approx 1$ 时保留，$\approx 0$ 时遗忘）
> - **输入门** $i_t$：控制新信息的加入量（$\approx 1$ 时接纳新信息，$\approx 0$ 时忽视）
> - **输出门** $o_t$：控制隐状态输出量（$\approx 1$ 时完全输出，$\approx 0$ 时隐藏）
>
> **直观例子**：在语言建模中，长距离的代词回指
> - "我走进了一家咖啡厅...（许多词后）...我点了一杯咖啡"
> - 第一个"我"对应最后的"我"，中间许多不相关的词需遗忘（遗忘门）
> - 关键词（如动词"点"）需当时输出影响决策（输出门）
>
> **能否用更少门数**：
> - **单门（GRU）**：合并输入门和遗忘门为更新门，参数少 ~33%，效果通常相近
> - **无门（标准 RNN）**：不行，会梯度消失（已证明）
> - **理论下界**：需至少某种形式的"选择机制"，三门是完整设计，少于三门必然损失表达能力

> [!question] Q3: RNN vs Transformer 的本质差异是什么？为什么 Transformer 在 NLP 中逐渐取代了 RNN？
>
> **答案核心**：
>
> **本质差异**：
>
> | 维度 | RNN | Transformer |
> |---|---|---|
> | **依赖关系** | **顺序依赖**（逐步）| **全连接依赖**（一步） |
> | **计算复杂度** | $O(T \cdot d^2)$ | $O(T^2 \cdot d)$ |
> | **梯度路径长度** | $O(T)$（梯度消失） | $O(1)$（直接） |
> | **并行性** | 无（逐步） | 完全（一步） |
> | **位置偏好** | 天然（时间步）| 需显式编码 |
>
> **梯度流对比**：
> - RNN：最后一步的梯度回传到第一步，需乘以 $T$ 个中间梯度，指数衰减
> - Transformer：任意两位置间的梯度路径长度为 1（self-attention），直接无衰减
>
> **为什么 Transformer 取代 RNN**：
> 1. **训练快**：完全并行，相同计算预算下 epoch 数更多
> 2. **精度高**：梯度更好，相同参数下精度更优
> 3. **长序列友好**：虽然 $O(T^2)$ 内存，但直接依赖无梯度消失，实际可处理更长序列
> 4. **预训练友好**：大规模预训练（BERT、GPT）更易收敛和扩展
> 5. **缺点**（相对）：$O(T^2)$ 内存对超长序列困难，但工程上可用分块、稀疏等优化
>
> **RNN 的坚持场景**：
> - 在线/流式推理（Transformer 需缓存所有历史）
> - 超长序列（>100K tokens）
> - 实时性要求（逐词输出 Transformer 有延迟）
> - 某些专业领域（如金融时序预测中 LSTM 仍领先）
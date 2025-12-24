---
tags: [算法, 深度学习, NLP, 序列模型, 注意力机制]
math: true
difficulty: 困难
---

# Transformer：从自注意力到序列建模

## 💡 核心直觉

- **一句话定义**：通过自注意力机制（Self-Attention）实现序列中任意位置间的直接交互，完全抛弃循环（RNN）和卷积（CNN），用纯并行的矩阵操作处理序列数据的革命性架构。

- **解决问题**：解决了 RNN（LSTM/GRU）的梯度消失和串行计算瓶颈、CNN 感受野有限的问题。Transformer 使得 NLP 从 LSTM 时代跃进到大规模预训练模型时代（BERT、GPT）。

- **核心逻辑**：Transformer = 编码器 + 解码器，每个都是多层的自注意力堆叠。关键创新是**自注意力**：每个 token 与所有 token 的相关性通过点积相似度计算，无需循环逐步处理。

- **几何意义**：
  - **传统 RNN**：序列依次处理，每一步对前面历史的依赖通过隐态传递（易梯度消失）
  - **Transformer**：所有位置同时计算，每个位置通过**加权求和**与其他位置交互，权重由 query-key 相似度决定
  - 完全并行 → 训练快 100 倍，可扩展到十亿参数

- **杀手锏 (Killer Feature)**：**完全可并行化** + **长距离依赖直接捕捉**（不需逐步传递）。奠定了大语言模型（LLM）时代：ChatGPT、Claude 等所有现代 LLM 的基础架构。

> [!TIP] Transformer 架构概览
>
> ```
> 输入序列 (tokens)
>    ↓
> 嵌入 + 位置编码
>    ↓
> ┌─────────────────────┐
> │  Encoder Layer × N  │
> │  ┌───────────────┐  │
> │  │ Multi-Head    │  │ ← 自注意力：token 间的全交互
> │  │ Self-Attention│  │
> │  ├───────────────┤  │
> │  │ Feed-Forward  │  │ ← 逐位置的非线性变换
> │  └───────────────┘  │
> └─────────────────────┘
>    ↓ (编码表示)
> ┌─────────────────────┐
> │  Decoder Layer × N  │ ← 可选（仅用于生成任务）
> │  ┌───────────────┐  │
> │  │ Cross-Attention│ │ ← encoder 输出与 decoder 的交互
> │  │(encoder→decoder)│ │
> │  └───────────────┘  │
> └─────────────────────┘
>    ↓
> 输出（分类、文本、回归等）
> ```

---

## 📐 数学原理

### 1. 缩放点积注意力 (Scaled Dot-Product Attention)

**核心公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q$ (Query)：形状 $(seq\_len, d_k)$，查询向量
- $K$ (Key)：形状 $(seq\_len, d_k)$，键向量
- $V$ (Value)：形状 $(seq\_len, d_v)$，值向量
- $d_k$：键的维度
- $\sqrt{d_k}$：缩放因子，防止 softmax 进入饱和区（梯度消失）

**直观解释**：
1. **相似度计算**：$QK^T$ 计算每个 query 与所有 key 的点积相似度
   - 形状：$(seq\_len, seq\_len)$，称为**注意力权重矩阵**
2. **缩放**：除以 $\sqrt{d_k}$ 确保梯度稳定（点积值随 $d_k$ 增大）
3. **归一化**：softmax 转换为概率分布（每行求和为 1）
4. **加权求和**：用注意力权重对 $V$ 加权求和，得到输出

> [!ABSTRACT] 为什么缩放很重要？
>
> 不缩放时，$QK^T$ 的每个元素是 $d_k$ 个独立项的和，方差为 $d_k$。
> 当 $d_k$ 很大时（如 512 或 1024），$QK^T$ 中的值会很大，导致 softmax 接近 0 或 1（梯度消失）。
> 除以 $\sqrt{d_k}$ 将方差归一化为 1，使 softmax 输出更均衡。

### 2. 多头注意力 (Multi-Head Attention)

单层注意力难以捕捉多种关系，用多个**头**并行计算：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

其中每个 head：

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- $h$ = 头数（通常 8 或 16）
- $W_i^Q, W_i^K, W_i^V$：可学习的线性投影矩阵（每个头独立）
- $W^O$：输出线性投影矩阵

**形状示例**（假设 $d_{model} = 512$，$h = 8$）：
- 原始 $Q, K, V$：$(seq\_len, 512)$
- 每个 head 中的 $QW_i^Q$：$(seq\_len, 64)$（$512 / 8 = 64$）
- Attention 输出：$(seq\_len, 64)$
- 拼接后：$(seq\_len, 512)$
- 最终输出：$(seq\_len, 512)$

**直观意义**：
- head 1 可能学习语法关系（如主谓搭配）
- head 2 可能学习语义关系（如同义词）
- head 3 可能学习句子位置关系
- ... 多个 head 并行，信息融合更丰富

### 3. 前向网络 (Position-wise Feed-Forward Network)

每个 token 独立通过两层全连接（与 ANN 的隐层类似）：

$$\text{FFN}(x) = \max(0, xW_1 + b_1) W_2 + b_2$$

- 第一层：$(seq\_len, d_{model}) \to (seq\_len, d_{ff})$，通常 $d_{ff} = 4 \times d_{model}$
- ReLU 激活
- 第二层：$(seq\_len, d_{ff}) \to (seq\_len, d_{model})$

**关键**：这个操作**逐位置**应用（对序列中每个位置独立计算），不涉及位置间的交互。

### 4. 位置编码 (Positional Encoding)

RNN/CNN 天然有位置信息（时间步/空间步）。Transformer 完全并行，需显式编码位置：

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

其中：
- $pos$：序列中的位置（0, 1, 2, ...)
- $i$：维度索引（0 到 $d_{model}/2 - 1$）
- 偶数维用 sin，奇数维用 cos

**直观解释**：
- 不同位置的编码向量不同（捕捉位置信息）
- 相对位置差一定大小时，PE 之间的关系是线性的（位置间隔有规律）
- 无需学习，对所有位置直接计算

### 5. 残差连接与层归一化

每个子层后跟**残差 + 层归一化**（Residual + LayerNorm）：

$$\text{out} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**两大作用**：
- **残差连接**：梯度直接回传，解决深网络梯度消失
- **层归一化**：每个样本独立归一化，稳定训练

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中 $\mu, \sigma$ 是沿特征维度计算（与 BatchNorm 沿样本维度不同）。

---

## 💻 算法实现

### PyTorch 完整实现（从零构建 Transformer Block）

```python
import torch
import torch.nn as nn
import torch.optim as optim
import math
from torch.utils.data import DataLoader, TensorDataset

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        Args:
            Q, K, V: (batch_size, seq_len, d_k or d_v)
            mask: (batch_size, seq_len) 或 (batch_size, seq_len, seq_len)
        Returns:
            output: (batch_size, seq_len, d_v)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        d_k = Q.shape[-1]

        # 1️⃣ 相似度计算：QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # 2️⃣ Mask（可选，用于防止看到未来信息）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3️⃣ Softmax 归一化
        attention_weights = torch.softmax(scores, dim=-1)

        # 4️⃣ Dropout
        attention_weights = self.dropout(attention_weights)

        # 5️⃣ 加权求和
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """多头注意力"""

    def __init__(self, d_model=512, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性投影
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.shape[0]

        # 1️⃣ 线性投影
        Q = self.W_Q(Q)  # (batch_size, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # 2️⃣ 分割成多个 head
        # (batch_size, seq_len, d_model) → (batch_size, seq_len, num_heads, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # 现在形状：(batch_size, num_heads, seq_len, d_k)

        # 3️⃣ 多头注意力（并行）
        context, attention_weights = self.attention(Q, K, V, mask)

        # 4️⃣ 拼接各 head
        context = context.transpose(1, 2).contiguous()  # (batch_size, seq_len, num_heads, d_k)
        context = context.view(batch_size, -1, self.num_heads * self.d_k)  # (batch_size, seq_len, d_model)

        # 5️⃣ 输出线性投影
        output = self.W_O(context)

        return output, attention_weights

class FeedForwardNetwork(nn.Module):
    """前向网络（逐位置）"""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class TransformerBlock(nn.Module):
    """Transformer 编码器块"""

    def __init__(self, d_model=512, num_heads=8, d_ff=2048, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len, d_model)
        mask: 用于屏蔽填充符号或因果mask
        """
        # 自注意力 + 残差 + 层归一化
        attn_output, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前向网络 + 残差 + 层归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model=512, max_seq_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # 计算位置编码
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维：sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维：cos

        # 注册为 buffer（不是参数，但参与前向传播）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """x: (batch_size, seq_len, d_model)"""
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)

class SimpleTransformer(nn.Module):
    """简单的 Transformer（编码器部分）"""

    def __init__(self, vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048,
                 dropout=0.1, max_seq_len=5000, num_classes=10):
        super(SimpleTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        """
        x: (batch_size, seq_len)
        mask: (batch_size, seq_len) - 1 表示有效，0 表示填充
        """
        # 嵌入 + 位置编码
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoding(x)

        # 通过多层 Transformer Block
        for block in self.transformer_blocks:
            x = block(x, mask)

        # 层归一化
        x = self.layer_norm(x)

        # 全局平均池化
        if mask is not None:
            mask = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)  # 仅对有效位置求平均
        else:
            x = x.mean(dim=1)  # (batch_size, d_model)

        # 分类
        output = self.fc(x)  # (batch_size, num_classes)

        return output

# ===== 训练示例 =====
class TransformerTrainer:
    """Transformer 训练器"""

    def __init__(self, model, learning_rate=1e-4, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train_epoch(self, train_loader):
        """训练一个 epoch"""
        self.model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # 梯度裁剪
            self.optimizer.step()

            epoch_loss += loss.item()

        self.scheduler.step()
        return epoch_loss / len(train_loader)

    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
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
    # 生成简单的模拟数据
    vocab_size = 1000
    seq_len = 50
    batch_size = 32
    num_samples = 1000

    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(X, y)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 创建模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        num_classes=10
    )

    # 训练
    trainer = TransformerTrainer(model, learning_rate=1e-4, device=device)
    train_losses, val_losses, val_accs = trainer.train(train_loader, val_loader, epochs=10)

    print("\n训练完成！")
```

---

## 🔧 超参数调优

### 关键参数详解

| 参数 | 含义 | 对性能的影响 | 推荐值 |
|---|---|---|---|
| **d_model** | 隐藏维度 | 越大 → 表达能力强但参数多；深度学习实践中 256-1024 常见 | 512（标准）, 768（BERT-base） |
| **num_heads** | 多头数 | 必须整除 d_model；太少 → 单一关注，太多 → 过度分割信息 | 8（标准）, 12（BERT） |
| **d_ff** | FFN 的隐层维度 | 通常 $d_{ff} = 4 \times d_{model}$；越大越有表达力但更耗内存 | $4 \times d_{model}$ |
| **num_layers** | 编码器层数 | 越深 → 越复杂的模式，但 warmup 需调整；深层时需 LayerNorm 和残差 | 6（标准）, 12（BERT-base） |
| **dropout** | Dropout 比例 | 防过拟合；通常 0.1-0.2；小数据集用高 dropout | 0.1 |
| **learning_rate** | 学习率 | Transformer 对 lr 敏感；用 warmup 和学习率衰减；太大易散发 | 1e-4（with warmup）|
| **warmup_steps** | 学习率预热步数 | 初期小 lr 逐渐增大，稳定训练；通常总步数的 10% | 4000（10% of steps） |
| **max_seq_len** | 最大序列长度 | 位置编码的范围；超过此值需重新编码或 alibi（ALiBi） | 512-2048 |

> [!TIP] Transformer 特有的调优技巧
>
> 1. **Warmup + 学习率衰减**：不同于 CNN，Transformer 需要 warmup 和余弦衰减
>    $$\text{lr}_t = d_{model}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup\_steps}^{-1.5})$$
>
>    或用**余弦衰减**（更实用）：
>    $$\text{lr}_t = \text{lr}_{init} \cdot \frac{1 + \cos(\pi \cdot t / T)}{2}$$
>
>    其中 $t$ 是当前步，$T$ 是总步数。实现示例：
>    ```python
>    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
>    ```
> 2. **梯度裁剪（Gradient Clipping）**：防止梯度爆炸（Transformer 易出现）
>    ```python
>    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
>    ```
> 3. **批量归一化顺序**：Post-LN（LayerNorm 在残差后）vs Pre-LN（LayerNorm 在残差前）
>    - Pre-LN 更稳定，但 Post-LN（标准 Transformer）效果更好
> 4. **位置编码策略**：绝对位置编码 vs 相对位置偏差（ALiBi）
>    - 绝对：固定且参数无关
>    - 相对（ALiBi）：学习相对位置，更灵活

---

## ⚖️ 优缺点与场景

### ✅ 优势 (Pros)

1. **完全可并行化**：所有位置同时处理，不像 RNN 需逐步迭代（快 100 倍）
2. **长距离依赖直接捕捉**：任意两个 token 之间直接交互，不需逐步传递（解决梯度消失）
3. **多头机制**：多个"视角"同时学习，信息融合丰富
4. **位置信息清晰**：不依赖隐态传递，位置编码显式
5. **易于扩展**：支持从 BERT（双向）到 GPT（单向生成）的多种变体

### ❌ 劣势 (Cons)

1. **计算复杂度高**：注意力是 $O(n^2)$（$n$ = 序列长度），长序列内存爆炸
2. **绝对位置编码无泛化**：训练序列长 512，测试长 1024 会性能下降（除非用 ALiBi）
3. **对小数据集过拟合风险**：参数多，需大数据或强正则化
4. **缺乏归纳偏置**：CNN 的卷积有局部性偏好，RNN 有时间偏好，Transformer 无，需更多数据
5. **需要 warmup 和精细调参**：不如 CNN/RNN 稳定

### 🎯 适用场景

| 场景 | 适用度 | 原因 |
|---|---|---|
| **NLP 分类**（文本分类、情感分析） | ⭐⭐⭐⭐⭐ | BERT 等预训练模型强大 |
| **NLP 生成**（文本生成、翻译、摘要） | ⭐⭐⭐⭐⭐ | GPT 等模型标配，性能最优 |
| **大规模预训练 (LLM)** | ⭐⭐⭐⭐⭐ | ChatGPT、Claude 等基础 |
| **序列标注**（NER、POS） | ⭐⭐⭐⭐ | 优于 BiLSTM，预训练模型可用 |
| **长距离依赖**（长文档） | ⭐⭐⭐⭐⭐ | 直接全局交互，RNN 做不到 |
| **时间序列**（数值预测） | ⭐⭐⭐ | 可用但不如 LSTM；需绝对位置编码 |
| **短序列**（<64 tokens）| ⭐⭐ | 过度设计，CNN 或 RNN 足够；Transformer 开销大 |
| **长序列（>4K tokens）** | ⭐⭐ | 二次复杂度致命；需用 Linformer、Longformer 等高效变体 |

---

## 💬 面试必问

> [!question] Q1: 推导缩放点积注意力（Scaled Dot-Product Attention），为什么需要缩放？
>
> **答案框架**：
>
> **公式推导**：
>
> $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
>
> **缩放的必要性**：
>
> 不缩放时，$QK^T$ 中每个元素是 $d_k$ 个向量内积项的和：
> - 每项独立同分布，方差为 1
> - 总和的方差为 $d_k$（线性相加）
> - 标准差为 $\sqrt{d_k}$
>
> 当 $d_k$ 很大（如 512）时：
> - $QK^T$ 的值很大（范围可能 $[-1000, 1000]$）
> - softmax 的导数 $\approx 0$（梯度消失）
> - 注意力权重接近 one-hot（完全关注某一个 token）
>
> **缩放后**：
> - $QK^T / \sqrt{d_k}$ 的方差 $= d_k / d_k = 1$
> - softmax 输入在合理范围，梯度正常
> - 注意力权重分布均匀，更好地捕捉多个 token 的关系

> [!question] Q2: 多头注意力为什么有效？不同 head 学到什么？能否用单头代替？
>
> **答案核心**：
>
> **多头的价值**：
>
> 单个注意力头只能学习**一种关系**。多头分工合作：
> - Head 1：短距离依赖（如相邻词语法结构）
> - Head 2：长距离语义关系（如代词回指）
> - Head 3：句子结构关系（如从句、并列）
> - Head 4-8：其他语义-语法模式
>
> **无法用单头代替的原因**：
> 1. **表达能力**：单头输出维度 $d_k$（如 64），多头拼接后 $h \times d_k = 512$，参数利用率高
> 2. **正交表示学习**：多头可学习多个互补的表示空间
> 3. **正则化效果**：多头类似 ensemble，提升泛化
>
> **数学角度**：
>
> 单头注意力的输出空间秩最多为 $d_v = 64$，多头拼接后秩可达 $h \times d_v = 512$，表达更丰富。

> [!question] Q3: Transformer 相比 RNN/LSTM 的优势是什么？为什么长距离依赖上 Transformer 更优？
>
> **答案核心**：
>
> **梯度流角度**：
>
> **RNN/LSTM**：
> $$h_t = \text{LSTM}(x_t, h_{t-1})$$
> 从 $h_0$ 到 $h_n$ 的梯度需通过 $n$ 步，每步乘以 $|\partial h_t / \partial h_{t-1}| \approx \lambda$：
> - 若 $\lambda < 1$，梯度为 $\lambda^n$（消失）
> - 即使 LSTM 通过 gate 机制 ($\lambda \approx 1$)，仍需 $O(n)$ 步
> - 序列长 1000 时，梯度衰减：$0.99^{1000} \approx 10^{-5}$
>
> **Transformer**：
> $$\text{Attention}(Q, K, V) = \text{softmax}(QK^T) V$$
> 任意两位置间的路径长度为 1（直接交互），梯度不衰减（只要 softmax 输出非 0）。
>
> **计算复杂度角度**：
>
> | 模型 | 时间复杂度 | 空间复杂度 | 并行性 |
> |---|---|---|---|
> | RNN | $O(n \cdot d^2)$ | $O(d^2)$ | 无（逐步）|
> | Transformer | $O(n^2 \cdot d)$ | $O(n^2)$ | 完全 |
>
> 对于长序列，Transformer 的 $n^2$ 致命，但其直接全交互和完全并行化，使其在相同计算预算下精度更高。
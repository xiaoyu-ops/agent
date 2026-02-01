import torch
import torch.nn as nn
import math

# --- 占位符模块，将在后续小节中实现 ---

class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络模块
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x 形状: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # 最终输出形状: (batch_size, seq_len, d_model)
        return x


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制 (Multi-Head Attention) 模块
    该模块将 d_model 维度的输入拆分成 num_heads 个头，
    每个头独立地执行缩放点积注意力，然后将结果拼接并进行线性变换。
    这使得模型能够同时关注来自不同表示子空间的信息。
    """
    def __init__(self, d_model, num_heads):
        """
        初始化函数
        :param d_model: 输入的维度 (embedding dimension)
        :param num_heads: 注意力头的数量
        """
        super(MultiHeadAttention, self).__init__()
        # 确保 d_model 可以被 num_heads 整除
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model       # 模型的总维度
        self.num_heads = num_heads   # 注意力头的数量
        self.d_k = d_model // num_heads # 每个头的维度 (key/query的维度)

        # 定义 Query, Key, Value 和输出的线性变换层
        # 这些层将输入的 Q, K, V 向量映射到多头注意力的空间
        self.W_q = nn.Linear(d_model, d_model) # Query 的权重矩阵
        self.W_k = nn.Linear(d_model, d_model) # Key 的权重矩阵
        self.W_v = nn.Linear(d_model, d_model) # Value 的权重矩阵
        self.W_o = nn.Linear(d_model, d_model) # 输出的权重矩阵

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        计算缩放点积注意力 (Scaled Dot-Product Attention)
        公式为: Attention(Q, K, V) = softmax( (QK^T) / sqrt(d_k) ) * V
        :param Q: 查询 (Query), 形状: (batch_size, num_heads, seq_length, d_k)
        :param K: 键 (Key), 形状: (batch_size, num_heads, seq_length, d_k)
        :param V: 值 (Value), 形状: (batch_size, num_heads, seq_length, d_k)
        :param mask: 掩码, 用于屏蔽某些位置的注意力 (例如, padding 位置或未来的位置)
        :return: 注意力加权后的输出, 形状与 V 相同
        """
        # 1. 计算注意力得分 (QK^T)
        # K.transpose(-2, -1) 将最后两个维度转置, 实现序列长度维度上的矩阵乘法
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. 应用掩码 (如果提供)
        if mask is not None:
            # 将掩码中为 0 的位置 (即需要被屏蔽的位置) 的注意力得分设置为一个非常小的负数 (-1e9)
            # 这样在经过 softmax 后，这些位置的权重会趋近于 0
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 3. 计算注意力权重 (Softmax)
        # 在最后一个维度 (键的序列长度维度) 上应用 softmax，得到归一化的注意力权重
        attn_probs = torch.softmax(attn_scores, dim=-1)

        # 4. 加权求和 (权重 * V)
        # 将计算出的注意力权重与 V 相乘，得到加权后的输出
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        """
        将输入张量 x 拆分成多个头
        :param x: 输入张量, 形状: (batch_size, seq_length, d_model)
        :return: 拆分后的张量, 形状: (batch_size, num_heads, seq_length, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        # 1. view: 将 d_model 维度拆分为 num_heads * d_k
        #    形状变为: (batch_size, seq_length, num_heads, d_k)
        # 2. transpose(1, 2): 交换 num_heads 和 seq_length 维度
        #    最终形状变为: (batch_size, num_heads, seq_length, d_k)
        #    这是为了让每个头都能独立地处理整个序列
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        将多个头的输出合并回一个张量
        :param x: 多头输出张量, 形状: (batch_size, num_heads, seq_length, d_k)
        :return: 合并后的张量, 形状: (batch_size, seq_length, d_model)
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        # 1. transpose(1, 2): 交换回 num_heads 和 seq_length 维度
        #    形状变为: (batch_size, seq_length, num_heads, d_k)
        # 2. contiguous(): 确保张量在内存中是连续的，这是调用 view 前的必要操作
        # 3. view: 将 num_heads 和 d_k 维度合并为 d_model
        #    最终形状变为: (batch_size, seq_length, d_model)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        前向传播函数
        :param Q: 查询 (Query), 形状: (batch_size, seq_length, d_model)
        :param K: 键 (Key), 形状: (batch_size, seq_length, d_model)
        :param V: 值 (Value), 形状: (batch_size, seq_length, d_model)
        :param mask: 掩码
        :return: 多头注意力模块的最终输出, 形状: (batch_size, seq_length, d_model)
        """
        # 1. 对 Q, K, V 进行线性变换，然后拆分成多个头
        #    例如: self.W_q(Q) -> (batch_size, seq_length, d_model)
        #    self.split_heads(...) -> (batch_size, num_heads, seq_length, d_k)
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        # 2. 计算缩放点积注意力
        #    attn_output 形状: (batch_size, num_heads, seq_length, d_k)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. 合并多头输出并进行最终的线性变换
        #    self.combine_heads(attn_output) -> (batch_size, seq_length, d_model)
        #    self.W_o(...) -> (batch_size, seq_length, d_model)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionalEncoding(nn.Module):
    """
    为输入序列的词嵌入向量添加位置编码。
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建一个足够长的位置编码矩阵
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # pe (positional encoding) 的大小为 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)

        # 偶数维度使用 sin, 奇数维度使用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 将 pe 注册为 buffer，这样它就不会被视为模型参数，但会随模型移动（例如 to(device)）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.size(1) 是当前输入的序列长度
        # 将位置编码加到输入向量上
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# --- 编码器核心层 ---

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 残差连接与层归一化将在 3.1.2.4 节中详细解释
        # 1. 多头自注意力
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x

# --- 解码器核心层 ---

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention() # 待实现
        self.cross_attn = MultiHeadAttention() # 待实现
        self.feed_forward = PositionWiseFeedForward() # 待实现
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # 1. 掩码多头自注意力 (对自己)
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 交叉注意力 (对编码器输出)
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 3. 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x

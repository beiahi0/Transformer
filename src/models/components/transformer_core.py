import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """实现正弦位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)  # (Batch, Seq_Len, Dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # 注册为 buffer，不参与梯度更新

    def forward(self, x):
        """x: (N, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """实现缩放点积注意力"""

    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        output = attn @ v
        return output, attn


class MultiHeadAttention(nn.Module):
    """实现多头注意力"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout=dropout)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        N = q.size(0)
        q = (
            self.q_linear(q).view(N, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # (N, 8, seq_len, 64)
        k = self.k_linear(k).view(N, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(N, -1, self.n_heads, self.d_k).transpose(1, 2)

        x, _ = self.attention(q, k, v, mask=mask)  # (N, 8, seq_len,64  )
        x = (
            x.transpose(1, 2).contiguous().view(N, -1, self.n_heads * self.d_k)
        )  # (N, seq_len, 512)

        # 让模型融合8个头（8中不同注意力提供的信息）
        x = self.out_linear(x)

        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    """实现 FFN"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))


class EncoderLayer(nn.Module):
    """实现 Encoder Layer"""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x + self.dropout1(self.self_attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class DecoderLayer(nn.Module):
    """实现 Decoder Layer"""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.encoder_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.norm1(
            x + self.dropout1(self.masked_self_attn(x, x, x, tgt_mask))
        )
        x = self.norm2(
            x
            + self.dropout2(
                self.encoder_attn(x, encoder_output, encoder_output, src_mask)
            )
        )
        x = self.norm3(x + self.dropout3(self.ffn(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x, mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """完整的 Encoder-Decoder Transformer"""

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )
        self.decoder = Decoder(
            tgt_vocab_size, d_model, n_layers, n_heads, d_ff, dropout, max_len
        )
        self.final_linear = nn.Linear(d_model, tgt_vocab_size)
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.final_linear(decoder_output)
        return output

    def create_src_mask(self, src, pad_idx):
        # (N, 1, 1, src_seq_len)
        src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt, pad_idx):
        # (N, 1, tgt_seq_len, 1)
        tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        # (1, 1, seq_len, seq_len)
        tgt_future_mask = torch.triu(
            torch.ones(1, 1, seq_len, seq_len, device=tgt.device), diagonal=1
        ).bool()
        # (N, 1, tgt_seq_len, tgt_seq_len)
        tgt_mask = tgt_pad_mask & ~tgt_future_mask
        return tgt_mask

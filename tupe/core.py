import math

import torch
from torch import nn
from torch.nn import functional as F


def get_relative_positions(seq_len):
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        pe = torch.empty(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.tensor:
        # shape (1, seq_len, d_model)
        return self.pe[:, :seq_len]


class TUPEMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.1,
        relative_bias: bool = True,
    ) -> None:
        d_head, remainder = divmod(d_model, num_heads)
        assert remainder == 0, "`d_model` should be divisible by `num_heads`"
        super().__init__()
        self.max_len = max_len
        self.num_heads = num_heads
        self.scale = 1 / math.sqrt(2 * d_head)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos_embed = AbsolutePositionalEmbedding(d_model, max_len)

        # kqv in one pass
        self.pos_kq = nn.Linear(d_model, 2 * d_model, bias=False)
        self.tok_kqv = nn.Linear(d_model, 3 * d_model, bias=False)

        self.relative_bias = relative_bias
        if relative_bias:
            self.bias = nn.Embedding(max_len * 2, num_heads)

    def forward(self, x: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = x.shape

        pos_embed = self.pos_embed(seq_len).repeat(batch_size, 1, 1)
        # pos_embed.shape == (batch_size, seq_len, d_model)
        pos_key, pos_query = self.pos_kq(pos_embed).chunk(2, dim=-1)
        pos_key = pos_key.view(batch_size, seq_len, self.num_heads, -1).permute(
            0, 2, 3, 1
        )
        # pos_key.shape == (batch_size, num_heads, d_head, seq_len)
        pos_query = pos_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        # pos_query.shape == (batch_size, num_heads, seq_len, d_head)
        pos_attn = torch.matmul(pos_query, pos_key)
        # pos_attn.shape == (batch_size, num_heads, seq_len, seq_len)

        tok_key, tok_query, tok_value = self.tok_kqv(self.norm(x)).chunk(3, dim=-1)
        tok_key = tok_key.view(batch_size, seq_len, self.num_heads, -1).permute(
            0, 2, 3, 1
        )
        # tok_key.shape == (batch_size, num_heads, d_head, seq_len)
        tok_query = tok_query.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        tok_value = tok_value.view(batch_size, seq_len, self.num_heads, -1).transpose(
            1, 2
        )
        # tok_qv.shape == (batch_size, num_heads, seq_len, d_head)
        tok_attn = torch.matmul(tok_query, tok_key)
        # tok_attn.shape == (batch_size, num_heads, seq_len, seq_len)

        attn = (tok_attn + pos_attn) / self.scale
        if self.relative_bias:
            relative_positions = get_relative_positions(seq_len).to(attn.device)
            # relative_positions.shape == (seq_len, seq_len)
            bias = self.bias(relative_positions + self.max_len)
            # bias.shape == (seq_len, seq_len, num_heads)
            bias = bias.permute(2, 0, 1).unsqueeze(0)
            # bias.shape == (1, num_heads, seq_len, seq_len)
            attn = attn + bias

        attn = F.softmax(attn, dim=-1)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        out = torch.matmul(attn, tok_value)
        # out.shape == (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)

        return self.dropout(out)

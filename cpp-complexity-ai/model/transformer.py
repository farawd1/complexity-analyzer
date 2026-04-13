"""
ComplexityTransformer — built from scratch in PyTorch.
Do NOT replace any component with nn.MultiheadAttention or nn.Transformer wrappers.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product multi-head attention, fully from scratch."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        B, T, D = x.shape
        def reshape(t):
            return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        Q, K, V = reshape(self.q_proj(x)), reshape(self.k_proj(x)), reshape(self.v_proj(x))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)


class TransformerEncoderLayer(nn.Module):
    """Pre-norm: LayerNorm → sub-layer → residual."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


class ComplexityTransformer(nn.Module):
    """
    Input:  token_ids (B, T), ast_features (B, 13)
    Output: logits    (B, 14)  for 14 complexity classes
    """
    N_CLASSES = 14

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, d_ff: int = 512,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.feature_proj = nn.Linear(13, d_model)  # 13 hand-crafted features now
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, self.N_CLASSES),
        )

    def forward(self, token_ids, ast_features, padding_mask=None):
        x = self.embedding(token_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        x = self.norm(x)
        if padding_mask is not None:
            mask = (~padding_mask).float().unsqueeze(-1)
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        feat = self.feature_proj(ast_features)
        return self.classifier(torch.cat([x, feat], dim=-1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    context_length: int
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    rope_theta: float = 10000.0
    use_rmsnorm: bool = True
    pre_norm: bool = True
    use_rope: bool = True
    ffn_type: str = "swiglu"
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        batch, heads, seq_len, dim = x.shape
        if positions is None:
            positions = torch.arange(seq_len, device=x.device)
        if positions.dim() == 1:
            cos = self.cos_cached[positions].to(device=x.device, dtype=x.dtype)
            sin = self.sin_cached[positions].to(device=x.device, dtype=x.dtype)
            cos = cos.view(1, 1, seq_len, -1)
            sin = sin.view(1, 1, seq_len, -1)
        elif positions.dim() == 2:
            if positions.shape != (batch, seq_len):
                raise ValueError("positions must have shape (batch, seq_len)")
            cos = self.cos_cached[positions].to(device=x.device, dtype=x.dtype)
            sin = self.sin_cached[positions].to(device=x.device, dtype=x.dtype)
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
        else:
            raise ValueError("positions must be 1D or 2D")

        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        *,
        use_rope: bool,
        rope_theta: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, context_length, theta=rope_theta) if use_rope else None
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = self.rope(q)
            k = self.rope(k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.causal_mask[:seq_len, :seq_len]
        attn_scores = attn_scores.masked_fill(~mask, torch.finfo(attn_scores.dtype).min)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.resid_drop(self.out_proj(attn_out))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, ffn_type: str, dropout: float) -> None:
        super().__init__()
        self.ffn_type = ffn_type
        if ffn_type == "swiglu":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_ff, bias=False)
        elif ffn_type == "silu":
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)
            self.w3 = None
        else:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "swiglu":
            hidden = F.silu(self.w1(x)) * self.w3(x)
            return self.drop(self.w2(hidden))
        hidden = F.silu(self.w1(x))
        return self.drop(self.w2(hidden))


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.pre_norm = config.pre_norm
        self.use_rmsnorm = config.use_rmsnorm
        self.ln1 = RMSNorm(config.d_model) if config.use_rmsnorm else None
        self.ln2 = RMSNorm(config.d_model) if config.use_rmsnorm else None
        self.attn = CausalSelfAttention(
            config.d_model,
            config.num_heads,
            config.context_length,
            use_rope=config.use_rope,
            rope_theta=config.rope_theta,
            dropout=config.dropout,
        )
        self.ffn = FeedForward(config.d_model, config.d_ff, config.ffn_type, config.dropout)

    def _norm(self, norm: RMSNorm | None, x: torch.Tensor) -> torch.Tensor:
        return norm(x) if norm is not None else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self._norm(self.ln1, x))
            x = x + self.ffn(self._norm(self.ln2, x))
            return x
        x = self._norm(self.ln1, x + self.attn(x))
        x = self._norm(self.ln2, x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.d_model) if config.use_rmsnorm else None
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(token_ids)
        for block in self.blocks:
            x = block(x)
        if self.ln_f is not None:
            x = self.ln_f(x)
        return self.lm_head(x)

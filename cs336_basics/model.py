from __future__ import annotations

import math

import torch
from torch import Tensor

from .nn_utils import softmax


def linear(
    d_in: int,
    d_out: int,
    weights: Tensor,
    in_features: Tensor,
) -> Tensor:
    return torch.matmul(in_features, weights.t())


def embedding(
    vocab_size: int,
    d_model: int,
    weights: Tensor,
    token_ids: Tensor,
) -> Tensor:
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)


def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Tensor,
    w2_weight: Tensor,
    w3_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    hidden = silu(torch.matmul(in_features, w1_weight.t())) * torch.matmul(in_features, w3_weight.t())
    return torch.matmul(hidden, w2_weight.t())


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    attn = softmax(scores, dim=-1)
    return torch.matmul(attn, V)


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    *leading, seq_len, d_model = x.shape
    d_head = d_model // num_heads
    x = x.reshape(*leading, seq_len, num_heads, d_head)
    return x.transpose(-3, -2)


def _merge_heads(x: Tensor) -> Tensor:
    x = x.transpose(-3, -2)
    *leading, seq_len, num_heads, d_head = x.shape
    return x.reshape(*leading, seq_len, num_heads * d_head)


def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Tensor,
    token_positions: Tensor,
) -> Tensor:
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")

    device = in_query_or_key.device
    dtype = in_query_or_key.dtype
    positions = token_positions.to(device=device)
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device, dtype=dtype) / d_k))
    angles = positions.to(dtype)[..., None] * inv_freq
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    x = in_query_or_key.reshape(*in_query_or_key.shape[:-1], d_k // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1)
    return out.reshape_as(in_query_or_key)


def _multihead_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    *,
    use_rope: bool,
    max_seq_len: int | None = None,
    theta: float | None = None,
    token_positions: Tensor | None = None,
    mask: Tensor | None = None,
) -> Tensor:
    q = torch.matmul(in_features, q_proj_weight.t())
    k = torch.matmul(in_features, k_proj_weight.t())
    v = torch.matmul(in_features, v_proj_weight.t())

    q = _split_heads(q, num_heads)
    k = _split_heads(k, num_heads)
    v = _split_heads(v, num_heads)

    if use_rope:
        if theta is None or max_seq_len is None:
            raise ValueError("theta and max_seq_len must be provided when use_rope is True")
        if token_positions is None:
            seq_len = in_features.shape[-2]
            token_positions = torch.arange(seq_len, device=in_features.device)
        q = rope(d_model // num_heads, theta, max_seq_len, q, token_positions)
        k = rope(d_model // num_heads, theta, max_seq_len, k, token_positions)

    attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
    merged = _merge_heads(attn_out)
    return torch.matmul(merged, o_proj_weight.t())


def multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
) -> Tensor:
    return _multihead_attention(
        d_model,
        num_heads,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        in_features,
        use_rope=False,
    )


def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    in_features: Tensor,
    token_positions: Tensor | None = None,
) -> Tensor:
    return _multihead_attention(
        d_model,
        num_heads,
        q_proj_weight,
        k_proj_weight,
        v_proj_weight,
        o_proj_weight,
        in_features,
        use_rope=True,
        max_seq_len=max_seq_len,
        theta=theta,
        token_positions=token_positions,
    )


def rmsnorm(d_model: int, eps: float, weights: Tensor, in_features: Tensor) -> Tensor:
    variance = in_features.pow(2).mean(dim=-1, keepdim=True)
    normalized = in_features / torch.sqrt(variance + eps)
    return normalized * weights


def _causal_mask(seq_len: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))


def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Tensor,
) -> Tensor:
    x = in_features
    x_norm = rmsnorm(d_model, eps=1e-5, weights=weights["ln1.weight"], in_features=x)
    seq_len = x.shape[-2]
    mask = _causal_mask(seq_len, device=x.device)
    attn_out = _multihead_attention(
        d_model,
        num_heads,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        x_norm,
        use_rope=True,
        max_seq_len=max_seq_len,
        theta=theta,
        token_positions=None,
        mask=mask,
    )
    x = x + attn_out
    x_norm = rmsnorm(d_model, eps=1e-5, weights=weights["ln2.weight"], in_features=x)
    ffn_out = swiglu(
        d_model,
        d_ff,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
        x_norm,
    )
    return x + ffn_out


def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Tensor,
) -> Tensor:
    x = embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)
    seq_len = x.shape[-2]
    mask = _causal_mask(seq_len, device=x.device)
    for layer in range(num_layers):
        prefix = f"layers.{layer}."
        layer_weights = {
            "attn.q_proj.weight": weights[prefix + "attn.q_proj.weight"],
            "attn.k_proj.weight": weights[prefix + "attn.k_proj.weight"],
            "attn.v_proj.weight": weights[prefix + "attn.v_proj.weight"],
            "attn.output_proj.weight": weights[prefix + "attn.output_proj.weight"],
            "ln1.weight": weights[prefix + "ln1.weight"],
            "ffn.w1.weight": weights[prefix + "ffn.w1.weight"],
            "ffn.w2.weight": weights[prefix + "ffn.w2.weight"],
            "ffn.w3.weight": weights[prefix + "ffn.w3.weight"],
            "ln2.weight": weights[prefix + "ln2.weight"],
        }
        x_norm = rmsnorm(d_model, eps=1e-5, weights=layer_weights["ln1.weight"], in_features=x)
        attn_out = _multihead_attention(
            d_model,
            num_heads,
            layer_weights["attn.q_proj.weight"],
            layer_weights["attn.k_proj.weight"],
            layer_weights["attn.v_proj.weight"],
            layer_weights["attn.output_proj.weight"],
            x_norm,
            use_rope=True,
            max_seq_len=context_length,
            theta=rope_theta,
            token_positions=None,
            mask=mask,
        )
        x = x + attn_out
        x_norm = rmsnorm(d_model, eps=1e-5, weights=layer_weights["ln2.weight"], in_features=x)
        ffn_out = swiglu(
            d_model,
            d_ff,
            layer_weights["ffn.w1.weight"],
            layer_weights["ffn.w2.weight"],
            layer_weights["ffn.w3.weight"],
            x_norm,
        )
        x = x + ffn_out

    x = rmsnorm(d_model, eps=1e-5, weights=weights["ln_final.weight"], in_features=x)
    return torch.matmul(x, weights["lm_head.weight"].t())

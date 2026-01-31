from __future__ import annotations

import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    max_vals = in_features.max(dim=dim, keepdim=True).values
    exp_vals = torch.exp(in_features - max_vals)
    return exp_vals / exp_vals.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    logsumexp = torch.logsumexp(inputs, dim=-1)
    target_logits = inputs.gather(1, targets.unsqueeze(-1)).squeeze(-1)
    return (logsumexp - target_logits).mean()


def gradient_clipping(parameters, max_l2_norm: float) -> None:
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)

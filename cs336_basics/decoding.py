from __future__ import annotations

import torch


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int | torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative > top_p
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        next_token = torch.multinomial(sorted_probs, 1)
        next_token = sorted_idx.gather(-1, next_token)
    else:
        next_token = torch.multinomial(probs, 1)

    if logits.dim() == 1:
        return int(next_token.item())
    return next_token.squeeze(-1)


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    model.eval()
    device = next(model.parameters()).device
    tokens = input_ids.to(device)

    for _ in range(max_new_tokens):
        idx_cond = tokens[:, -context_length:]
        logits = model(idx_cond)
        next_logits = logits[:, -1, :]
        next_ids = sample_next_token(next_logits, temperature=temperature, top_p=top_p)
        if isinstance(next_ids, int):
            next_ids = torch.tensor([next_ids], device=device, dtype=tokens.dtype)
        next_ids = next_ids.to(device=device, dtype=tokens.dtype)
        tokens = torch.cat([tokens, next_ids[:, None]], dim=1)
        if eos_token_id is not None and (next_ids == eos_token_id).all():
            break

    return tokens

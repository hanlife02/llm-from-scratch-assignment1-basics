from __future__ import annotations

import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data = dataset
    max_start = data.shape[0] - context_length
    if max_start <= 0:
        raise ValueError("context_length must be smaller than dataset length")

    start_indices = torch.randint(0, max_start, (batch_size,), device="cpu").numpy()
    offsets = torch.arange(context_length, device="cpu").numpy()
    idx = start_indices[:, None] + offsets[None, :]
    x = torch.from_numpy(data[idx].copy()).to(device=device, dtype=torch.long)
    y = torch.from_numpy(data[idx + 1].copy()).to(device=device, dtype=torch.long)
    return x, y

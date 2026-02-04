from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import get_worker_info


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


class RandomBatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        batch_size: int,
        context_length: int,
        length: int,
    ) -> None:
        self.path = Path(path)
        self.batch_size = batch_size
        self.context_length = context_length
        self.length = length
        self._data: npt.NDArray | None = None
        self._max_start: int | None = None
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self.length

    def _get_data(self) -> npt.NDArray:
        if self._data is None:
            if not self.path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.path}")
            self._data = np.load(self.path, mmap_mode="r")
            max_start = self._data.shape[0] - self.context_length
            if max_start <= 0:
                raise ValueError("context_length must be smaller than dataset length")
            self._max_start = max_start
        return self._data

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            worker_info = get_worker_info()
            seed = worker_info.seed if worker_info is not None else torch.initial_seed()
            self._rng = np.random.default_rng(seed % (2**32))
        return self._rng

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._get_data()
        rng = self._get_rng()
        max_start = self._max_start if self._max_start is not None else data.shape[0] - self.context_length
        start_indices = rng.integers(0, max_start, size=self.batch_size, dtype=np.int64)
        offsets = np.arange(self.context_length, dtype=np.int64)
        idx = start_indices[:, None] + offsets[None, :]
        x = torch.from_numpy(data[idx].copy()).to(dtype=torch.long)
        y = torch.from_numpy(data[idx + 1].copy()).to(dtype=torch.long)
        return x, y

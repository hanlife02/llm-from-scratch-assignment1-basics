from __future__ import annotations

import math
from collections.abc import Callable
from typing import Optional

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss


def run_once(lr: float, steps: int = 10) -> list[float]:
    torch.manual_seed(0)
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr=lr)
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses


def main() -> None:
    for lr in (1e1, 1e2, 1e3):
        losses = run_once(lr)
        formatted = ", ".join(f"{x:.6f}" for x in losses)
        print(f"lr={lr} losses: {formatted}")


if __name__ == "__main__":
    main()

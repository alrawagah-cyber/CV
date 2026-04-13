"""Cosine-annealing LR scheduler with linear warmup."""

from __future__ import annotations

import math

import torch


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for `warmup_steps` then cosine decay to `min_lr`."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        last_epoch: int = -1,
    ):
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be < total_steps")
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = (step + 1) / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cos_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            scale = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos_scale
        return [base_lr * scale for base_lr in self.base_lrs]

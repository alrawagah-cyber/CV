"""Batching helpers for the classifier layers during inference."""

from __future__ import annotations

from typing import Iterator

import torch


def chunked(tensor: torch.Tensor, batch_size: int) -> Iterator[torch.Tensor]:
    """Yield consecutive batch_size-sized chunks of `tensor` along dim 0."""
    n = tensor.size(0)
    for start in range(0, n, batch_size):
        yield tensor[start : start + batch_size]

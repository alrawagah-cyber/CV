"""Lightweight full-image pre-screen for total-loss vehicles.

A small ResNet-18 binary classifier that runs on the full image *before* the
3-layer pipeline.  When confidence exceeds the configured threshold the caller
can short-circuit and return a total_loss report without per-part analysis.

Disabled by default (no checkpoint = always returns ``(False, 0.0)``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)

_SCREEN_TRANSFORM = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TotalLossScreener:
    """Binary classifier: *total_loss* vs *not* on the full image."""

    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "cpu",
    ):
        self.enabled = weights is not None
        self.device = torch.device(device)

        if not self.enabled:
            self.model: nn.Module | None = None
            return

        logger.info("Loading total-loss screener weights from %s", weights)
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        ckpt = torch.load(str(weights), map_location=self.device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        self.model = model

    def screen(self, image: np.ndarray) -> tuple[bool, float]:
        """Return ``(is_total_loss, confidence)`` for a single RGB uint8 image."""
        if not self.enabled or self.model is None:
            return False, 0.0

        tensor = _SCREEN_TRANSFORM(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(tensor).squeeze(-1)
            prob = torch.sigmoid(logit).item()
        return prob >= 0.5, prob

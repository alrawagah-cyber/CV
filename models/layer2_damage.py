"""Layer 2: Damage-type multi-label classifier.

Backbone: ConvNeXt-V2-Large (via timm), swappable to EfficientNet-V2-L.
Head: MultiLabelHead (sigmoid logits) over the damage vocabulary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.heads import MultiLabelHead

DEFAULT_DAMAGE_CLASSES: list[str] = [
    "dent",
    "scratch",
    "crack",
    "shatter",
    "tear",
    "deformation",
    "paint_loss",
    "puncture",
    "misalignment",
]


class DamageTypeClassifier(nn.Module):
    """timm backbone -> pooled features -> multi-label sigmoid head.

    Set `num_classes=0` on timm so it returns pooled features; attach our own
    head for clean checkpointing and easy swap-in of class vocabularies.
    """

    def __init__(
        self,
        backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k",
        classes: list[str] | None = None,
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        import timm

        self.classes = classes or DEFAULT_DAMAGE_CLASSES
        self.num_classes = len(self.classes)
        self.backbone_name = backbone

        # num_classes=0 -> global-pooled features
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        in_feats = self.backbone.num_features
        self.head = MultiLabelHead(in_feats, self.num_classes, dropout=dropout)

        # Expose a recommended input size + normalization from the backbone's cfg.
        data_cfg = getattr(self.backbone, "default_cfg", {}) or {}
        self.input_size: int = int(data_cfg.get("input_size", (3, 224, 224))[-1])
        self.mean: tuple[float, ...] = tuple(data_cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std: tuple[float, ...] = tuple(data_cfg.get("std", (0.229, 0.224, 0.225)))

    # --------------------------- forward ---------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.head(feats)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities, shape (B, num_classes)."""
        was_training = self.training
        self.eval()
        logits = self(x)
        probs = torch.sigmoid(logits)
        if was_training:
            self.train()
        return probs

    # --------------------------- i/o -------------------------------
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "backbone": self.backbone_name,
                "classes": self.classes,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> DamageTypeClassifier:
        ckpt: dict[str, Any] = torch.load(path, map_location=map_location)
        model = cls(
            backbone=ckpt.get("backbone", "convnextv2_large.fcmae_ft_in22k_in1k"),
            classes=ckpt.get("classes"),
            pretrained=False,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model

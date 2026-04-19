"""Layer 3: Severity assessment with ordinal regression.

Backbone: Swin Transformer V2 Large (timm), swappable to DINOv2 ViT-L.
Heads:
    - CoralOrdinalHead over severity grades {minor, moderate, severe, total_loss}
    - RepairReplaceHead (binary): recommend repair vs replace
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from models.class_constants import DEFAULT_SEVERITY_GRADES
from models.heads import CoralOrdinalHead, RepairReplaceHead

__all__ = ["DEFAULT_SEVERITY_GRADES", "SeverityOutput", "SeverityAssessor"]


@dataclass
class SeverityOutput:
    grade: str
    grade_index: int
    grade_confidence: float
    severity_probs: dict[str, float]
    repair_probability: float
    replace_probability: float
    recommendation: str  # "repair" | "replace"


class SeverityAssessor(nn.Module):
    """Shared backbone + CORAL ordinal head + repair/replace head."""

    def __init__(
        self,
        backbone: str = "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        grades: list[str] | None = None,
        pretrained: bool = True,
        dropout: float = 0.2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        import timm

        self.grades = grades or DEFAULT_SEVERITY_GRADES
        self.num_classes = len(self.grades)
        self.backbone_name = backbone

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        in_feats = self.backbone.num_features
        self.ordinal_head = CoralOrdinalHead(in_feats, self.num_classes, dropout=dropout)
        self.repair_head = RepairReplaceHead(in_feats, dropout=dropout)

        data_cfg = getattr(self.backbone, "default_cfg", {}) or {}
        # Swin V2 large often expects 384; fall back to 224 if unavailable.
        self.input_size: int = int(data_cfg.get("input_size", (3, 384, 384))[-1])
        self.mean: tuple[float, ...] = tuple(data_cfg.get("mean", (0.485, 0.456, 0.406)))
        self.std: tuple[float, ...] = tuple(data_cfg.get("std", (0.229, 0.224, 0.225)))

    # --------------------------- forward ---------------------------
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.backbone(x)
        ordinal_logits = self.ordinal_head(feats)  # (B, K-1)
        repair_logit = self.repair_head(feats)  # (B,)
        return {"ordinal_logits": ordinal_logits, "repair_logit": repair_logit}

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> list[SeverityOutput]:
        was_training = self.training
        self.eval()
        out = self(x)
        ordinal_probs = torch.sigmoid(out["ordinal_logits"])  # (B, K-1)
        ranks = CoralOrdinalHead.probs_to_rank(ordinal_probs)  # (B,)

        # Convert cumulative probs to per-class probabilities: p(k) = p(>k-1) - p(>k)
        # p(>k) for k = 0..K-2 is ordinal_probs[:, k]. p(>-1) = 1, p(>K-1) = 0.
        B = ordinal_probs.shape[0]
        cumulative = torch.cat(
            [
                torch.ones(B, 1, device=ordinal_probs.device),
                ordinal_probs,
                torch.zeros(B, 1, device=ordinal_probs.device),
            ],
            dim=1,
        )  # (B, K+1) with cumulative[:, k] = p(> k-1)
        per_class = (cumulative[:, :-1] - cumulative[:, 1:]).clamp(min=0.0)  # (B, K)
        # Normalize in case of numerical drift
        per_class = per_class / per_class.sum(dim=1, keepdim=True).clamp(min=1e-8)

        repair_prob = torch.sigmoid(out["repair_logit"])  # (B,)

        outputs: list[SeverityOutput] = []
        for i in range(B):
            idx = int(ranks[i].item())
            idx = max(0, min(self.num_classes - 1, idx))
            probs_dict = {g: float(per_class[i, j].item()) for j, g in enumerate(self.grades)}
            replace_p = float(repair_prob[i].item())  # head logit = P(replace)
            outputs.append(
                SeverityOutput(
                    grade=self.grades[idx],
                    grade_index=idx,
                    grade_confidence=probs_dict[self.grades[idx]],
                    severity_probs=probs_dict,
                    repair_probability=1.0 - replace_p,
                    replace_probability=replace_p,
                    recommendation="replace" if replace_p >= 0.5 else "repair",
                )
            )

        if was_training:
            self.train()
        return outputs

    # --------------------------- i/o -------------------------------
    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "backbone": self.backbone_name,
                "grades": self.grades,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> SeverityAssessor:
        ckpt: dict[str, Any] = torch.load(path, map_location=map_location)
        model = cls(
            backbone=ckpt.get("backbone", "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k"),
            grades=ckpt.get("grades"),
            pretrained=False,
        )
        model.load_state_dict(ckpt["state_dict"])
        return model

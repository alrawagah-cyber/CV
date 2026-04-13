"""Classifier heads used by layers 2 and 3.

- MultiLabelHead: simple sigmoid head for independent binary labels (damage types).
- CoralOrdinalHead: CORAL-style ordinal regression with K-1 cumulative logits
  plus a shared weight vector and per-rank biases. See Cao et al. 2020.
- RepairReplaceHead: auxiliary binary classifier attached to the severity
  backbone to recommend repair vs. replace.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelHead(nn.Module):
    """Dropout + linear projection -> logits for multi-label classification."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class CoralOrdinalHead(nn.Module):
    """CORAL ordinal regression head.

    Produces K-1 cumulative logits from a single shared weight vector plus
    K-1 independent biases. Guarantees monotonically decreasing predicted
    probabilities and consistent rank predictions.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORAL head requires num_classes >= 2")
        self.num_classes = num_classes
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Linear(in_features, 1, bias=False)
        # K-1 biases, one per rank threshold
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1) shared score, broadcast across K-1 thresholds
        shared = self.weight(self.dropout(x))  # (B, 1)
        return shared + self.bias  # (B, K-1) logits

    @staticmethod
    def probs_to_rank(probs: torch.Tensor) -> torch.Tensor:
        """Convert (B, K-1) sigmoid probs to integer rank in [0, K-1]."""
        return (probs > 0.5).long().sum(dim=1)

    @staticmethod
    def rank_to_levels(y: torch.Tensor, num_classes: int) -> torch.Tensor:
        """Encode integer label y in [0, K-1] as a (B, K-1) ordinal target.

        level[k] = 1 if y > k else 0
        """
        device = y.device
        thresholds = torch.arange(num_classes - 1, device=device).unsqueeze(0)  # (1, K-1)
        return (y.unsqueeze(1) > thresholds).float()


def coral_loss(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Binary cross entropy over ordinal thresholds."""
    levels = CoralOrdinalHead.rank_to_levels(targets, num_classes)
    return F.binary_cross_entropy_with_logits(logits, levels)


class RepairReplaceHead(nn.Module):
    """Single-logit head: probability that the part should be replaced (vs repaired)."""

    def __init__(self, in_features: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x)).squeeze(-1)

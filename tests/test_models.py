"""Tests for model heads and CORAL logic. No network/weight downloads."""

from __future__ import annotations

import torch

from models.heads import CoralOrdinalHead, MultiLabelHead, coral_loss
from models.registry import MODEL_REGISTRY, build_model


def test_multilabel_head_shape():
    head = MultiLabelHead(in_features=32, num_classes=5)
    x = torch.randn(4, 32)
    out = head(x)
    assert out.shape == (4, 5)


def test_coral_head_shape_and_rank():
    head = CoralOrdinalHead(in_features=16, num_classes=4)
    x = torch.randn(8, 16)
    logits = head(x)
    assert logits.shape == (8, 3)  # K-1 thresholds

    probs = torch.sigmoid(logits)
    ranks = CoralOrdinalHead.probs_to_rank(probs)
    assert ranks.shape == (8,)
    assert ranks.min() >= 0 and ranks.max() <= 3


def test_coral_loss_decreases_with_matched_targets():
    head = CoralOrdinalHead(in_features=8, num_classes=4)
    optimizer = torch.optim.SGD(head.parameters(), lr=0.5)
    x = torch.randn(32, 8)
    y = torch.randint(0, 4, (32,))

    def step_loss():
        return coral_loss(head(x), y, num_classes=4)

    initial = float(step_loss().item())
    for _ in range(60):
        optimizer.zero_grad()
        loss = step_loss()
        loss.backward()
        optimizer.step()
    final = float(step_loss().item())
    assert final < initial, f"CORAL loss did not decrease: {initial} -> {final}"


def test_rank_to_levels_matches_definition():
    y = torch.tensor([0, 1, 2, 3])
    levels = CoralOrdinalHead.rank_to_levels(y, num_classes=4)
    expected = torch.tensor(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0]]
    )
    assert torch.allclose(levels, expected)


def test_registry_contains_all_layers():
    assert {"part_detector", "damage_type", "severity"}.issubset(MODEL_REGISTRY.keys())


def test_build_model_unknown_raises():
    try:
        build_model("nope")
    except KeyError:
        return
    raise AssertionError("expected KeyError")

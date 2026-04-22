"""Active learning uncertainty scoring for damage assessments.

When the model is uncertain about a prediction (high entropy in the
probability distributions, or low detection confidence), the part is
flagged for human review.
"""

from __future__ import annotations

import math


def _entropy(probs: dict[str, float]) -> float:
    """Shannon entropy of a discrete probability distribution.

    Returns 0.0 for degenerate (empty / single-class) distributions.
    """
    total = sum(probs.values())
    if total <= 0 or len(probs) < 2:
        return 0.0
    h = 0.0
    for p in probs.values():
        if p > 0:
            normed = p / total
            h -= normed * math.log(normed)
    return h


def compute_l2_uncertainty(damage_probs: dict[str, float]) -> float:
    """Entropy of the damage-type probability distribution.

    Higher values indicate greater uncertainty about what kind of damage
    is present.
    """
    return _entropy(damage_probs)


def compute_l3_uncertainty(severity_probs: dict[str, float]) -> float:
    """Entropy of the severity-grade probability distribution.

    Higher values indicate greater uncertainty about how severe the
    damage is.
    """
    return _entropy(severity_probs)


def should_flag_for_review(
    l2_uncertainty: float,
    l3_uncertainty: float,
    detection_confidence: float,
    thresholds: dict,
) -> bool:
    """Decide whether a part assessment should be routed to human review.

    A part is flagged when *any* of the following hold:

    * L2 entropy exceeds ``thresholds["l2_entropy_threshold"]``
    * L3 entropy exceeds ``thresholds["l3_entropy_threshold"]``
    * Detection confidence is below ``thresholds["min_detection_confidence"]``
    """
    if detection_confidence < thresholds.get("min_detection_confidence", 0.4):
        return True
    if l2_uncertainty > thresholds.get("l2_entropy_threshold", 1.5):
        return True
    if l3_uncertainty > thresholds.get("l3_entropy_threshold", 1.0):
        return True
    return False

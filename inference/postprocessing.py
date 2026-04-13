"""Post-processing: rule-based repair/replace fallback + JSON report builder."""

from __future__ import annotations

from typing import Any


# Static repair-vs-replace heuristic table.
# Keyed by (part, damage_type, severity_index). Value is "repair" | "replace".
# Used only if no trained repair/replace head output is available OR if the
# config explicitly opts into rules. Non-matching combinations fall through to
# a sane severity-based default (severe/total_loss -> replace, else repair).
_RULES: dict[tuple[str, str, int], str] = {
    ("windshield", "crack", 0): "repair",
    ("windshield", "crack", 1): "replace",
    ("windshield", "crack", 2): "replace",
    ("windshield", "shatter", 0): "replace",
    ("windshield", "shatter", 1): "replace",
    ("windshield", "shatter", 2): "replace",
    ("headlight", "shatter", 0): "replace",
    ("headlight", "shatter", 1): "replace",
    ("headlight", "crack", 0): "replace",
    ("taillight", "shatter", 0): "replace",
    ("mirror", "shatter", 0): "replace",
    ("mirror", "misalignment", 0): "repair",
    ("bumper", "dent", 0): "repair",
    ("bumper", "dent", 1): "repair",
    ("bumper", "dent", 2): "replace",
    ("bumper", "tear", 0): "repair",
    ("bumper", "tear", 1): "replace",
    ("bumper", "puncture", 0): "repair",
    ("door", "dent", 0): "repair",
    ("door", "dent", 1): "repair",
    ("door", "dent", 2): "replace",
    ("door", "deformation", 1): "replace",
    ("door", "deformation", 2): "replace",
    ("hood", "dent", 0): "repair",
    ("hood", "dent", 1): "repair",
    ("hood", "dent", 2): "replace",
    ("fender", "dent", 0): "repair",
    ("fender", "dent", 1): "repair",
    ("fender", "dent", 2): "replace",
    ("quarter_panel", "deformation", 1): "replace",
}


def rule_repair_or_replace(part: str, damage_type: str, severity_index: int) -> str:
    """Look up heuristic recommendation; fallback on severity."""
    hit = _RULES.get((part, damage_type, severity_index))
    if hit is not None:
        return hit
    # Fallback: severe / total_loss -> replace
    return "replace" if severity_index >= 2 else "repair"


def build_part_assessment(
    *,
    detection: dict[str, Any],
    damage_probs: dict[str, float],
    damage_threshold: float,
    severity: dict[str, Any] | None,
    pretrained_baseline: bool,
    use_rule_override: bool = False,
) -> dict[str, Any]:
    """Build the per-part assessment dict."""
    damage_types = sorted(
        [(k, v) for k, v in damage_probs.items() if v >= damage_threshold],
        key=lambda kv: -kv[1],
    )
    primary_damage = damage_types[0][0] if damage_types else None

    recommendation = None
    repair_prob = None
    replace_prob = None

    if severity is not None:
        recommendation = severity.get("recommendation")
        repair_prob = severity.get("repair_probability")
        replace_prob = severity.get("replace_probability")
        if use_rule_override and primary_damage is not None:
            recommendation = rule_repair_or_replace(
                detection["part"], primary_damage, severity["grade_index"]
            )

    return {
        "part": detection["part"],
        "class_id": detection["class_id"],
        "detection_confidence": detection["confidence"],
        "bbox_xyxy_px": list(detection["bbox_xyxy_px"]),
        "bbox_xyxy_norm": list(detection["bbox_xyxy_norm"]),
        "damage_types": [
            {"type": k, "probability": round(float(v), 4)} for k, v in damage_types
        ],
        "damage_probs_all": {k: round(float(v), 4) for k, v in damage_probs.items()},
        "primary_damage_type": primary_damage,
        "severity": (
            {
                "grade": severity["grade"],
                "grade_index": severity["grade_index"],
                "grade_confidence": round(float(severity["grade_confidence"]), 4),
                "probs": {k: round(float(v), 4) for k, v in severity["severity_probs"].items()},
            }
            if severity is not None
            else None
        ),
        "recommendation": recommendation,
        "repair_probability": None if repair_prob is None else round(float(repair_prob), 4),
        "replace_probability": None if replace_prob is None else round(float(replace_prob), 4),
        "pretrained_baseline": pretrained_baseline,
    }


def build_report(
    *,
    image_id: str,
    image_width: int,
    image_height: int,
    parts: list[dict[str, Any]],
    pretrained_baseline: bool,
    model_versions: dict[str, str],
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Top-level JSON report for one image."""
    n_damaged = sum(1 for p in parts if p.get("primary_damage_type") is not None)
    n_replace = sum(1 for p in parts if p.get("recommendation") == "replace")
    overall = "clean" if not parts else (
        "total_loss" if n_replace >= max(3, len(parts) // 2)
        else ("major_damage" if n_replace >= 1 else "minor_damage")
    )
    return {
        "image_id": image_id,
        "image_width": image_width,
        "image_height": image_height,
        "parts_detected": len(parts),
        "parts_damaged": n_damaged,
        "parts_requiring_replacement": n_replace,
        "overall_assessment": overall,
        "parts": parts,
        "pretrained_baseline": pretrained_baseline,
        "model_versions": model_versions,
        "warnings": warnings or [],
        "schema_version": "1.0",
    }

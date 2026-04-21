"""Tests for the no_damage class handling in L2 V2 vocabulary.

Verifies that when ``damage_probs`` includes ``no_damage`` and it dominates,
the part is reported as undamaged and the severity block is suppressed.
"""

from __future__ import annotations

from inference.postprocessing import build_part_assessment, build_report


def _detection(part: str = "door") -> dict:
    return {
        "part": part,
        "class_id": 3,
        "confidence": 0.95,
        "bbox_xyxy_px": (10, 20, 100, 200),
        "bbox_xyxy_norm": (0.01, 0.04, 0.1, 0.2),
    }


def _severity(grade: str = "severe", idx: int = 2) -> dict:
    return {
        "grade": grade,
        "grade_index": idx,
        "grade_confidence": 0.7,
        "severity_probs": {"minor": 0.1, "moderate": 0.1, "severe": 0.7, "total_loss": 0.1},
        "repair_probability": 0.3,
        "replace_probability": 0.7,
        "recommendation": "replace",
    }


def test_no_damage_wins_clears_damage_block():
    # V2 vocabulary: no_damage dominates -> part is reported as clean
    probs = {
        "no_damage": 0.92,
        "dent": 0.05,
        "scratch": 0.02,
        "crack": 0.0,
        "shatter": 0.0,
        "tear": 0.0,
        "deformation": 0.0,
        "paint_loss": 0.0,
        "puncture": 0.0,
        "misalignment": 0.0,
    }
    out = build_part_assessment(
        detection=_detection(),
        damage_probs=probs,
        damage_threshold=0.5,
        severity=_severity(),
        pretrained_baseline=False,
    )
    assert out["damaged"] is False
    assert out["damage_types"] == []
    assert out["primary_damage_type"] is None
    assert out["severity"] is None
    assert out["recommendation"] is None
    assert out["repair_probability"] is None
    # probs still echoed for audit trail
    assert out["damage_probs_all"]["no_damage"] == 0.92


def test_no_damage_loses_to_a_damage_type():
    # Even though no_damage is above 0.5, dent is higher -> part is damaged
    probs = {
        "no_damage": 0.55,
        "dent": 0.82,
        "scratch": 0.1,
        "crack": 0.0,
        "shatter": 0.0,
        "tear": 0.0,
        "deformation": 0.0,
        "paint_loss": 0.0,
        "puncture": 0.0,
        "misalignment": 0.0,
    }
    out = build_part_assessment(
        detection=_detection(),
        damage_probs=probs,
        damage_threshold=0.5,
        severity=_severity(),
        pretrained_baseline=False,
    )
    assert out["damaged"] is True
    assert out["primary_damage_type"] == "dent"
    assert out["severity"] is not None
    assert out["recommendation"] == "replace"


def test_v1_vocabulary_unchanged():
    # No no_damage class present -> legacy V1 behavior (dent wins)
    probs = {
        "dent": 0.9,
        "scratch": 0.1,
        "crack": 0.0,
        "shatter": 0.0,
        "tear": 0.0,
        "deformation": 0.0,
        "paint_loss": 0.0,
        "puncture": 0.0,
        "misalignment": 0.0,
    }
    out = build_part_assessment(
        detection=_detection(),
        damage_probs=probs,
        damage_threshold=0.5,
        severity=_severity(),
        pretrained_baseline=False,
    )
    assert out["damaged"] is True
    assert out["primary_damage_type"] == "dent"
    assert out["severity"] is not None


def test_all_parts_clean_overall_is_clean():
    clean_part = build_part_assessment(
        detection=_detection("door"),
        damage_probs={"no_damage": 0.95, "dent": 0.02, "scratch": 0.03},
        damage_threshold=0.5,
        severity=_severity(),
        pretrained_baseline=False,
    )
    other_clean = build_part_assessment(
        detection=_detection("hood"),
        damage_probs={"no_damage": 0.9, "dent": 0.05, "scratch": 0.05},
        damage_threshold=0.5,
        severity=_severity(),
        pretrained_baseline=False,
    )
    report = build_report(
        image_id="test.jpg",
        image_width=640,
        image_height=480,
        parts=[clean_part, other_clean],
        pretrained_baseline=False,
        model_versions={"layer1": "v1", "layer2": "v2", "layer3": "v1"},
    )
    assert report["overall_assessment"] == "clean"
    assert report["parts_damaged"] == 0
    assert report["parts_requiring_replacement"] == 0


def test_mixed_parts_clean_and_damaged():
    clean = build_part_assessment(
        detection=_detection("door"),
        damage_probs={"no_damage": 0.95, "dent": 0.02},
        damage_threshold=0.5,
        severity=_severity(),
        pretrained_baseline=False,
    )
    damaged = build_part_assessment(
        detection=_detection("hood"),
        damage_probs={"no_damage": 0.05, "dent": 0.92},
        damage_threshold=0.5,
        severity=_severity("severe", 2),
        pretrained_baseline=False,
    )
    report = build_report(
        image_id="test.jpg",
        image_width=640,
        image_height=480,
        parts=[clean, damaged],
        pretrained_baseline=False,
        model_versions={"layer1": "v1", "layer2": "v2", "layer3": "v1"},
    )
    # One damaged part with recommendation=replace -> major_damage
    assert report["overall_assessment"] == "major_damage"
    assert report["parts_damaged"] == 1
    assert report["parts_requiring_replacement"] == 1

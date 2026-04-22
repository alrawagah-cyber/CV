"""Tests for active learning uncertainty scoring and drift monitoring."""

from __future__ import annotations

import math

import numpy as np
import pytest

from inference.uncertainty import (
    compute_l2_uncertainty,
    compute_l3_uncertainty,
    should_flag_for_review,
)


# --------------------------------------------------------------------------- #
# Uncertainty helpers
# --------------------------------------------------------------------------- #
class TestEntropy:
    def test_uniform_distribution_high_entropy(self):
        probs = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        h = compute_l2_uncertainty(probs)
        assert h == pytest.approx(math.log(4), rel=1e-6)

    def test_peaked_distribution_low_entropy(self):
        probs = {"a": 0.99, "b": 0.005, "c": 0.005}
        h = compute_l2_uncertainty(probs)
        assert h < 0.1

    def test_empty_distribution(self):
        assert compute_l2_uncertainty({}) == 0.0

    def test_single_class(self):
        assert compute_l2_uncertainty({"only": 1.0}) == 0.0

    def test_l3_uncertainty_matches_l2_for_same_input(self):
        probs = {"minor": 0.5, "moderate": 0.3, "severe": 0.15, "total_loss": 0.05}
        assert compute_l3_uncertainty(probs) == compute_l2_uncertainty(probs)


class TestShouldFlag:
    thresholds = {
        "l2_entropy_threshold": 1.5,
        "l3_entropy_threshold": 1.0,
        "min_detection_confidence": 0.4,
    }

    def test_low_confidence_flags(self):
        assert should_flag_for_review(0.0, 0.0, 0.2, self.thresholds) is True

    def test_high_l2_entropy_flags(self):
        assert should_flag_for_review(2.0, 0.0, 0.9, self.thresholds) is True

    def test_high_l3_entropy_flags(self):
        assert should_flag_for_review(0.0, 1.5, 0.9, self.thresholds) is True

    def test_all_good_no_flag(self):
        assert should_flag_for_review(0.5, 0.3, 0.9, self.thresholds) is False

    def test_boundary_confidence_no_flag(self):
        # Exactly at threshold should NOT flag (not strictly less).
        assert should_flag_for_review(0.0, 0.0, 0.4, self.thresholds) is False


# --------------------------------------------------------------------------- #
# Integration: uncertainty flows through build_part_assessment
# --------------------------------------------------------------------------- #
class TestBuildPartAssessmentUncertainty:
    def _make_detection(self, confidence: float = 0.85):
        return {
            "part": "bumper",
            "class_id": 0,
            "confidence": confidence,
            "bbox_xyxy_px": [100, 100, 300, 300],
            "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.3],
        }

    def test_no_active_learning_fields_when_disabled(self):
        from inference.postprocessing import build_part_assessment

        result = build_part_assessment(
            detection=self._make_detection(),
            damage_probs={"dent": 0.9, "scratch": 0.1},
            damage_threshold=0.5,
            severity={
                "grade": "minor",
                "grade_index": 0,
                "grade_confidence": 0.8,
                "severity_probs": {"minor": 0.8, "moderate": 0.1, "severe": 0.05, "total_loss": 0.05},
                "recommendation": "repair",
                "repair_probability": 0.8,
                "replace_probability": 0.2,
            },
            pretrained_baseline=False,
            active_learning_thresholds=None,
        )
        assert result["uncertainty_score"] is None
        assert result["flagged_for_review"] is False

    def test_uncertainty_score_present_when_enabled(self):
        from inference.postprocessing import build_part_assessment

        thresholds = {
            "l2_entropy_threshold": 1.5,
            "l3_entropy_threshold": 1.0,
            "min_detection_confidence": 0.4,
        }
        result = build_part_assessment(
            detection=self._make_detection(confidence=0.9),
            damage_probs={"dent": 0.9, "scratch": 0.1},
            damage_threshold=0.5,
            severity={
                "grade": "minor",
                "grade_index": 0,
                "grade_confidence": 0.8,
                "severity_probs": {"minor": 0.8, "moderate": 0.1, "severe": 0.05, "total_loss": 0.05},
                "recommendation": "repair",
                "repair_probability": 0.8,
                "replace_probability": 0.2,
            },
            pretrained_baseline=False,
            active_learning_thresholds=thresholds,
        )
        assert result["uncertainty_score"] is not None
        assert isinstance(result["uncertainty_score"], float)
        assert result["flagged_for_review"] is False

    def test_low_confidence_triggers_flag(self):
        from inference.postprocessing import build_part_assessment

        thresholds = {
            "l2_entropy_threshold": 1.5,
            "l3_entropy_threshold": 1.0,
            "min_detection_confidence": 0.4,
        }
        result = build_part_assessment(
            detection=self._make_detection(confidence=0.2),
            damage_probs={"dent": 0.9, "scratch": 0.1},
            damage_threshold=0.5,
            severity={
                "grade": "minor",
                "grade_index": 0,
                "grade_confidence": 0.8,
                "severity_probs": {"minor": 0.8, "moderate": 0.1, "severe": 0.05, "total_loss": 0.05},
                "recommendation": "repair",
                "repair_probability": 0.8,
                "replace_probability": 0.2,
            },
            pretrained_baseline=False,
            active_learning_thresholds=thresholds,
        )
        assert result["flagged_for_review"] is True


# --------------------------------------------------------------------------- #
# build_report review_flags_count
# --------------------------------------------------------------------------- #
class TestBuildReportReviewFlags:
    def test_review_flags_count(self):
        from inference.postprocessing import build_report

        parts = [
            {"part": "bumper", "flagged_for_review": True, "recommendation": "repair", "damaged": True},
            {"part": "door", "flagged_for_review": False, "recommendation": "repair", "damaged": True},
            {"part": "hood", "flagged_for_review": True, "recommendation": None, "damaged": True},
        ]
        report = build_report(
            image_id="test",
            image_width=640,
            image_height=480,
            parts=parts,
            pretrained_baseline=False,
            model_versions={"layer1": "v1", "layer2": "v1", "layer3": "v1"},
        )
        assert report["review_flags_count"] == 2

    def test_zero_flags_when_none_flagged(self):
        from inference.postprocessing import build_report

        parts = [
            {"part": "bumper", "flagged_for_review": False, "recommendation": "repair", "damaged": True},
        ]
        report = build_report(
            image_id="test",
            image_width=640,
            image_height=480,
            parts=parts,
            pretrained_baseline=False,
            model_versions={"layer1": "v1", "layer2": "v1", "layer3": "v1"},
        )
        assert report["review_flags_count"] == 0


# --------------------------------------------------------------------------- #
# DriftMonitor
# --------------------------------------------------------------------------- #
class TestDriftMonitor:
    def test_record_and_stats(self):
        from api.drift_monitor import DriftMonitor

        mon = DriftMonitor(window=100)
        report = {
            "overall_assessment": "minor_damage",
            "parts": [
                {"part": "bumper", "primary_damage_type": "dent", "severity": {"grade": "minor"}},
                {"part": "door", "primary_damage_type": "scratch", "severity": {"grade": "moderate"}},
            ],
        }
        mon.record(report)
        stats = mon.get_stats()
        assert stats["total_reports"] == 1
        assert stats["total_parts"] == 2
        assert stats["damage_type_distribution"]["dent"] == 1
        assert stats["severity_distribution"]["moderate"] == 1
        assert stats["overall_assessment_distribution"]["minor_damage"] == 1
        assert stats["parts_distribution"]["bumper"] == 1

    def test_rolling_window(self):
        from api.drift_monitor import DriftMonitor

        mon = DriftMonitor(window=3)
        for i in range(5):
            mon.record({"overall_assessment": "minor_damage", "parts": []})
        stats = mon.get_stats()
        assert stats["total_reports"] == 3

    def test_empty_stats(self):
        from api.drift_monitor import DriftMonitor

        mon = DriftMonitor()
        stats = mon.get_stats()
        assert stats["total_reports"] == 0
        assert stats["avg_parts_per_report"] == 0.0


# --------------------------------------------------------------------------- #
# Pydantic schema round-trip
# --------------------------------------------------------------------------- #
class TestSchemaFields:
    def test_part_assessment_schema_accepts_uncertainty(self):
        from api.schemas import PartAssessment

        data = {
            "part": "bumper",
            "class_id": 0,
            "detection_confidence": 0.85,
            "bbox_xyxy_px": [100, 100, 300, 300],
            "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.3],
            "damaged": True,
            "damage_types": [{"type": "dent", "probability": 0.9}],
            "damage_probs_all": {"dent": 0.9},
            "pretrained_baseline": False,
            "uncertainty_score": 0.42,
            "flagged_for_review": True,
        }
        pa = PartAssessment.model_validate(data)
        assert pa.uncertainty_score == 0.42
        assert pa.flagged_for_review is True

    def test_claim_report_schema_accepts_review_flags(self):
        from api.schemas import ClaimReport

        data = {
            "image_id": "test",
            "image_width": 640,
            "image_height": 480,
            "parts_detected": 1,
            "parts_damaged": 1,
            "parts_requiring_replacement": 0,
            "review_flags_count": 1,
            "overall_assessment": "minor_damage",
            "parts": [
                {
                    "part": "bumper",
                    "class_id": 0,
                    "detection_confidence": 0.85,
                    "bbox_xyxy_px": [100, 100, 300, 300],
                    "bbox_xyxy_norm": [0.1, 0.1, 0.3, 0.3],
                    "damaged": True,
                    "damage_types": [],
                    "damage_probs_all": {},
                    "pretrained_baseline": False,
                    "uncertainty_score": 0.5,
                    "flagged_for_review": True,
                }
            ],
            "pretrained_baseline": False,
            "model_versions": {"layer1": "v1", "layer2": "v1", "layer3": "v1"},
        }
        cr = ClaimReport.model_validate(data)
        assert cr.review_flags_count == 1


# --------------------------------------------------------------------------- #
# End-to-end: stub assessor produces uncertainty fields
# --------------------------------------------------------------------------- #
def test_assessor_produces_uncertainty_fields(stub_assessor, sample_rgb_image: np.ndarray):
    """Verify the full pipeline returns uncertainty_score and flagged_for_review."""
    # Enable active learning on the stub assessor config.
    stub_assessor.cfg.active_learning_thresholds = {
        "l2_entropy_threshold": 1.5,
        "l3_entropy_threshold": 1.0,
        "min_detection_confidence": 0.4,
    }
    report = stub_assessor.assess(sample_rgb_image, image_id="unc_test")
    part = report["parts"][0]
    assert "uncertainty_score" in part
    assert "flagged_for_review" in part
    assert isinstance(part["uncertainty_score"], float)
    assert "review_flags_count" in report

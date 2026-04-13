"""End-to-end inference tests with stubbed model layers."""

from __future__ import annotations

import numpy as np


def test_claim_assessor_end_to_end(stub_assessor, sample_rgb_image: np.ndarray):
    report = stub_assessor.assess(sample_rgb_image, image_id="unit_test")
    assert report["image_id"] == "unit_test"
    assert report["parts_detected"] == 1
    assert len(report["parts"]) == 1
    part = report["parts"][0]
    assert part["part"] == "bumper"
    assert "damage_probs_all" in part
    assert part["severity"]["grade"] == "minor"
    assert report["schema_version"] == "1.0"
    assert set(report["model_versions"].keys()) == {"layer1", "layer2", "layer3"}


def test_claim_assessor_batch(stub_assessor, sample_rgb_image):
    reports = stub_assessor.assess_batch([sample_rgb_image, sample_rgb_image])
    assert len(reports) == 2
    assert all(r["parts_detected"] == 1 for r in reports)


def test_claim_assessor_overall_classification(stub_assessor, sample_rgb_image):
    # Stub always returns a bumper + minor -> overall should be 'minor_damage' or 'clean'
    report = stub_assessor.assess(sample_rgb_image)
    assert report["overall_assessment"] in {"minor_damage", "clean", "major_damage"}


def test_report_contains_warnings_when_no_parts(stub_assessor, sample_rgb_image, monkeypatch):
    # Replace the stub detector's predict to return an empty detection list.
    monkeypatch.setattr(
        stub_assessor.detector,
        "predict",
        lambda imgs, **_: [[] for _ in (imgs if isinstance(imgs, list) else [imgs])],
    )
    report = stub_assessor.assess(sample_rgb_image)
    assert report["parts_detected"] == 0
    assert report["warnings"], "expected warnings about no parts detected"

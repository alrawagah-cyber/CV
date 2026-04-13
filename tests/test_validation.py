"""Smoke tests for the dataset validator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.validate_dataset import validate_layer1, validate_layer2, validate_layer3

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_layer1_samples_pass():
    # Point directly at the bundled samples dir.
    errors = validate_layer1(REPO_ROOT / "data" / "layer1")
    assert errors == [] or all("images without matching" not in e.lower() for e in errors), errors


def test_layer2_sample_csv_passes():
    errors = validate_layer2(
        REPO_ROOT / "data" / "layer2",
        REPO_ROOT / "data" / "layer2" / "annotations.sample.csv",
    )
    assert errors == []


def test_layer3_sample_csv_passes():
    errors = validate_layer3(
        REPO_ROOT / "data" / "layer3",
        REPO_ROOT / "data" / "layer3" / "annotations.sample.csv",
    )
    assert errors == []


def test_layer2_detects_bad_value(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "image": ["x.jpg"],
            "dent": [2],  # out of range
            "scratch": [0],
            "crack": [0],
            "shatter": [0],
            "tear": [0],
            "deformation": [0],
            "paint_loss": [0],
            "puncture": [0],
            "misalignment": [0],
        }
    ).to_csv(bad, index=False)
    (tmp_path / "crops").mkdir()
    errors = validate_layer2(tmp_path, bad)
    assert any("non-binary" in e for e in errors), errors


def test_layer3_accepts_unknown_placeholders(tmp_path: Path):
    """Bootstrap data from the L3 Roboflow ingestor uses 'unknown' for
    part + damage_type. The validator must accept these."""
    bootstrap = tmp_path / "bs.csv"
    pd.DataFrame(
        {
            "image": ["x.jpg"],
            "part": ["unknown"],
            "damage_type": ["unknown"],
            "severity": [2],
            "repair_or_replace": [1],
        }
    ).to_csv(bootstrap, index=False)
    (tmp_path / "crops").mkdir()
    errors = validate_layer3(tmp_path, bootstrap)
    assert errors == []


def test_layer3_flags_blank_repair_replace(tmp_path: Path):
    """If the L3 ingestor was run with --no-rule-repair, the column is blank
    and the validator must reject until the annotator fills it in."""
    bad = tmp_path / "bs.csv"
    pd.DataFrame(
        {
            "image": ["x.jpg"],
            "part": ["unknown"],
            "damage_type": ["unknown"],
            "severity": [1],
            "repair_or_replace": [""],
        }
    ).to_csv(bad, index=False)
    (tmp_path / "crops").mkdir()
    errors = validate_layer3(tmp_path, bad)
    assert any("repair_or_replace is blank" in e for e in errors), errors


def test_layer3_detects_bad_severity(tmp_path: Path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "image": ["x.jpg"],
            "part": ["bumper"],
            "damage_type": ["dent"],
            "severity": [9],  # out of range
            "repair_or_replace": [0],
        }
    ).to_csv(bad, index=False)
    (tmp_path / "crops").mkdir()
    errors = validate_layer3(tmp_path, bad)
    assert any("severity" in e for e in errors), errors

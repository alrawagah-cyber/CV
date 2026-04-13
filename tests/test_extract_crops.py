"""Tests for scripts/extract_crops.py using a stubbed detector."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.layer2_damage import DEFAULT_DAMAGE_CLASSES


@pytest.fixture
def _images_dir(tmp_path: Path) -> Path:
    d = tmp_path / "src_images"
    d.mkdir()
    rng = np.random.default_rng(1)
    for i in range(3):
        arr = rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
        Image.fromarray(arr).save(d / f"shot_{i:03d}.jpg")
    return d


def _patch_detector(monkeypatch):
    """Replace PartDetector inside extract_crops with a stub yielding two boxes."""
    import scripts.extract_crops as mod
    from tests.conftest import StubDetection, StubDetector

    class TwoBoxDetector(StubDetector):
        def predict(self, images, **kwargs):  # noqa: ARG002
            out = []
            det_a = StubDetection()
            det_a.part = "bumper"
            det_a.confidence = 0.9
            det_a.bbox_xyxy_px = (10, 20, 80, 100)
            det_b = StubDetection()
            det_b.part = "headlight"
            det_b.confidence = 0.8
            det_b.bbox_xyxy_px = (90, 15, 150, 70)
            for _ in images if isinstance(images, list) else [images]:
                out.append([det_a, det_b])
            return out

    monkeypatch.setattr(mod, "PartDetector", TwoBoxDetector)


def test_extract_crops_layer2_writes_crops_and_manifest(_images_dir, tmp_path: Path, monkeypatch):
    _patch_detector(monkeypatch)
    import scripts.extract_crops as mod

    out_dir = tmp_path / "l2_crops"
    manifest = tmp_path / "l2.csv"

    argv = [
        "extract_crops.py",
        "--source",
        str(_images_dir),
        "--weights",
        "yolov8x.pt",
        "--layer",
        "2",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--conf",
        "0.1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert mod.main() == 0

    # 3 images × 2 boxes = 6 crops
    files = sorted(out_dir.iterdir())
    assert len(files) == 6

    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    expected_cols = {
        "image",
        "source_image",
        "part",
        "detection_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "source_width",
        "source_height",
        *DEFAULT_DAMAGE_CLASSES,
    }
    assert expected_cols.issubset(rows[0].keys())
    assert {r["part"] for r in rows} == {"bumper", "headlight"}
    # All damage-type columns start zeroed
    for r in rows:
        for c in DEFAULT_DAMAGE_CLASSES:
            assert r[c] == "0"


def test_extract_crops_layer3_manifest_has_placeholders(_images_dir, tmp_path: Path, monkeypatch):
    _patch_detector(monkeypatch)
    import scripts.extract_crops as mod

    out_dir = tmp_path / "l3_crops"
    manifest = tmp_path / "l3.csv"

    argv = [
        "extract_crops.py",
        "--source",
        str(_images_dir),
        "--layer",
        "3",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest),
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert mod.main() == 0

    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    assert {"damage_type", "severity", "repair_or_replace"}.issubset(rows[0].keys())
    assert all(r["severity"] == "" for r in rows)


def test_extract_crops_whitelist_filters_parts(_images_dir, tmp_path: Path, monkeypatch):
    _patch_detector(monkeypatch)
    import scripts.extract_crops as mod

    out_dir = tmp_path / "crops"
    manifest = tmp_path / "m.csv"
    argv = [
        "extract_crops.py",
        "--source",
        str(_images_dir),
        "--layer",
        "2",
        "--out",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--classes",
        "headlight",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert mod.main() == 0

    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    assert rows and {r["part"] for r in rows} == {"headlight"}

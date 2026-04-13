"""Tests for dataset parsing + manifest validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from models.layer2_damage import DEFAULT_DAMAGE_CLASSES
from models.layer3_severity import DEFAULT_SEVERITY_GRADES
from training.datasets import (
    DamageTypeDataset,
    PartDetectionManifest,
    SeverityDataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_parse_sample_yolo_labels():
    sample = REPO_ROOT / "data" / "layer1" / "samples" / "000001.txt"
    lines = PartDetectionManifest.parse_label_file(sample)
    assert len(lines) == 3
    for ll in lines:
        assert 0 <= ll.class_id < 13
        for v in (ll.cx, ll.cy, ll.w, ll.h):
            assert 0 <= v <= 1


def _seed_image(dir_: Path, name: str) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(dir_ / name)


def test_damage_type_dataset_parses_sample_csv(tmp_path: Path):
    # Copy sample CSV, create fake crops so __getitem__ works.
    import shutil

    root = tmp_path / "layer2"
    (root / "crops").mkdir(parents=True)
    src_csv = REPO_ROOT / "data" / "layer2" / "annotations.sample.csv"
    dst_csv = root / "annotations.sample.csv"
    shutil.copy(src_csv, dst_csv)
    # Seed dummy crops
    for i in range(1, 6):
        _seed_image(root / "crops", f"crop_00000{i}.jpg")

    ds = DamageTypeDataset(root=root, annotations_csv=dst_csv, classes=DEFAULT_DAMAGE_CLASSES, transform=None)
    assert len(ds) == 5
    item = ds[0]
    assert item["image"].shape[0] == 3
    assert item["labels"].shape == (len(DEFAULT_DAMAGE_CLASSES),)


def test_severity_dataset_parses_sample_csv(tmp_path: Path):
    import shutil

    root = tmp_path / "layer3"
    (root / "crops").mkdir(parents=True)
    src_csv = REPO_ROOT / "data" / "layer3" / "annotations.sample.csv"
    dst_csv = root / "annotations.sample.csv"
    shutil.copy(src_csv, dst_csv)
    for i in range(1, 6):
        _seed_image(root / "crops", f"crop_00000{i}.jpg")

    ds = SeverityDataset(root=root, annotations_csv=dst_csv, grades=DEFAULT_SEVERITY_GRADES, transform=None)
    assert len(ds) == 5
    item = ds[0]
    assert item["image"].shape[0] == 3
    assert item["severity"].dtype.is_floating_point is False
    assert 0 <= int(item["severity"]) < len(DEFAULT_SEVERITY_GRADES)


def test_damage_dataset_rejects_missing_column(tmp_path: Path):
    import pandas as pd

    root = tmp_path / "l2"
    (root / "crops").mkdir(parents=True)
    csv = root / "bad.csv"
    pd.DataFrame({"image": ["x.jpg"], "dent": [1]}).to_csv(csv, index=False)
    with pytest.raises(ValueError):
        DamageTypeDataset(root=root, annotations_csv=csv, classes=DEFAULT_DAMAGE_CLASSES, transform=None)

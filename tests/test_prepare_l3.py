"""Tests for scripts/prepare_roboflow_l3_dataset.py (detection + classification modes)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from models.layer3_severity import DEFAULT_SEVERITY_GRADES
from scripts.prepare_roboflow_l3_dataset import (
    L3_HEADER,
    _ingest_classification,
    _ingest_detection,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
L3_MAPPING = REPO_ROOT / "configs" / "roboflow_mappings" / "severity.yaml"


# --------------------------------------------------------------------------- #
# Detection mode
# --------------------------------------------------------------------------- #
def _write_detection_export(root: Path, names: list[str], rows: list[tuple[float, ...]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names, "nc": len(names)}))
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    arr = np.zeros((200, 300, 3), dtype=np.uint8)
    Image.fromarray(arr).save(root / "train" / "images" / "im1.jpg")
    (root / "train" / "labels" / "im1.txt").write_text(
        "\n".join(" ".join(str(x) for x in r) for r in rows) + "\n"
    )


def test_l3_detection_emits_severity_rows(tmp_path: Path):
    src = tmp_path / "rf"
    # Source classes: low / medium / high -> minor / moderate / severe
    _write_detection_export(
        src,
        names=["low", "medium", "high"],
        rows=[
            (0, 0.3, 0.3, 0.2, 0.2),
            (1, 0.5, 0.5, 0.2, 0.2),
            (2, 0.7, 0.7, 0.2, 0.2),
        ],
    )
    out = tmp_path / "layer3"
    stats = _ingest_detection(
        input_dir=src,
        output_dir=out,
        splits=["train"],
        prefix="rf",
        mapping_path=L3_MAPPING,
        on_unknown="error",
        margin=0.05,
        use_rule=True,
        min_box_frac=0.0,
        dry_run=False,
    )
    assert stats["boxes_kept"] == 3
    assert stats["per_severity"]["minor"] == 1
    assert stats["per_severity"]["moderate"] == 1
    assert stats["per_severity"]["severe"] == 1

    with open(out / "rf_train.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r for r in rows if r["severity"] == "0"][0]["repair_or_replace"] == "0"  # minor -> repair
    assert [r for r in rows if r["severity"] == "2"][0]["repair_or_replace"] == "1"  # severe -> replace
    # All rows: part + damage_type are placeholder 'unknown'
    for r in rows:
        assert r["part"] == "unknown"
        assert r["damage_type"] == "unknown"
    # Columns match the schema
    assert set(L3_HEADER).issubset(rows[0].keys())


def test_l3_detection_no_rule_leaves_repair_blank(tmp_path: Path):
    src = tmp_path / "rf"
    _write_detection_export(src, names=["severe"], rows=[(0, 0.5, 0.5, 0.2, 0.2)])
    out = tmp_path / "layer3"
    _ingest_detection(
        input_dir=src,
        output_dir=out,
        splits=["train"],
        prefix="",
        mapping_path=L3_MAPPING,
        on_unknown="error",
        margin=0.05,
        use_rule=False,
        min_box_frac=0.0,
        dry_run=False,
    )
    with open(out / "train.csv") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["repair_or_replace"] == ""


def test_l3_detection_errors_on_unknown(tmp_path: Path):
    src = tmp_path / "rf"
    _write_detection_export(src, names=["not_a_real_severity"], rows=[(0, 0.5, 0.5, 0.2, 0.2)])
    with pytest.raises(SystemExit):
        _ingest_detection(
            input_dir=src,
            output_dir=tmp_path / "out",
            splits=["train"],
            prefix="",
            mapping_path=L3_MAPPING,
            on_unknown="error",
            margin=0.05,
            use_rule=True,
            min_box_frac=0.0,
            dry_run=True,
        )


# --------------------------------------------------------------------------- #
# Classification mode
# --------------------------------------------------------------------------- #
def _write_classification_export(root: Path, folders_per_split: dict[str, dict[str, int]]) -> None:
    for split, folders in folders_per_split.items():
        for name, count in folders.items():
            d = root / split / name
            d.mkdir(parents=True)
            for i in range(count):
                Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(d / f"img{i}.jpg")


def test_l3_classification_emits_one_row_per_image(tmp_path: Path):
    src = tmp_path / "rf"
    _write_classification_export(src, {"train": {"minor": 3, "severe": 2, "no_damage": 1}})
    out = tmp_path / "layer3"
    stats = _ingest_classification(
        input_dir=src,
        output_dir=out,
        splits=["train"],
        prefix="cls",
        mapping_path=L3_MAPPING,
        on_unknown="error",
        use_rule=True,
        dry_run=False,
    )
    # no_damage is in the skip set, so 3 + 2 = 5 kept
    assert stats["images_kept"] == 5
    assert stats["per_severity"]["minor"] == 3
    assert stats["per_severity"]["severe"] == 2

    with open(out / "cls_train.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    # repair_or_replace rule: minor -> 0, severe -> 1
    minor_rows = [r for r in rows if r["severity"] == str(DEFAULT_SEVERITY_GRADES.index("minor"))]
    severe_rows = [r for r in rows if r["severity"] == str(DEFAULT_SEVERITY_GRADES.index("severe"))]
    assert all(r["repair_or_replace"] == "0" for r in minor_rows)
    assert all(r["repair_or_replace"] == "1" for r in severe_rows)


def test_l3_classification_skip_unknown_folder(tmp_path: Path):
    src = tmp_path / "rf"
    _write_classification_export(src, {"train": {"minor": 2, "mystery_grade_xyz": 3}})
    out = tmp_path / "layer3"
    stats = _ingest_classification(
        input_dir=src,
        output_dir=out,
        splits=["train"],
        prefix="",
        mapping_path=L3_MAPPING,
        on_unknown="skip",
        use_rule=True,
        dry_run=False,
    )
    assert stats["images_kept"] == 2
    assert "mystery_grade_xyz" in stats["unknown_src_names"]

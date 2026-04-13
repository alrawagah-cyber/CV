"""Tests for scripts/prepare_roboflow_dataset.py — class normalization + remap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.prepare_roboflow_dataset import _norm, load_mapping, prepare

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_norm_collapses_separators():
    assert _norm("Front-Bumper") == "front_bumper"
    assert _norm("front bumper") == "front_bumper"
    assert _norm("FRONT__BUMPER!") == "front_bumper"
    assert _norm("  door  ") == "door"


def test_default_mapping_is_valid():
    target, mapping, skip = load_mapping(REPO_ROOT / "configs" / "roboflow_mappings" / "default.yaml")
    assert "bumper" in target and "wheel" in target
    # a few representative synonyms resolve correctly
    assert mapping[_norm("front_bumper")] == "bumper"
    assert mapping[_norm("Head Lamp")] == "headlight"
    assert mapping[_norm("rear_quarter")] == "quarter_panel"
    assert _norm("license_plate") in skip


def _write_roboflow_export(root: Path, source_names: list[str], rows: list[tuple[int, ...]]) -> None:
    """Create a minimal Roboflow-shaped directory with one image + label under train/."""
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": source_names, "nc": len(source_names)}))
    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    img.save(root / "train" / "images" / "img001.jpg")
    label_lines = [" ".join(str(x) for x in r) for r in rows]
    (root / "train" / "labels" / "img001.txt").write_text("\n".join(label_lines) + "\n")


def test_prepare_remaps_known_classes(tmp_path: Path):
    src = tmp_path / "rf_input"
    _write_roboflow_export(
        src,
        source_names=["front_bumper", "license_plate", "headlamp"],
        rows=[
            (0, 0.5, 0.5, 0.3, 0.2),  # front_bumper -> bumper (target_id 0)
            (1, 0.1, 0.1, 0.05, 0.05),  # license_plate -> skip
            (2, 0.6, 0.3, 0.2, 0.1),  # headlamp -> headlight (target_id 5)
        ],
    )
    out = tmp_path / "layer1"
    stats = prepare(
        input_dir=src,
        mapping_path=REPO_ROOT / "configs" / "roboflow_mappings" / "default.yaml",
        output_dir=out,
        splits=["train"],
        prefix="t",
        on_unknown="error",
        copy_mode="copy",
        dry_run=False,
    )
    assert stats["images"] == 1
    assert stats["boxes_total"] == 3
    assert stats["boxes_kept"] == 2
    assert stats["boxes_dropped"] == 1
    assert stats["boxes_per_target"]["bumper"] == 1
    assert stats["boxes_per_target"]["headlight"] == 1

    # Verify the rewritten label file uses our target IDs (0=bumper, 5=headlight).
    label_files = list((out / "labels").glob("*.txt"))
    assert len(label_files) == 1
    ids = [int(line.split()[0]) for line in label_files[0].read_text().strip().splitlines()]
    assert set(ids) == {0, 5}

    # data.yaml regenerated with our vocabulary
    data_yaml = yaml.safe_load((out / "data.yaml").read_text())
    assert data_yaml["nc"] == 13
    assert data_yaml["names"][0] == "bumper"


def test_prepare_errors_on_unknown_by_default(tmp_path: Path):
    src = tmp_path / "rf_input"
    _write_roboflow_export(src, source_names=["bumper", "spoiler_wing_thing"], rows=[(0, 0.5, 0.5, 0.1, 0.1)])
    with pytest.raises(SystemExit):
        prepare(
            input_dir=src,
            mapping_path=REPO_ROOT / "configs" / "roboflow_mappings" / "default.yaml",
            output_dir=tmp_path / "out",
            splits=["train"],
            prefix="",
            on_unknown="error",
            copy_mode="copy",
            dry_run=True,
        )


def test_prepare_skip_mode_tolerates_unknown(tmp_path: Path):
    src = tmp_path / "rf_input"
    _write_roboflow_export(
        src,
        source_names=["bumper", "spoiler_wing_thing"],
        rows=[(0, 0.5, 0.5, 0.1, 0.1), (1, 0.3, 0.3, 0.1, 0.1)],
    )
    stats = prepare(
        input_dir=src,
        mapping_path=REPO_ROOT / "configs" / "roboflow_mappings" / "default.yaml",
        output_dir=tmp_path / "out",
        splits=["train"],
        prefix="",
        on_unknown="skip",
        copy_mode="copy",
        dry_run=False,
    )
    assert stats["boxes_kept"] == 1
    assert stats["boxes_dropped"] == 1
    assert "spoiler_wing_thing" in stats["unknown_src_names"]

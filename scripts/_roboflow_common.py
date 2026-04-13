"""Shared helpers for the Roboflow-ingest scripts (L1 / L2 / L3).

- _norm           : normalize a class name for case-insensitive matching.
- load_mapping    : parse a mapping YAML -> (target_classes, mapping, skip).
- load_source_classes : read `names:` from a Roboflow data.yaml.
- iterate_split   : locate a split's images/labels dir, handling val vs valid.
- resolve_splits  : expand 'all' into the canonical split list.
- yolo_to_xyxy_px : convert YOLO normalized bbox to pixel xyxy.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _norm(name: str) -> str:
    """Lowercase, collapse non-alphanumerics to '_', strip trailing '_'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def load_mapping(path: Path) -> tuple[list[str], dict[str, str], set[str]]:
    """Parse a roboflow_mappings YAML.

    Returns
    -------
    target : list[str]
        Ordered target vocabulary (indexes become the integer class ids).
    mapping : dict[str, str]
        Normalized source name -> target class.
    skip : set[str]
        Normalized source names to drop silently.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    target = list(cfg.get("target_classes") or [])
    if not target:
        raise ValueError(f"{path}: 'target_classes' missing or empty")
    raw_map = cfg.get("mapping") or {}
    skip = {_norm(s) for s in (cfg.get("skip") or [])}
    mapping: dict[str, str] = {}
    for src, tgt in raw_map.items():
        if tgt not in target:
            raise ValueError(f"{path}: mapping target '{tgt}' not in target_classes")
        mapping[_norm(src)] = tgt
    return target, mapping, skip


def load_source_classes(data_yaml: Path) -> list[str]:
    """Read the `names:` list from a Roboflow data.yaml export."""
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names.keys())]
    if not names:
        raise ValueError(f"{data_yaml}: 'names' missing")
    return [str(n) for n in names]


def resolve_splits(split: str) -> list[str]:
    return ["train", "valid", "test"] if split == "all" else [split]


def iterate_split(input_dir: Path, split: str) -> tuple[Path, Path] | None:
    """Return (images_dir, labels_dir) for a split, or None if absent.

    Roboflow alternates between 'valid' and 'val' for the validation split.
    """
    candidates = [split] + (["val"] if split == "valid" else [])
    for c in candidates:
        imgs = input_dir / c / "images"
        lbls = input_dir / c / "labels"
        if imgs.exists() and lbls.exists():
            return imgs, lbls
    return None


def yolo_to_xyxy_px(
    cx: float, cy: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Convert YOLO normalized (cx, cy, w, h) to pixel (x1, y1, x2, y2)."""
    x1 = max(0, int(round((cx - w / 2) * img_w)))
    y1 = max(0, int(round((cy - h / 2) * img_h)))
    x2 = min(img_w, int(round((cx + w / 2) * img_w)))
    y2 = min(img_h, int(round((cy + h / 2) * img_h)))
    if x2 <= x1:
        x2 = min(img_w, x1 + 1)
    if y2 <= y1:
        y2 = min(img_h, y1 + 1)
    return x1, y1, x2, y2


def resolve_remap(
    src_classes: list[str],
    mapping: dict[str, str],
    skip: set[str],
    target: list[str],
    on_unknown: str,
) -> tuple[dict[int, str | None], list[str]]:
    """Build a source-id -> target-class-name map (None = drop).

    Returns (remap, unknown_names). `unknown_names` is non-empty when the
    caller should surface them (either error or report-only).
    """
    remap: dict[int, str | None] = {}
    unknown: list[str] = []
    target_set = set(target)
    for src_id, src_name in enumerate(src_classes):
        key = _norm(src_name)
        if key in skip:
            remap[src_id] = None
            continue
        if key in mapping:
            remap[src_id] = mapping[key]
            continue
        unknown.append(src_name)
        if on_unknown == "skip":
            remap[src_id] = None
        elif on_unknown == "keep":
            if key not in target_set:
                raise ValueError(
                    f"--on-unknown=keep requires '{src_name}' (normalized '{key}') "
                    f"to already be a target class."
                )
            remap[src_id] = key
        else:  # "error"
            remap[src_id] = None  # placeholder; caller decides
    return remap, unknown

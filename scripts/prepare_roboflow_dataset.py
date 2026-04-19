"""Normalize a Roboflow YOLOv8 export to our 13-class vocabulary and
merge it into ``data/layer1/``.

A Roboflow YOLOv8 export looks like::

    <export_dir>/
    ├── data.yaml               # contains `names: [...]`, `nc: N`
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/                   # optional

Usage
-----

    python scripts/prepare_roboflow_dataset.py \\
        --input ~/Downloads/car-part-q8otu \\
        --mapping configs/roboflow_mappings/default.yaml \\
        --output data/layer1 \\
        --split all \\
        --prefix cpq8otu \\
        --on-unknown error

Options
-------
--split         {train,valid,test,all}  Which sub-split(s) to ingest.
--prefix        String prepended to every output filename to prevent collisions
                when you merge multiple datasets.
--on-unknown    {error,skip,keep}
    error   (default) — abort if a source class has no mapping and is not in `skip`.
    skip    — drop unknown classes silently (log once per name).
    keep    — treat unknown classes as a target class of the same name
              (only makes sense if your target vocabulary is an extension of ours).
--dry-run       Walk the input and print stats without writing anything.
--copy-mode     {copy,symlink,move}   How to place images in the output tree.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("prepare_roboflow_dataset")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm(name: str) -> str:
    """Lowercase + alphanumerics-only (separators -> '_')."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def load_mapping(path: Path) -> tuple[list[str], dict[str, str], set[str]]:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    target = list(cfg.get("target_classes") or [])
    if not target:
        raise ValueError(f"{path}: 'target_classes' missing or empty")
    raw_map = cfg.get("mapping") or {}
    skip = {_norm(s) for s in (cfg.get("skip") or [])}

    # Normalize source keys
    mapping: dict[str, str] = {}
    for src, tgt in raw_map.items():
        if tgt not in target:
            raise ValueError(f"{path}: mapping target '{tgt}' not in target_classes")
        mapping[_norm(str(src))] = tgt
    return target, mapping, skip


def load_source_classes(data_yaml: Path) -> list[str]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        # Roboflow sometimes emits {0: 'car', 1: 'hood', ...}
        names = [names[i] for i in sorted(names.keys())]
    if not names:
        raise ValueError(f"{data_yaml}: 'names' missing")
    return [str(n) for n in names]


def resolve_split(split: str) -> list[str]:
    if split == "all":
        return ["train", "valid", "test"]
    return [split]


def iterate_split(input_dir: Path, split: str) -> tuple[Path, Path] | None:
    """Return (images_dir, labels_dir) if the split exists on disk, else None."""
    # Roboflow sometimes names the validation split 'valid' or 'val'.
    candidates = [split]
    if split == "valid":
        candidates.append("val")
    for c in candidates:
        imgs = input_dir / c / "images"
        lbls = input_dir / c / "labels"
        if imgs.exists() and lbls.exists():
            return imgs, lbls
    return None


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def prepare(
    input_dir: Path,
    mapping_path: Path,
    output_dir: Path,
    splits: list[str],
    prefix: str,
    on_unknown: str,
    copy_mode: str,
    dry_run: bool,
) -> dict[str, Any]:
    target, mapping, skip = load_mapping(mapping_path)
    target_to_id = {name: i for i, name in enumerate(target)}

    src_classes = load_source_classes(input_dir / "data.yaml")
    logger.info("Source dataset classes (%d): %s", len(src_classes), src_classes)

    # Pre-resolve per-source-id target info
    remap: dict[int, int | None] = {}  # src_id -> target_id (None = drop)
    unknown_names: set[str] = set()
    for src_id, src_name in enumerate(src_classes):
        key = _norm(src_name)
        if key in skip:
            remap[src_id] = None
            continue
        if key in mapping:
            remap[src_id] = target_to_id[mapping[key]]
            continue
        # unknown
        unknown_names.add(src_name)
        if on_unknown == "skip":
            remap[src_id] = None
        elif on_unknown == "keep":
            if key not in target_to_id:
                raise ValueError(
                    f"--on-unknown=keep requires '{src_name}' to already exist "
                    f"as a target class (normalized: '{key}')"
                )
            remap[src_id] = target_to_id[key]
        else:  # "error"
            pass  # validated after loop

    if on_unknown == "error" and unknown_names:
        raise SystemExit(
            "Unmapped source classes (edit your mapping YAML or use "
            "--on-unknown {skip,keep}):\n  - " + "\n  - ".join(sorted(unknown_names))
        )

    out_images = output_dir / "images"
    out_labels = output_dir / "labels"
    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "images": 0,
        "labels_written": 0,
        "boxes_total": 0,
        "boxes_kept": 0,
        "boxes_dropped": 0,
        "boxes_per_target": Counter(),
        "unknown_src_names": sorted(unknown_names) if on_unknown != "error" else [],
    }

    for split in splits:
        pair = iterate_split(input_dir, split)
        if pair is None:
            logger.info("Split '%s' not present, skipping.", split)
            continue
        imgs_dir, lbls_dir = pair
        logger.info("Ingesting split '%s' from %s", split, imgs_dir)

        for img_path in sorted(imgs_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            stem = img_path.stem
            label_path = lbls_dir / f"{stem}.txt"

            new_stem = f"{prefix}_{split}_{stem}" if prefix else f"{split}_{stem}"
            new_img = out_images / f"{new_stem}{img_path.suffix.lower()}"
            new_lbl = out_labels / f"{new_stem}.txt"

            # Transform label file (handles both bbox and polygon-segmentation formats).
            new_lines: list[str] = []
            if label_path.exists():
                for i, raw in enumerate(label_path.read_text().splitlines(), start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    parts = raw.split()
                    if len(parts) < 5:
                        logger.warning("%s:%d malformed line, skipping: %r", label_path, i, raw)
                        continue
                    try:
                        src_id = int(parts[0])
                        nums = [float(x) for x in parts[1:]]
                    except ValueError:
                        logger.warning("%s:%d non-numeric fields, skipping: %r", label_path, i, raw)
                        continue

                    if len(nums) == 4:
                        # Plain YOLO bbox: cx cy w h
                        cx, cy, w, h = nums
                    elif len(nums) >= 6 and len(nums) % 2 == 0:
                        # YOLO segmentation polygon: x1 y1 x2 y2 ... -> min/max bbox
                        xs = nums[0::2]
                        ys = nums[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        w = max(1e-6, x_max - x_min)
                        h = max(1e-6, y_max - y_min)
                    else:
                        logger.warning("%s:%d unexpected coord count %d, skipping", label_path, i, len(nums))
                        continue

                    stats["boxes_total"] += 1
                    tgt_id = remap.get(src_id)
                    if tgt_id is None:
                        stats["boxes_dropped"] += 1
                        continue
                    stats["boxes_kept"] += 1
                    stats["boxes_per_target"][target[tgt_id]] += 1
                    new_lines.append(f"{tgt_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Only emit images that have at least one kept box.
            # If the label file was missing, still emit as background-only (Ultralytics is fine with empty .txt).
            if not label_path.exists():
                continue
            if not new_lines:
                # Dataset has labels but all were dropped — skip to avoid noisy negatives.
                continue

            if not dry_run:
                _place(img_path, new_img, copy_mode)
                new_lbl.write_text("\n".join(new_lines) + "\n")

            stats["images"] += 1
            stats["labels_written"] += 1

    # Also write/refresh data.yaml with our target vocabulary.
    if not dry_run:
        data_yaml_out = output_dir / "data.yaml"
        data_yaml_out.write_text(_render_data_yaml(output_dir, target))

    return stats


def _place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:  # copy
        shutil.copy2(src, dst)


def _render_data_yaml(root: Path, names: list[str]) -> str:
    return (
        f"# Auto-generated by scripts/prepare_roboflow_dataset.py\n"
        f"path: {root}\n"
        f"train: images\n"
        f"val: images   # TODO: replace with a disjoint val split\n"
        f"nc: {len(names)}\n"
        f"names:\n" + "".join(f"  - {n}\n" for n in names)
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Roboflow YOLOv8 export directory (contains data.yaml + train/).",
    )
    parser.add_argument(
        "--mapping",
        required=True,
        type=Path,
        help="YAML mapping file (see configs/roboflow_mappings/default.yaml).",
    )
    parser.add_argument("--output", default=Path("data/layer1"), type=Path)
    parser.add_argument("--split", default="all", choices=["train", "valid", "test", "all"])
    parser.add_argument(
        "--prefix", default="", help="Filename prefix to avoid collisions across datasets (e.g. 'cpq8otu')."
    )
    parser.add_argument("--on-unknown", default="error", choices=["error", "skip", "keep"])
    parser.add_argument("--copy-mode", default="copy", choices=["copy", "symlink", "move"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error("Input dir does not exist: %s", args.input)
        return 1
    if not (args.input / "data.yaml").exists():
        logger.error("Missing data.yaml in %s (is this a Roboflow YOLOv8 export?)", args.input)
        return 1

    stats = prepare(
        input_dir=args.input,
        mapping_path=args.mapping,
        output_dir=args.output,
        splits=resolve_split(args.split),
        prefix=args.prefix,
        on_unknown=args.on_unknown,
        copy_mode=args.copy_mode,
        dry_run=args.dry_run,
    )

    print("=" * 64)
    print(f"Images written:   {stats['images']}")
    print(f"Labels written:   {stats['labels_written']}")
    print(f"Boxes (total):    {stats['boxes_total']}")
    print(f"Boxes (kept):     {stats['boxes_kept']}")
    print(f"Boxes (dropped):  {stats['boxes_dropped']}")
    print("Per-target counts:")
    for name, n in stats["boxes_per_target"].most_common():
        print(f"  {name:<16}  {n}")
    if stats["unknown_src_names"]:
        print("Unknown source classes (treated per --on-unknown):")
        for n in stats["unknown_src_names"]:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run a trained (or pretrained) Layer-1 detector over a folder of images
and save expanded bbox crops + an annotation manifest ready for Layer 2
or Layer 3 labeling.

This bridges Layer 1 → Layers 2/3: once you have a working part detector,
you can produce candidate crops in bulk so your annotators only label
damage type and severity on each crop, not bounding boxes.

Usage
-----

Layer 2 (damage-type multi-label) manifest with zeros, ready to flip bits::

    python scripts/extract_crops.py \\
        --source data/layer1/images \\
        --weights runs/layer1/exp/weights/best.pt \\
        --layer 2 \\
        --out data/layer2/crops \\
        --manifest data/layer2/crops_manifest.csv \\
        --conf 0.3 --margin 0.1

Layer 3 (severity) manifest with placeholders::

    python scripts/extract_crops.py \\
        --source data/layer1/images \\
        --weights runs/layer1/exp/weights/best.pt \\
        --layer 3 \\
        --out data/layer3/crops \\
        --manifest data/layer3/crops_manifest.csv

If you pass ``--weights`` pointing at the generic ``yolov8x.pt`` baseline,
you'll get COCO 'car' detections rather than your fine-tuned part
vocabulary — useful for smoke-testing the script, but the resulting
manifest won't be meaningfully pre-filled.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import logging
import sys
from pathlib import Path

from inference.preprocessing import crop as crop_image
from inference.preprocessing import expand_bbox, load_image
from models.layer1_detector import DEFAULT_PART_CLASSES, PartDetector
from models.layer2_damage import DEFAULT_DAMAGE_CLASSES
from models.layer3_severity import DEFAULT_SEVERITY_GRADES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("extract_crops")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMG_EXTS else []
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXTS)


def _layer2_header() -> list[str]:
    # Metadata + zeroed multi-label columns so the CSV opens directly in Excel / Label Studio
    return [
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
    ]


def _layer3_header() -> list[str]:
    return [
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
        "damage_type",
        "severity",
        "repair_or_replace",
    ]


def _render_row(
    layer: int,
    crop_name: str,
    src_name: str,
    part: str,
    conf: float,
    bbox: tuple[int, int, int, int],
    size: tuple[int, int],
) -> list[str]:
    x1, y1, x2, y2 = bbox
    w, h = size
    base = [crop_name, src_name, part, f"{conf:.4f}", str(x1), str(y1), str(x2), str(y2), str(w), str(h)]
    if layer == 2:
        # All damage columns default 0; annotator flips to 1 where present.
        return base + ["0"] * len(DEFAULT_DAMAGE_CLASSES)
    # Layer 3: placeholders — labeler fills these.
    return base + ["", "", ""]  # damage_type, severity, repair_or_replace


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, type=Path, help="Single image or directory (recursively scanned)."
    )
    parser.add_argument(
        "--weights",
        default="yolov8x.pt",
        type=str,
        help="Path to YOLO weights. Use your fine-tuned Layer-1 best.pt in practice.",
    )
    parser.add_argument(
        "--layer", required=True, choices=[2, 3], type=int, help="Which downstream layer's manifest to emit."
    )
    parser.add_argument("--out", required=True, type=Path, help="Output crops directory.")
    parser.add_argument("--manifest", required=True, type=Path, help="Output CSV manifest path.")
    parser.add_argument("--conf", default=0.3, type=float, help="Detection confidence threshold.")
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--margin", default=0.1, type=float, help="Fraction by which to expand each bbox.")
    parser.add_argument("--img-size", default=640, type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--max-per-image", default=32, type=int)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Optional whitelist of part names to keep (e.g. --classes bumper door).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    images = _iter_images(args.source)
    if not images:
        logger.error("No images found under %s", args.source)
        return 1
    logger.info("Found %d images under %s", len(images), args.source)

    out_dir = args.out
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)

    detector = PartDetector(
        weights=args.weights,
        classes=DEFAULT_PART_CLASSES,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
        device=args.device,
    )

    header = _layer2_header() if args.layer == 2 else _layer3_header()
    whitelist = set(args.classes) if args.classes else None

    total_crops = 0
    per_part_counts: dict[str, int] = {}
    images_with_detections = 0

    with contextlib.ExitStack() as stack:
        manifest_fh = None if args.dry_run else stack.enter_context(args.manifest.open("w", newline=""))
        writer = csv.writer(manifest_fh) if manifest_fh else None
        if writer is not None:
            writer.writerow(header)

        for img_path in images:
            try:
                img = load_image(img_path)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", img_path, exc)
                continue

            detections = detector.predict([img])[0]
            if not detections:
                continue
            images_with_detections += 1

            # Keep top-N by confidence
            detections = sorted(detections, key=lambda d: -d.confidence)[: args.max_per_image]

            h, w = int(img.shape[0]), int(img.shape[1])
            for idx, det in enumerate(detections):
                if whitelist is not None and det.part not in whitelist:
                    continue
                bbox = expand_bbox(det.bbox_xyxy_px, w, h, margin=args.margin)
                crop_arr = crop_image(img, bbox)
                crop_name = f"{img_path.stem}_{idx:02d}_{det.part}.jpg"
                if not args.dry_run:
                    from PIL import Image

                    Image.fromarray(crop_arr).save(out_dir / crop_name, quality=92)
                per_part_counts[det.part] = per_part_counts.get(det.part, 0) + 1
                total_crops += 1
                if writer is not None:
                    writer.writerow(
                        _render_row(
                            args.layer,
                            crop_name=crop_name,
                            src_name=str(img_path.name),
                            part=det.part,
                            conf=det.confidence,
                            bbox=bbox,
                            size=(w, h),
                        )
                    )

    # Report
    print("=" * 64)
    print(f"Scanned images:          {len(images)}")
    print(f"Images with detections:  {images_with_detections}")
    print(f"Total crops written:     {total_crops}")
    print(f"Output dir:              {out_dir}")
    print(f"Manifest:                {args.manifest}")
    print("Per-part breakdown:")
    for name, n in sorted(per_part_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<16}  {n}")

    if args.layer == 2:
        print()
        print("Next step: open the manifest in your labeling tool (Label Studio,")
        print("a spreadsheet, etc.) and flip 0 -> 1 in each damage-type column")
        print("for every crop that shows that damage. Then split into train.csv/val.csv.")
    else:
        print()
        print("Next step: fill damage_type (from L2 vocab), severity (0..3), and")
        print("repair_or_replace (0/1) for each row. Then split into train.csv/val.csv.")
        print(f"Valid severity indices: {list(range(len(DEFAULT_SEVERITY_GRADES)))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

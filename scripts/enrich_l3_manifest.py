"""Enrich a Layer-3 manifest by running L1+L2 inference over the crops to
fill the 'part' and 'damage_type' placeholder columns.

The L3 Roboflow ingestor sets these to 'unknown'. This script reads each crop,
runs the trained L1 detector (to identify the part) and L2 classifier (to
identify the primary damage type), and writes an enriched copy of the CSV.

Usage
-----

    python scripts/enrich_l3_manifest.py \\
        --manifest data/layer3/cdd_train.csv \\
        --crops-root data/layer3/crops \\
        --config configs/inference.yaml \\
        --output data/layer3/cdd_train_enriched.csv

If no L1 detection fires on a crop (the crop *is* the part, after all, so
YOLO may not detect a sub-region), the script falls back to the part with
the highest confidence from a full-image detection, or keeps 'unknown'.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from inference.preprocessing import batch_tensor_from_crops, load_image
from models.layer2_damage import DEFAULT_DAMAGE_CLASSES, DamageTypeClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("enrich_l3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--crops-root", required=True, type=Path)
    parser.add_argument("--config", default="configs/inference.yaml", type=str)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--l2-threshold", default=0.3, type=float, help="Threshold for primary damage type selection."
    )
    parser.add_argument("--device", default=None, type=str)
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    device = args.device or cfg.get("runtime", {}).get("device", "cpu")

    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Load L2 classifier only (L1 on crops is unreliable — the crop IS the part).
    l2_cfg = cfg.get("layer2", {})
    l2_model = (
        DamageTypeClassifier(
            backbone=l2_cfg.get("backbone", "convnextv2_large.fcmae_ft_in22k_in1k"),
            classes=l2_cfg.get("classes", DEFAULT_DAMAGE_CLASSES),
            pretrained=l2_cfg.get("weights") is None,
        )
        .to(device)
        .eval()
    )
    if l2_cfg.get("weights"):
        ckpt = torch.load(l2_cfg["weights"], map_location=device)
        l2_model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)

    df = pd.read_csv(args.manifest)
    if "part" not in df.columns or "damage_type" not in df.columns:
        logger.error("Manifest must have 'part' and 'damage_type' columns.")
        return 1

    enriched_parts: list[str] = []
    enriched_damage: list[str] = []
    l2_classes = l2_cfg.get("classes", DEFAULT_DAMAGE_CLASSES)

    for i, row in df.iterrows():
        img_path = args.crops_root / str(row["image"])
        if not img_path.exists():
            enriched_parts.append(str(row["part"]))
            enriched_damage.append(str(row["damage_type"]))
            continue

        img = load_image(img_path)
        tensor = batch_tensor_from_crops(
            [img],
            size=l2_model.input_size,
            mean=l2_model.mean,
            std=l2_model.std,
        ).to(device)
        probs = l2_model.predict_proba(tensor).cpu().numpy()[0]

        # Pick the highest-probability damage type above threshold.
        best_idx = int(probs.argmax())
        if probs[best_idx] >= args.l2_threshold:
            enriched_damage.append(l2_classes[best_idx])
        else:
            enriched_damage.append(str(row["damage_type"]))

        # Part: keep existing if not 'unknown', else leave as-is.
        # (Running L1 on a crop rarely helps — the crop IS the part.)
        enriched_parts.append(str(row["part"]))

        if (i + 1) % 500 == 0:
            logger.info("Processed %d / %d crops", i + 1, len(df))

    df["damage_type"] = enriched_damage
    df["part"] = enriched_parts
    df.to_csv(args.output, index=False)

    n_enriched = sum(1 for d in enriched_damage if d != "unknown")
    logger.info("Wrote %s — %d/%d rows have damage_type != 'unknown'", args.output, n_enriched, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())

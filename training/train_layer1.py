"""Fine-tune Layer 1 (car-part detector) using the Ultralytics trainer.

Ultralytics handles its own training loop, mosaic augmentation, AMP, schedule,
and ONNX export. We read our YAML config, translate it to ultralytics kwargs,
and drive the training.

Usage:
    python training/train_layer1.py --config configs/layer1.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml

from models.layer1_detector import PartDetector


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_layer1")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Layer 1 (YOLO).")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--data", type=str, default=None,
                        help="Override the data yaml path from the config.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info("Loaded config: %s", cfg)

    detector = PartDetector(
        weights=cfg["model"]["weights"],
        classes=cfg["model"].get("classes"),
        device=cfg.get("device"),
        img_size=cfg["training"].get("img_size", 640),
    )

    data_yaml = args.data or cfg["data"]["data_yaml"]

    train_kwargs = {
        "data": data_yaml,
        "epochs": cfg["training"].get("epochs", 100),
        "imgsz": cfg["training"].get("img_size", 640),
        "batch": cfg["training"].get("batch_size", 16),
        "lr0": cfg["training"].get("lr", 1e-3),
        "lrf": cfg["training"].get("lrf", 0.01),
        "momentum": cfg["training"].get("momentum", 0.937),
        "weight_decay": cfg["training"].get("weight_decay", 5e-4),
        "warmup_epochs": cfg["training"].get("warmup_epochs", 3.0),
        "mosaic": cfg["augmentation"].get("mosaic", 1.0),
        "mixup": cfg["augmentation"].get("mixup", 0.0),
        "hsv_h": cfg["augmentation"].get("hsv_h", 0.015),
        "hsv_s": cfg["augmentation"].get("hsv_s", 0.7),
        "hsv_v": cfg["augmentation"].get("hsv_v", 0.4),
        "degrees": cfg["augmentation"].get("degrees", 0.0),
        "translate": cfg["augmentation"].get("translate", 0.1),
        "scale": cfg["augmentation"].get("scale", 0.5),
        "fliplr": cfg["augmentation"].get("fliplr", 0.5),
        "patience": cfg["training"].get("early_stop_patience", 20),
        "project": cfg.get("project_dir", "runs/layer1"),
        "name": cfg.get("run_name", "exp"),
        "amp": cfg["training"].get("amp", True),
        "resume": args.resume,
        "seed": cfg.get("seed", 42),
    }

    logger.info("Starting YOLO training with kwargs: %s", train_kwargs)
    detector.train(**train_kwargs)
    logger.info("Training complete. Best weights saved under %s/%s",
                train_kwargs["project"], train_kwargs["name"])


if __name__ == "__main__":
    main()

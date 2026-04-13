"""Download / cache pretrained weights for all three layers.

- Layer 1: ultralytics YOLO. Specifying the weights name triggers a download.
- Layer 2/3: timm models; creating them with pretrained=True caches the weights.

Run once on a fresh environment before the first inference. Safe to re-run.
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("download_weights")


def download_layer1(weights: str = "yolov8x.pt") -> None:
    from ultralytics import YOLO

    logger.info("Fetching Layer 1 weights: %s", weights)
    YOLO(weights)  # triggers download & cache
    logger.info("Layer 1 OK.")


def download_layer2(backbone: str = "convnextv2_large.fcmae_ft_in22k_in1k") -> None:
    import timm

    logger.info("Fetching Layer 2 backbone: %s", backbone)
    timm.create_model(backbone, pretrained=True, num_classes=0)
    logger.info("Layer 2 OK.")


def download_layer3(backbone: str = "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k") -> None:
    import timm

    logger.info("Fetching Layer 3 backbone: %s", backbone)
    timm.create_model(backbone, pretrained=True, num_classes=0)
    logger.info("Layer 3 OK.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer1", default="yolov8x.pt")
    parser.add_argument("--layer2", default="convnextv2_large.fcmae_ft_in22k_in1k")
    parser.add_argument("--layer3", default="swinv2_large_window12to24_192to384.ms_in22k_ft_in1k")
    parser.add_argument("--only", choices=["1", "2", "3"], default=None)
    args = parser.parse_args()

    try:
        if args.only in (None, "1"):
            download_layer1(args.layer1)
        if args.only in (None, "2"):
            download_layer2(args.layer2)
        if args.only in (None, "3"):
            download_layer3(args.layer3)
    except Exception as exc:
        logger.exception("Failed to download weights: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

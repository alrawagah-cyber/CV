"""Albumentations pipelines shared by the layer 2/3 datasets.

Layer 1 uses Ultralytics' built-in mosaic/augmentation (configured via YAML).
Layer 2 and 3 share this module: both operate on cropped parts.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transform(
    image_size: int,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
    strength: str = "medium",
) -> A.Compose:
    """Training-time augmentation for cropped damage images.

    strength: "light" | "medium" | "heavy"
    """
    common = [
        A.LongestMaxSize(max_size=int(image_size * 1.15)),
        A.PadIfNeeded(
            min_height=int(image_size * 1.15),
            min_width=int(image_size * 1.15),
            border_mode=0,
            fill=0,
        ),
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
    ]

    if strength == "light":
        aug = common + [
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02, p=0.3),
            A.Rotate(limit=10, border_mode=0, p=0.3),
        ]
    elif strength == "heavy":
        aug = common + [
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.6),
            A.Rotate(limit=30, border_mode=0, p=0.6),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(image_size // 20, image_size // 10),
                hole_width_range=(image_size // 20, image_size // 10),
                p=0.2,
            ),
        ]
    else:  # medium
        aug = common + [
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.5),
            A.Rotate(limit=20, border_mode=0, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        ]

    aug += [
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(aug)


def build_val_transform(
    image_size: int,
    mean: tuple[float, ...] = (0.485, 0.456, 0.406),
    std: tuple[float, ...] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """Deterministic transform for validation / inference."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, fill=0),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )


def augmentation_from_config(cfg: dict[str, Any], mean: tuple, std: tuple) -> tuple[A.Compose, A.Compose]:
    """Build (train, val) transforms from a config dict."""
    image_size = int(cfg.get("image_size", 384))
    strength = cfg.get("strength", "medium")
    return (
        build_train_transform(image_size, mean=mean, std=std, strength=strength),
        build_val_transform(image_size, mean=mean, std=std),
    )

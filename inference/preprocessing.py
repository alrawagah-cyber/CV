"""Image preprocessing: loading, crop-expansion, classifier normalization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


def load_image(source: str | Path | np.ndarray | bytes) -> np.ndarray:
    """Load an image from a path/bytes/np.ndarray into an RGB uint8 ndarray."""
    if isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        return arr.astype(np.uint8, copy=False)
    if isinstance(source, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(source)).convert("RGB")
        return np.array(img)
    p = Path(source)
    img = Image.open(p).convert("RGB")
    return np.array(img)


def expand_bbox(
    bbox_xyxy_px: tuple[int, int, int, int],
    img_w: int,
    img_h: int,
    margin: float = 0.1,
) -> tuple[int, int, int, int]:
    """Expand a bounding box by `margin` fraction, clipped to image bounds."""
    x1, y1, x2, y2 = bbox_xyxy_px
    w = x2 - x1
    h = y2 - y1
    dx = int(w * margin)
    dy = int(h * margin)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)
    if nx2 <= nx1 or ny2 <= ny1:
        # Degenerate box fallback: keep the original within-bounds crop
        nx1 = max(0, min(x1, img_w - 1))
        ny1 = max(0, min(y1, img_h - 1))
        nx2 = max(nx1 + 1, min(x2, img_w))
        ny2 = max(ny1 + 1, min(y2, img_h))
    return nx1, ny1, nx2, ny2


def crop(image: np.ndarray, bbox_xyxy_px: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy_px
    return image[y1:y2, x1:x2].copy()


def letterbox_resize(image: np.ndarray, size: int, pad_value: int = 0) -> np.ndarray:
    """Resize the longest side to `size` and pad to a square of `size`x`size`."""
    h, w = image.shape[:2]
    scale = size / max(h, w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    if cv2 is not None:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized = np.array(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))
    canvas = np.full((size, size, image.shape[2] if image.ndim == 3 else 1), pad_value, dtype=image.dtype)
    y0 = (size - new_h) // 2
    x0 = (size - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def to_tensor_normalized(
    image: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...]
) -> torch.Tensor:
    """uint8 HxWxC -> float CxHxW normalized."""
    arr = image.astype(np.float32) / 255.0
    mean_a = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
    std_a = np.array(std, dtype=np.float32).reshape(1, 1, -1)
    arr = (arr - mean_a) / std_a
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def batch_tensor_from_crops(
    crops: Iterable[np.ndarray],
    size: int,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> torch.Tensor:
    """Prep a list of RGB uint8 crops into a normalized (B, 3, size, size) tensor."""
    tensors = [
        to_tensor_normalized(letterbox_resize(c, size), mean, std)
        for c in crops
    ]
    if not tensors:
        return torch.empty(0, 3, size, size)
    return torch.stack(tensors, dim=0)

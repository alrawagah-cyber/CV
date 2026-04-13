"""Generate a synthetic 640x480 'test_car.jpg' so the pipeline can run end-to-end
without needing a real image. Produces a simple rendered placeholder — the
pipeline will likely detect zero parts on this, but it exercises every layer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_synthetic_car(path: str | Path, width: int = 640, height: int = 480) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    bg = rng.integers(60, 100, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(bg)
    d = ImageDraw.Draw(img)

    # "road"
    d.rectangle([0, int(height * 0.7), width, height], fill=(40, 40, 45))
    # "car body"
    d.rounded_rectangle(
        [int(width * 0.15), int(height * 0.35), int(width * 0.85), int(height * 0.72)],
        radius=30, fill=(140, 30, 40), outline=(10, 10, 10), width=3,
    )
    # "windshield"
    d.polygon(
        [
            (int(width * 0.32), int(height * 0.38)),
            (int(width * 0.68), int(height * 0.38)),
            (int(width * 0.62), int(height * 0.50)),
            (int(width * 0.38), int(height * 0.50)),
        ],
        fill=(120, 150, 170), outline=(20, 20, 30),
    )
    # wheels
    d.ellipse([int(width * 0.22), int(height * 0.63), int(width * 0.32), int(height * 0.78)],
              fill=(20, 20, 20))
    d.ellipse([int(width * 0.68), int(height * 0.63), int(width * 0.78), int(height * 0.78)],
              fill=(20, 20, 20))

    img.save(path, quality=92)
    return path


if __name__ == "__main__":
    p = make_synthetic_car(Path(__file__).resolve().parent.parent / "data" / "samples" / "test_car.jpg")
    print(f"Wrote {p}")

"""Run the full pipeline on a single image and print the JSON report.

Usage:
    python scripts/run_inference.py --image data/samples/test_car.jpg
    python scripts/run_inference.py --image path/to/photo.jpg --config configs/inference.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--config", default="configs/inference.yaml", type=str)
    parser.add_argument("--output", default=None, type=str, help="Optional: write JSON report to this file.")
    args = parser.parse_args()

    from inference.claim_assessor import ClaimAssessor

    img_path = Path(args.image)
    if not img_path.exists():
        # Fall back to the synthetic sample if the user passed the default path.
        from scripts.generate_sample_image import make_synthetic_car

        if img_path.name == "test_car.jpg":
            print(f"[info] {img_path} not found; generating a synthetic one.", file=sys.stderr)
            make_synthetic_car(img_path)
        else:
            print(f"[error] Image not found: {img_path}", file=sys.stderr)
            return 1

    assessor = ClaimAssessor.from_config(args.config)
    report = assessor.assess(str(img_path))

    serialized = json.dumps(report, indent=2, default=str)
    print(serialized)
    if args.output:
        Path(args.output).write_text(serialized)
    return 0


if __name__ == "__main__":
    sys.exit(main())

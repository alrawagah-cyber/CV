"""Export each layer to ONNX and TorchScript for production deployment.

Usage:
    python scripts/export_onnx.py --layer all --out exports/
    python scripts/export_onnx.py --layer 2 --weights checkpoints/layer2/layer2_best.pt

Layer 1 uses Ultralytics' own exporter (yields yolov8x.onnx / .torchscript).
Layer 2/3 are exported via torch.onnx.export + torch.jit.trace.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("export_onnx")


def export_layer1(out_dir: Path, weights: str) -> None:
    from models.layer1_detector import PartDetector

    detector = PartDetector(weights=weights)
    logger.info("Exporting Layer 1 (YOLO) to ONNX...")
    onnx_path = detector.export(format="onnx", opset=17)
    ts_path = detector.export(format="torchscript")
    logger.info("Layer 1: onnx=%s torchscript=%s", onnx_path, ts_path)


def export_timm_model(model: torch.nn.Module, name: str, out_dir: Path) -> None:
    model.eval()
    size = getattr(model, "input_size", 384)
    example = torch.randn(1, 3, size, size)

    onnx_path = out_dir / f"{name}.onnx"
    ts_path = out_dir / f"{name}.torchscript.pt"

    logger.info("Tracing TorchScript -> %s", ts_path)
    traced = torch.jit.trace(model, example, strict=False)
    traced.save(str(ts_path))

    logger.info("Exporting ONNX -> %s", onnx_path)
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        opset_version=17,
        input_names=["image"],
        output_names=["output"],
        dynamic_axes={"image": {0: "batch"}, "output": {0: "batch"}},
    )


def export_layer2(out_dir: Path, weights: str | None) -> None:
    from models.layer2_damage import DamageTypeClassifier

    model = DamageTypeClassifier(pretrained=weights is None)
    if weights:
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    export_timm_model(model, "layer2_damage", out_dir)


def export_layer3(out_dir: Path, weights: str | None) -> None:
    from models.layer3_severity import SeverityAssessor

    model = SeverityAssessor(pretrained=weights is None)
    if weights:
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    export_timm_model(model, "layer3_severity", out_dir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--out", type=str, default="exports")
    parser.add_argument("--l1-weights", type=str, default="yolov8x.pt")
    parser.add_argument("--l2-weights", type=str, default=None)
    parser.add_argument("--l3-weights", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.layer in ("1", "all"):
            export_layer1(out_dir, args.l1_weights)
        if args.layer in ("2", "all"):
            export_layer2(out_dir, args.l2_weights)
        if args.layer in ("3", "all"):
            export_layer3(out_dir, args.l3_weights)
    except Exception as exc:
        logger.exception("Export failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

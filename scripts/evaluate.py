"""End-to-end evaluation harness for the 3-layer pipeline.

Runs the full ClaimAssessor on a labeled test set and emits:
    - Per-layer confusion matrices (as CSV + optional matplotlib PNG).
    - Per-class precision / recall / F1 tables.
    - Ordinal severity MAE, accuracy, and calibration stats.
    - Repair/replace accuracy breakdown.
    - A JSON summary for programmatic consumption.

The test set format mirrors the training CSVs:

    test_manifest.csv:
        image,part_gt,damage_type_gt,severity_gt,repair_or_replace_gt

    Where `image` is a path relative to --images-root (full car images, not
    crops — the evaluator runs L1 detection end-to-end).

Alternatively, run in **crop mode** (--crop-mode) where each row is already a
cropped part image and only L2+L3 are evaluated.

Usage
-----

    # Full pipeline (L1 -> L2 -> L3):
    python scripts/evaluate.py \\
        --manifest data/test/test_manifest.csv \\
        --images-root data/test/images \\
        --config configs/inference.yaml \\
        --output-dir eval_results/

    # Crop-only (L2 + L3 only):
    python scripts/evaluate.py \\
        --manifest data/test/crops_manifest.csv \\
        --images-root data/test/crops \\
        --config configs/inference.yaml \\
        --crop-mode \\
        --output-dir eval_results/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("evaluate")


def _confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1
    return cm


def _prf(cm: np.ndarray, labels: list[str]) -> list[dict[str, Any]]:
    """Per-class precision, recall, F1 from a confusion matrix."""
    rows = []
    for i, name in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        support = int(cm[i, :].sum())
        rows.append(
            {
                "class": name,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "support": support,
            }
        )
    return rows


def _ordinal_stats(y_true: list[int], y_pred: list[int], num_classes: int) -> dict[str, float]:
    t = np.array(y_true)
    p = np.array(y_pred)
    acc = float((t == p).mean()) if len(t) else 0.0
    mae = float(np.abs(t - p).mean()) if len(t) else 0.0
    off_by_one = float((np.abs(t - p) <= 1).mean()) if len(t) else 0.0
    return {
        "accuracy": round(acc, 4),
        "mae": round(mae, 4),
        "off_by_one_accuracy": round(off_by_one, 4),
        "n": len(t),
    }


def _calibration_bins(
    y_true: list[int], probs: list[dict[str, float]], labels: list[str], n_bins: int = 10
) -> list[dict[str, Any]]:
    """Reliability diagram data: bin predicted confidence vs actual accuracy."""
    bins: list[dict[str, Any]] = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    confidences: list[float] = []
    corrects: list[int] = []
    for yt, p in zip(y_true, probs, strict=True):
        pred_label = max(p, key=lambda k: p[k])
        pred_conf = p[pred_label]
        correct = 1 if labels[yt] == pred_label else 0
        confidences.append(pred_conf)
        corrects.append(correct)
    conf_arr = np.array(confidences)
    corr_arr = np.array(corrects)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf_arr >= lo) & (conf_arr < hi)
        count = int(mask.sum())
        if count == 0:
            bins.append(
                {
                    "bin_lo": round(lo, 2),
                    "bin_hi": round(hi, 2),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "avg_accuracy": 0.0,
                }
            )
        else:
            bins.append(
                {
                    "bin_lo": round(lo, 2),
                    "bin_hi": round(hi, 2),
                    "count": count,
                    "avg_confidence": round(float(conf_arr[mask].mean()), 4),
                    "avg_accuracy": round(float(corr_arr[mask].mean()), 4),
                }
            )
    ece = sum(abs(b["avg_confidence"] - b["avg_accuracy"]) * b["count"] for b in bins) / max(
        sum(b["count"] for b in bins), 1
    )
    return bins, round(ece, 4)


def evaluate_crops(
    manifest_path: Path,
    images_root: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate L2 + L3 on pre-cropped images."""
    from inference.claim_assessor import ClaimAssessor
    from models.layer2_damage import DEFAULT_DAMAGE_CLASSES
    from models.layer3_severity import DEFAULT_SEVERITY_GRADES

    df = pd.read_csv(manifest_path)
    has_l2 = all(c in df.columns for c in DEFAULT_DAMAGE_CLASSES)
    has_severity = "severity_gt" in df.columns or "severity" in df.columns
    sev_col = "severity_gt" if "severity_gt" in df.columns else "severity"
    rr_col = "repair_or_replace_gt" if "repair_or_replace_gt" in df.columns else "repair_or_replace"

    assessor = ClaimAssessor.from_config(str(config_path))
    results: dict[str, Any] = {"n_samples": len(df), "layers_evaluated": []}

    # --- L2 evaluation ---
    if has_l2:
        logger.info("Evaluating Layer 2 (damage type) on %d crops...", len(df))
        results["layers_evaluated"].append("layer2")
        l2_true_labels: list[list[str]] = []
        l2_pred_labels: list[list[str]] = []
        per_class_tp: dict[str, int] = defaultdict(int)
        per_class_fp: dict[str, int] = defaultdict(int)
        per_class_fn: dict[str, int] = defaultdict(int)

        from inference.preprocessing import batch_tensor_from_crops, load_image

        for _, row in df.iterrows():
            img_path = images_root / str(row["image"])
            if not img_path.exists():
                continue
            img = load_image(img_path)
            tensor = batch_tensor_from_crops(
                [img],
                size=assessor.damage_model.input_size,
                mean=assessor.damage_model.mean,
                std=assessor.damage_model.std,
            ).to(assessor.device)
            probs = assessor.damage_model.predict_proba(tensor).cpu().numpy()[0]

            gt_vec = [int(row[c]) for c in DEFAULT_DAMAGE_CLASSES]
            pred_vec = [1 if p >= assessor.cfg.l2_threshold else 0 for p in probs]

            gt_labels = [c for c, v in zip(DEFAULT_DAMAGE_CLASSES, gt_vec, strict=True) if v == 1]
            pred_labels = [c for c, v in zip(DEFAULT_DAMAGE_CLASSES, pred_vec, strict=True) if v == 1]
            l2_true_labels.append(gt_labels)
            l2_pred_labels.append(pred_labels)

            for c, g, p in zip(DEFAULT_DAMAGE_CLASSES, gt_vec, pred_vec, strict=True):
                if g == 1 and p == 1:
                    per_class_tp[c] += 1
                elif g == 0 and p == 1:
                    per_class_fp[c] += 1
                elif g == 1 and p == 0:
                    per_class_fn[c] += 1

        l2_metrics = []
        for c in DEFAULT_DAMAGE_CLASSES:
            tp, fp, fn = per_class_tp[c], per_class_fp[c], per_class_fn[c]
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-8)
            l2_metrics.append(
                {
                    "class": c,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1": round(f1, 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

        total_tp = sum(per_class_tp.values())
        total_fp = sum(per_class_fp.values())
        total_fn = sum(per_class_fn.values())
        micro_p = total_tp / max(total_tp + total_fp, 1)
        micro_r = total_tp / max(total_tp + total_fn, 1)
        micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-8)

        results["layer2"] = {
            "per_class": l2_metrics,
            "micro_precision": round(micro_p, 4),
            "micro_recall": round(micro_r, 4),
            "micro_f1": round(micro_f1, 4),
        }

    # --- L3 evaluation ---
    if has_severity:
        logger.info("Evaluating Layer 3 (severity) on %d crops...", len(df))
        results["layers_evaluated"].append("layer3")
        sev_true: list[int] = []
        sev_pred: list[int] = []
        sev_probs: list[dict[str, float]] = []
        rr_true: list[int] = []
        rr_pred: list[int] = []

        from inference.preprocessing import batch_tensor_from_crops, load_image

        for _, row in df.iterrows():
            img_path = images_root / str(row["image"])
            if not img_path.exists():
                continue
            gt_sev = int(row[sev_col])
            img = load_image(img_path)
            tensor = batch_tensor_from_crops(
                [img],
                size=assessor.severity_model.input_size,
                mean=assessor.severity_model.mean,
                std=assessor.severity_model.std,
            ).to(assessor.device)
            preds = assessor.severity_model.predict(tensor)
            pred = preds[0]

            sev_true.append(gt_sev)
            sev_pred.append(pred.grade_index)
            sev_probs.append(pred.severity_probs)

            if rr_col in df.columns and not pd.isna(row.get(rr_col)):
                rr_true.append(int(row[rr_col]))
                rr_pred.append(1 if pred.replace_probability >= 0.5 else 0)

        cm = _confusion_matrix(
            [DEFAULT_SEVERITY_GRADES[i] for i in sev_true],
            [DEFAULT_SEVERITY_GRADES[i] for i in sev_pred],
            DEFAULT_SEVERITY_GRADES,
        )
        prf = _prf(cm, DEFAULT_SEVERITY_GRADES)
        ordinal = _ordinal_stats(sev_true, sev_pred, len(DEFAULT_SEVERITY_GRADES))
        cal_bins, ece = _calibration_bins(sev_true, sev_probs, DEFAULT_SEVERITY_GRADES)

        rr_acc = None
        if rr_true:
            rr_correct = sum(1 for t, p in zip(rr_true, rr_pred, strict=True) if t == p)
            rr_acc = round(rr_correct / len(rr_true), 4)

        results["layer3"] = {
            "severity_confusion_matrix": cm.tolist(),
            "severity_labels": DEFAULT_SEVERITY_GRADES,
            "severity_per_class": prf,
            "severity_ordinal": ordinal,
            "calibration_bins": cal_bins,
            "ece": ece,
            "repair_replace_accuracy": rr_acc,
            "repair_replace_n": len(rr_true),
        }

        # Save confusion matrix as CSV
        cm_df = pd.DataFrame(cm, index=DEFAULT_SEVERITY_GRADES, columns=DEFAULT_SEVERITY_GRADES)
        cm_df.to_csv(output_dir / "severity_confusion_matrix.csv")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the car damage assessment pipeline.")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV with ground-truth labels.")
    parser.add_argument(
        "--images-root", required=True, type=Path, help="Root dir for image paths in the manifest."
    )
    parser.add_argument("--config", default="configs/inference.yaml", type=str)
    parser.add_argument("--output-dir", default="eval_results", type=Path)
    parser.add_argument(
        "--crop-mode",
        action="store_true",
        help="Images are already cropped parts (skip L1, evaluate L2+L3 only).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.crop_mode:
        results = evaluate_crops(args.manifest, args.images_root, Path(args.config), args.output_dir)
    else:
        logger.info("Full-pipeline evaluation not yet implemented — use --crop-mode for L2+L3.")
        logger.info("Full-pipeline mode requires a manifest linking full images to per-part GT labels.")
        return 1

    # Write JSON summary
    summary_path = args.output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Evaluation summary written to %s", summary_path)

    # Print headline metrics
    print("=" * 64)
    print(f"Samples evaluated: {results['n_samples']}")
    if "layer2" in results:
        l2 = results["layer2"]
        print("\nLayer 2 (damage type):")
        print(f"  Micro F1:        {l2['micro_f1']}")
        print(f"  Micro Precision: {l2['micro_precision']}")
        print(f"  Micro Recall:    {l2['micro_recall']}")
        for m in l2["per_class"]:
            print(f"    {m['class']:<14}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}")
    if "layer3" in results:
        l3 = results["layer3"]
        print("\nLayer 3 (severity):")
        print(f"  Accuracy:        {l3['severity_ordinal']['accuracy']}")
        print(f"  MAE:             {l3['severity_ordinal']['mae']}")
        print(f"  Off-by-1 Acc:    {l3['severity_ordinal']['off_by_one_accuracy']}")
        print(f"  ECE:             {l3['ece']}")
        if l3["repair_replace_accuracy"] is not None:
            print(f"  Repair/Replace:  {l3['repair_replace_accuracy']} (n={l3['repair_replace_n']})")
        for m in l3["severity_per_class"]:
            print(
                f"    {m['class']:<12}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())

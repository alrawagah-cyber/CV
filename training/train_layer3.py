"""Fine-tune Layer 3: severity ordinal regression + repair/replace head (Swin V2 L).

Usage:
    python training/train_layer3.py --config configs/layer3.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from models.heads import CoralOrdinalHead, coral_loss
from models.layer3_severity import DEFAULT_SEVERITY_GRADES, SeverityAssessor
from training.augmentations import augmentation_from_config
from training.datasets import SeverityDataset
from training.tracking import build_tracker
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_layer3")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _step(
    model: nn.Module, batch: dict[str, Any], device: torch.device, num_classes: int, repair_weight: float
) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["image"].to(device, non_blocking=True)
    y_sev = batch["severity"].to(device, non_blocking=True)
    y_rep = batch["repair_or_replace"].to(device, non_blocking=True)
    out = model(x)
    ord_loss = coral_loss(out["ordinal_logits"], y_sev, num_classes)
    rep_loss = F.binary_cross_entropy_with_logits(out["repair_logit"], y_rep)
    loss = ord_loss + repair_weight * rep_loss
    return loss, {
        "ord_loss": float(ord_loss.item()),
        "rep_loss": float(rep_loss.item()),
        "_batch_size": x.size(0),
    }


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, n = 0.0, 0
    mae_sum = 0.0
    acc_sum = 0
    rep_correct = 0
    num_classes = model.num_classes
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y_sev = batch["severity"].to(device, non_blocking=True)
        y_rep = batch["repair_or_replace"].to(device, non_blocking=True)
        out = model(x)
        ord_loss = coral_loss(out["ordinal_logits"], y_sev, num_classes)
        rep_loss = F.binary_cross_entropy_with_logits(out["repair_logit"], y_rep)
        loss = ord_loss + rep_loss

        probs = torch.sigmoid(out["ordinal_logits"])
        ranks = CoralOrdinalHead.probs_to_rank(probs)
        acc_sum += int((ranks == y_sev).sum().item())
        mae_sum += float((ranks.float() - y_sev.float()).abs().sum().item())

        rep_pred = (torch.sigmoid(out["repair_logit"]) > 0.5).float()
        rep_correct += int((rep_pred == y_rep).sum().item())

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

    denom = max(n, 1)
    val_acc = acc_sum / denom
    val_mae = mae_sum / denom
    rep_acc = rep_correct / denom
    # Combined metric: accuracy minus normalized MAE (range ~ [−1, 1])
    combined = val_acc - (val_mae / max(num_classes - 1, 1)) + 0.25 * rep_acc

    return {
        "val_loss": total_loss / denom,
        "val_acc": float(val_acc),
        "val_mae": float(val_mae),
        "val_repair_acc": float(rep_acc),
        "val_metric": float(combined),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Layer 3 (severity).")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger.info("Loaded config: %s", cfg)

    grades = cfg["model"].get("grades") or DEFAULT_SEVERITY_GRADES

    model = SeverityAssessor(
        backbone=cfg["model"]["backbone"],
        grades=grades,
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.2),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    )

    train_tf, val_tf = augmentation_from_config(cfg.get("augmentation", {}), model.mean, model.std)

    train_ds = SeverityDataset(
        root=cfg["data"]["root"],
        annotations_csv=cfg["data"]["train_csv"],
        grades=grades,
        transform=train_tf,
    )
    val_ds = SeverityDataset(
        root=cfg["data"]["root"],
        annotations_csv=cfg["data"]["val_csv"],
        grades=grades,
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=True,
    )

    num_classes = len(grades)
    repair_weight = cfg["training"].get("repair_loss_weight", 0.5)
    step_fn = lambda m, b, d: _step(m, b, d, num_classes, repair_weight)  # noqa: E731

    tracker = build_tracker(cfg.get("tracking"), run_name=cfg.get("run_name", "layer3"), full_config=cfg)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=step_fn,
        eval_fn=_evaluate,
        epochs=cfg["training"]["epochs"],
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"].get("weight_decay", 0.05),
        warmup_ratio=cfg["training"].get("warmup_ratio", 0.05),
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        amp=cfg["training"].get("amp", True),
        use_ema=cfg["training"].get("use_ema", True),
        ema_decay=cfg["training"].get("ema_decay", 0.999),
        device=cfg.get("device", "cuda"),
        monitor="val_metric",
        monitor_mode="max",
        early_stop_patience=cfg["training"].get("early_stop_patience", 5),
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/layer3"),
        checkpoint_name=cfg.get("checkpoint_name", "layer3_best.pt"),
        tracker=tracker,
    )

    state = trainer.fit()
    logger.info(
        "Training done. Best %s=%.6f at epoch %d", trainer.monitor, state.best_metric, state.best_epoch
    )
    tracker.finish()


if __name__ == "__main__":
    main()

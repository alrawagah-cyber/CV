"""Fine-tune Layer 2: multi-label damage-type classifier (ConvNeXt-V2-L).

Usage:
    python training/train_layer2.py --config configs/layer2.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from models.layer2_damage import DamageTypeClassifier, DEFAULT_DAMAGE_CLASSES
from training.augmentations import augmentation_from_config
from training.datasets import DamageTypeDataset
from training.tracking import build_tracker
from training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("train_layer2")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _step(model: nn.Module, batch: dict[str, Any], device: torch.device,
          loss_fn: nn.Module) -> tuple[torch.Tensor, dict[str, float]]:
    x = batch["image"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    return loss, {"_batch_size": x.size(0)}


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_p, all_y = [], []
    total_loss, n = 0.0, 0
    loss_fn = nn.BCEWithLogitsLoss()
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        probs = torch.sigmoid(logits)
        all_p.append(probs.cpu().numpy())
        all_y.append(y.cpu().numpy())
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

    probs_np = np.concatenate(all_p, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    pred_np = (probs_np > 0.5).astype(np.int32)
    # micro F1 + macro AP-style stats
    tp = ((pred_np == 1) & (y_np == 1)).sum()
    fp = ((pred_np == 1) & (y_np == 0)).sum()
    fn = ((pred_np == 0) & (y_np == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "val_loss": total_loss / max(n, 1),
        "val_precision": float(precision),
        "val_recall": float(recall),
        "val_f1_micro": float(f1),
        "val_metric": float(f1),   # monitored metric
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Layer 2 (damage-type classifier).")
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    cfg = load_config(args.config)
    logger.info("Loaded config: %s", cfg)

    classes = cfg["model"].get("classes") or DEFAULT_DAMAGE_CLASSES

    model = DamageTypeClassifier(
        backbone=cfg["model"]["backbone"],
        classes=classes,
        pretrained=cfg["model"].get("pretrained", True),
        dropout=cfg["model"].get("dropout", 0.2),
        drop_path_rate=cfg["model"].get("drop_path_rate", 0.1),
    )

    train_tf, val_tf = augmentation_from_config(cfg.get("augmentation", {}), model.mean, model.std)

    train_ds = DamageTypeDataset(
        root=cfg["data"]["root"],
        annotations_csv=cfg["data"]["train_csv"],
        classes=classes,
        transform=train_tf,
    )
    val_ds = DamageTypeDataset(
        root=cfg["data"]["root"],
        annotations_csv=cfg["data"]["val_csv"],
        classes=classes,
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True,
        num_workers=cfg["training"].get("num_workers", 4), pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4), pin_memory=True,
    )

    # class-balanced pos_weight for BCE
    pos_weight = None
    if cfg["training"].get("class_balance", True):
        df = train_ds.df
        pos = df[classes].sum(axis=0).values.astype(np.float32)
        neg = len(df) - pos
        pw = np.clip(neg / np.maximum(pos, 1.0), 0.5, 20.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss_fn_device = lambda m, b, d: _step(m, b, d, loss_fn.to(d))  # noqa: E731

    tracker = build_tracker(
        cfg.get("tracking"),
        run_name=cfg.get("run_name", "layer2"),
        full_config=cfg,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        step_fn=loss_fn_device,
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
        checkpoint_dir=cfg.get("checkpoint_dir", "checkpoints/layer2"),
        checkpoint_name=cfg.get("checkpoint_name", "layer2_best.pt"),
        tracker=tracker,
    )

    state = trainer.fit()
    logger.info("Training done. Best %s=%.6f at epoch %d",
                trainer.monitor, state.best_metric, state.best_epoch)
    tracker.finish()


if __name__ == "__main__":
    main()

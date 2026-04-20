"""Generic AMP training loop used by layers 2 and 3.

Layer-specific step functions are injected so we can reuse the boilerplate
(mixed precision, grad clipping, EMA, early stopping, cosine schedule with
warmup, experiment tracking, checkpointing).
"""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.schedulers import WarmupCosineScheduler
from training.tracking import Tracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -math.inf
    best_epoch: int = -1
    patience_left: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self._aligned = False

    def _align_to(self, model: nn.Module) -> None:
        """Move shadow tensors onto the same device/dtype as the live model."""
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k] = self.shadow[k].to(device=v.device, dtype=v.dtype)
        self._aligned = True

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if not self._aligned:
            self._align_to(model)
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Swap model state with EMA weights. Returns original state for restore."""
        if not self._aligned:
            self._align_to(model)
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return original


# ---------------------------------------------------------------------------
StepFn = Callable[[nn.Module, dict[str, Any], torch.device], tuple[torch.Tensor, dict[str, float]]]
"""(model, batch, device) -> (loss, metrics_dict_for_logging)."""

MetricFn = Callable[[nn.Module, DataLoader, torch.device], dict[str, float]]
"""(model, loader, device) -> {'val_metric': float, ...}  must include 'monitor'."""


class Trainer:
    """Configurable AMP trainer with early stopping + EMA + cosine schedule."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        step_fn: StepFn,
        eval_fn: MetricFn,
        *,
        epochs: int = 10,
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_ratio: float = 0.05,
        grad_clip: float | None = 1.0,
        amp: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        device: str | torch.device = "cuda",
        monitor: str = "val_metric",
        monitor_mode: str = "max",
        early_stop_patience: int = 5,
        checkpoint_dir: str | Path = "checkpoints",
        checkpoint_name: str = "best.pt",
        tracker: Tracker | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.step_fn = step_fn
        self.eval_fn = eval_fn

        self.epochs = epochs
        self.grad_clip = grad_clip
        self.device = torch.device(device if torch.cuda.is_available() or str(device) != "cuda" else "cpu")
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            self.device = torch.device(device)

        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device=self.device.type, enabled=self.amp)

        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )
        total_steps = max(1, epochs * len(train_loader))
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=int(total_steps * warmup_ratio),
        )

        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None

        self.monitor = monitor
        self.monitor_mode = monitor_mode
        self.early_stop_patience = early_stop_patience

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_name = checkpoint_name

        self.tracker = tracker
        self.state = TrainState(
            best_metric=(-math.inf if monitor_mode == "max" else math.inf),
            patience_left=early_stop_patience,
        )

        self.model.to(self.device)

    # ------------------------------------------------------------ epoch
    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        running: dict[str, float] = {"loss": 0.0}
        n = 0
        t0 = time.time()
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp):
                loss, metrics = self.step_fn(self.model, batch, self.device)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            if self.ema is not None:
                self.ema.update(self.model)

            self.state.global_step += 1
            bs = metrics.pop("_batch_size", 1)
            n += bs
            running["loss"] += float(loss.item()) * bs
            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + float(v) * bs

        dur = time.time() - t0
        avg = {k: v / max(n, 1) for k, v in running.items()}
        avg["train_time_s"] = dur
        avg["lr"] = self.optimizer.param_groups[0]["lr"]
        return avg

    # ------------------------------------------------------------ fit
    def fit(self) -> TrainState:
        for epoch in range(self.epochs):
            self.state.epoch = epoch
            train_metrics = self._train_epoch()

            # Validation on EMA weights if available
            original = None
            if self.ema is not None:
                original = self.ema.apply_to(self.model)
            try:
                val_metrics = self.eval_fn(self.model, self.val_loader, self.device)
            finally:
                if original is not None:
                    self.model.load_state_dict(original, strict=False)

            merged = {
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"val/{k}": v for k, v in val_metrics.items()},
            }
            self.state.history.append(merged)
            logger.info("epoch %d: %s", epoch, merged)
            if self.tracker is not None:
                self.tracker.log_metrics(merged, step=epoch)

            current = val_metrics.get(self.monitor, val_metrics.get("val_metric"))
            if current is None:
                raise RuntimeError(f"eval_fn must return '{self.monitor}'")
            improved = (
                current > self.state.best_metric
                if self.monitor_mode == "max"
                else current < self.state.best_metric
            )
            if improved:
                self.state.best_metric = current
                self.state.best_epoch = epoch
                self.state.patience_left = self.early_stop_patience
                self._save_checkpoint()
            else:
                self.state.patience_left -= 1
                if self.state.patience_left <= 0:
                    logger.info("Early stopping at epoch %d (best=%s)", epoch, self.state.best_metric)
                    break

        return self.state

    # ------------------------------------------------------------ io
    def _save_checkpoint(self) -> None:
        path = self.checkpoint_dir / self.checkpoint_name
        state_dict = (
            copy.deepcopy(self.ema.shadow)
            if self.ema is not None
            else {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        )
        torch.save(
            {"state_dict": state_dict, "epoch": self.state.epoch, "best_metric": self.state.best_metric}, path
        )
        if self.tracker is not None:
            self.tracker.log_artifact(str(path), name=self.checkpoint_name)
        logger.info("Saved checkpoint to %s (metric=%.6f)", path, self.state.best_metric)


# ---------------------------------------------------------------------------
def iter_batches(loader: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Small helper so code sites that want `for batch in iter_batches(...)` read well."""
    yield from loader

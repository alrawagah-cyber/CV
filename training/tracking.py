"""Experiment tracking abstraction with wandb + mlflow backends.

Selected via config:

    tracking:
      backend: wandb | mlflow | none
      project: "car-damage"
      run_name: "layer2-convnextv2-v1"
"""

from __future__ import annotations

from typing import Any


class Tracker:
    """No-op base class; real backends override these."""

    def __init__(self, project: str, run_name: str, config: dict[str, Any]):
        self.project = project
        self.run_name = run_name
        self.config = config

    def log_params(self, params: dict[str, Any]) -> None:  # noqa: ARG002
        pass

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:  # noqa: ARG002
        pass

    def log_artifact(self, path: str, name: str | None = None) -> None:  # noqa: ARG002
        pass

    def finish(self) -> None:
        pass


class WandbTracker(Tracker):
    def __init__(self, project: str, run_name: str, config: dict[str, Any]):
        super().__init__(project, run_name, config)
        import wandb

        self._wandb = wandb
        self._run = wandb.init(project=project, name=run_name, config=config, reinit=True)

    def log_params(self, params: dict[str, Any]) -> None:
        self._run.config.update(params, allow_val_change=True)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._wandb.log(metrics, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:
        art = self._wandb.Artifact(name or "model", type="model")
        art.add_file(path)
        self._run.log_artifact(art)

    def finish(self) -> None:
        self._run.finish()


class MLflowTracker(Tracker):
    def __init__(self, project: str, run_name: str, config: dict[str, Any]):
        super().__init__(project, run_name, config)
        import mlflow

        self._mlflow = mlflow
        mlflow.set_experiment(project)
        self._active = mlflow.start_run(run_name=run_name)
        mlflow.log_params({k: str(v) for k, v in config.items() if _is_scalar(v)})

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items() if _is_scalar(v)})

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str, name: str | None = None) -> None:  # noqa: ARG002
        self._mlflow.log_artifact(path)

    def finish(self) -> None:
        self._mlflow.end_run()


def _is_scalar(v: Any) -> bool:
    return isinstance(v, (int, float, str, bool))


def build_tracker(cfg: dict[str, Any] | None, run_name: str, full_config: dict[str, Any]) -> Tracker:
    cfg = cfg or {}
    backend = (cfg.get("backend") or "none").lower()
    project = cfg.get("project", "car-damage-pipeline")
    if backend == "wandb":
        return WandbTracker(project, run_name, full_config)
    if backend == "mlflow":
        return MLflowTracker(project, run_name, full_config)
    return Tracker(project, run_name, full_config)

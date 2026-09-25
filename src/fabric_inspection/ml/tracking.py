"""MLflow tracking on a local SQLite backend (never exposed to the network, #28)."""

import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

EXPERIMENT = "fabric-defects-carpet"


def code_version() -> str:
    """Current git commit, with a +dirty suffix when there are uncommitted changes."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}+dirty" if dirty else commit


def _flatten(prefix: str, value: Any, out: dict[str, float]) -> None:
    if isinstance(value, bool | int | float):
        out[prefix] = float(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif (
        isinstance(value, list | tuple)
        and len(value) == 2
        and all(isinstance(item, int | float) for item in value)
    ):
        out[f"{prefix}.low"], out[f"{prefix}.high"] = float(value[0]), float(value[1])


def flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    _flatten("", metrics, flat)
    return {key.replace(" ", "_"): value for key, value in flat.items()}


@contextmanager
def tracked_run(tracking_uri: str, name: str, params: Mapping[str, Any]) -> Iterator[str]:
    """Open an MLflow run and yield its id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params({key: str(value) for key, value in params.items()})
        yield str(run.info.run_id)


def log_metrics(metrics: Mapping[str, Any]) -> None:
    mlflow.log_metrics(flatten_metrics(metrics))


def log_artifacts(paths: list[Path]) -> None:
    for path in paths:
        if path.is_file():
            mlflow.log_artifact(str(path))

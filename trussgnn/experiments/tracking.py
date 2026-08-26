"""Resolve MLflow settings and record a Phase 4A connection check."""

from contextlib import redirect_stdout
from dataclasses import dataclass
from io import StringIO
import os
from pathlib import Path
from typing import Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment


DEFAULT_EXPERIMENT_NAME = "TrussGNN"


class TrackingConnectionError(RuntimeError):
    """The configured MLflow tracking destination could not be used."""


@dataclass(frozen=True)
class TrackingConfig:
    """Resolved MLflow destination and experiment."""

    tracking_uri: str
    experiment_name: str


@dataclass(frozen=True)
class ConnectionCheckResult:
    """Identifiers produced by a successful Phase 4A smoke run."""

    experiment_name: str
    experiment_id: str
    run_id: str


def _local_tracking_uri(local_directory: str | Path | None) -> str:
    directory = Path(local_directory or ".mlflow").resolve()
    database = directory / "mlflow.db"
    return f"sqlite:///{database.as_posix()}"


def _local_artifact_uri(tracking_uri: str) -> str | None:
    """Place artifacts beside a local SQLite database; servers choose their own."""

    parsed = urlsplit(tracking_uri)
    if parsed.scheme != "sqlite":
        return None
    database = Path(parsed.path).resolve()
    return (database.parent / "artifacts").as_uri()


def resolve_tracking_config(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    local_directory: str | Path | None = None,
) -> TrackingConfig:
    """Resolve explicit arguments, then environment values, then safe defaults."""

    environment = os.environ if environment is None else environment
    resolved_uri = tracking_uri if tracking_uri is not None else (
        environment.get("MLFLOW_TRACKING_URI") or _local_tracking_uri(local_directory)
    )
    resolved_experiment = experiment_name if experiment_name is not None else (
        environment.get("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT_NAME
    )

    if not resolved_uri.strip():
        raise ValueError("MLflow tracking URI cannot be empty")
    if not resolved_experiment.strip():
        raise ValueError("MLflow experiment name cannot be empty")

    return TrackingConfig(resolved_uri, resolved_experiment)


def redact_tracking_uri(uri: str) -> str:
    """Remove user information, query values, and fragments from a URI."""

    parsed = urlsplit(uri)
    if not parsed.scheme:
        return uri.split("?", 1)[0].split("#", 1)[0]

    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port is not None:
        host = f"{host}:{port}"
    if parsed.username is not None or parsed.password is not None:
        host = f"<redacted>@{host}"

    safe_query = "<redacted>" if parsed.query else ""
    safe_parts = SplitResult(parsed.scheme, host, parsed.path, safe_query, "")
    return urlunsplit(safe_parts)


def _configure_mlflow(config: TrackingConfig) -> MlflowClient:
    """Point MLflow at the resolved destination."""

    mlflow.set_tracking_uri(config.tracking_uri)
    return MlflowClient(tracking_uri=config.tracking_uri)


def _get_or_create_experiment(client: MlflowClient, config: TrackingConfig) -> Experiment:
    """Verify connectivity and return the configured experiment."""

    experiment = client.get_experiment_by_name(config.experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            config.experiment_name,
            artifact_location=_local_artifact_uri(config.tracking_uri),
        )
        experiment = client.get_experiment(experiment_id)
    mlflow.set_experiment(experiment_id=experiment.experiment_id)
    return experiment


def _log_connection_check(experiment_id: str, run_name: str) -> str:
    """Log and close one short Phase 4A run."""

    # MLflow prints raw server links on run closure. The CLI prints a sanitized
    # destination instead, so suppress those links here.
    with redirect_stdout(StringIO()):
        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
            mlflow.log_params({"phase": "4A", "check_type": "mlflow_connection"})
            mlflow.log_metric("connection_check", 1.0)
            mlflow.set_tags({"phase": "4A", "purpose": "tracking_smoke_test"})
            mlflow.log_dict(
                {"status": "success", "phase": "4A", "purpose": "tracking_smoke_test"},
                "connection_check.json",
            )
            return run.info.run_id


def run_connection_check(
    config: TrackingConfig,
    run_name: str = "phase-4a-connection-check",
) -> ConnectionCheckResult:
    """Create/select an experiment and record one short, finished smoke run."""

    try:
        client = _configure_mlflow(config)
        experiment = _get_or_create_experiment(client, config)
        run_id = _log_connection_check(experiment.experiment_id, run_name)
    except Exception:
        if mlflow.active_run() is not None:
            mlflow.end_run(status="FAILED")
        safe_uri = redact_tracking_uri(config.tracking_uri)
        raise TrackingConnectionError(
            f"Unable to use configured MLflow tracking destination: {safe_uri}"
        ) from None

    return ConnectionCheckResult(config.experiment_name, experiment.experiment_id, run_id)

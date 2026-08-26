"""MLflow experiment-tracking helpers."""

from typing import Any

from .tracking import (
    ConnectionCheckResult,
    TrackingConfig,
    TrackingConnectionError,
    configure_experiment,
    redact_tracking_uri,
    resolve_tracking_config,
    run_connection_check,
)

__all__ = [
    "ConnectionCheckResult",
    "TrackingConfig",
    "TrackingConnectionError",
    "configure_experiment",
    "redact_tracking_uri",
    "resolve_tracking_config",
    "run_connection_check",
    "ExperimentResult",
    "run_experiment",
]


def __getattr__(name: str) -> Any:
    """Load the experiment runner lazily so its module remains executable."""

    if name in {"ExperimentResult", "run_experiment"}:
        from .run_experiment import ExperimentResult, run_experiment

        exports = {"ExperimentResult": ExperimentResult, "run_experiment": run_experiment}
        return exports[name]
    raise AttributeError(name)

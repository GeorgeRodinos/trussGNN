"""Isolated verification of the Phase 4A MLflow tracking foundation."""

import json
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from trussgnn.experiments.check_mlflow import main
from trussgnn.experiments.tracking import (
    TrackingConnectionError,
    redact_tracking_uri,
    resolve_tracking_config,
    run_connection_check,
)


@pytest.fixture(autouse=True)
def no_active_mlflow_run():
    """Prevent MLflow active-run state from leaking into or out of a test."""

    if mlflow.active_run() is not None:
        mlflow.end_run()
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture
def tracking_config(tmp_path: Path):
    """Use an isolated SQLite tracking store for a test run."""

    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    return resolve_tracking_config(
        tracking_uri=tracking_uri,
        experiment_name="Phase4A-Test",
        environment={},
    )


def test_connection_check_creates_finished_verifiable_run_and_artifact(tracking_config) -> None:
    result = run_connection_check(tracking_config)
    client = MlflowClient(tracking_uri=tracking_config.tracking_uri)

    experiment = client.get_experiment_by_name(tracking_config.experiment_name)
    run = client.get_run(result.run_id)
    artifacts = client.list_artifacts(result.run_id)

    assert experiment is not None
    assert experiment.experiment_id == result.experiment_id
    assert run.info.status == "FINISHED"
    assert run.data.params == {"phase": "4A", "check_type": "mlflow_connection"}
    assert run.data.metrics["connection_check"] == pytest.approx(1.0)
    assert run.data.tags["phase"] == "4A"
    assert run.data.tags["purpose"] == "tracking_smoke_test"
    assert any(artifact.path == "connection_check.json" for artifact in artifacts)

    artifact_path = client.download_artifacts(result.run_id, "connection_check.json")
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    assert artifact == {
        "status": "success",
        "phase": "4A",
        "purpose": "tracking_smoke_test",
    }
    assert mlflow.active_run() is None


def test_explicit_configuration_overrides_environment(tmp_path) -> None:
    explicit_uri = f"sqlite:///{(tmp_path / 'explicit.db').as_posix()}"
    config = resolve_tracking_config(
        tracking_uri=explicit_uri,
        experiment_name="Explicit-Experiment",
        environment={
            "MLFLOW_TRACKING_URI": "http://environment.invalid:5000",
            "MLFLOW_EXPERIMENT_NAME": "Environment-Experiment",
        },
    )

    assert config.tracking_uri == explicit_uri
    assert config.experiment_name == "Explicit-Experiment"


def test_environment_and_safe_local_defaults_are_resolved(tmp_path) -> None:
    environment_config = resolve_tracking_config(
        environment={
            "MLFLOW_TRACKING_URI": "file:///tmp/environment-mlflow",
            "MLFLOW_EXPERIMENT_NAME": "Environment-Experiment",
        }
    )
    default_config = resolve_tracking_config(environment={}, local_directory=tmp_path / "local")

    assert environment_config.tracking_uri == "file:///tmp/environment-mlflow"
    assert environment_config.experiment_name == "Environment-Experiment"
    assert default_config.tracking_uri == f"sqlite:///{(tmp_path / 'local' / 'mlflow.db').as_posix()}"
    assert default_config.experiment_name == "TrussGNN"


def test_sensitive_uri_components_are_redacted() -> None:
    uri = "https://private-user:private-password@example.com:5000/mlflow?token=secret#private"
    displayed = redact_tracking_uri(uri)

    assert displayed == "https://<redacted>@example.com:5000/mlflow?<redacted>"
    for secret in ("private-user", "private-password", "token", "secret", "private"):
        assert secret not in displayed


def test_malformed_port_is_safely_omitted_during_redaction() -> None:
    uri = "http://private-user:private-password@127.0.0.1:notaport/path?token=secret"

    displayed = redact_tracking_uri(uri)
    message = f"Unable to connect to: {displayed}"

    assert displayed == "http://<redacted>@127.0.0.1/path?<redacted>"
    for secret in ("private-user", "private-password", "notaport", "token", "secret"):
        assert secret not in message


def test_invalid_explicit_endpoint_raises_without_fallback(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    config = resolve_tracking_config(
        tracking_uri="http://127.0.0.1:1?token=do-not-display",
        experiment_name="Unavailable",
        environment={},
    )

    with pytest.raises(TrackingConnectionError) as caught:
        run_connection_check(config)

    message = str(caught.value)
    assert "127.0.0.1:1" in message
    assert "do-not-display" not in message
    assert mlflow.active_run() is None


def test_cli_prints_experiment_and_run_id_without_secrets(tmp_path, capsys) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'cli.db').as_posix()}?timeout=10"
    main(
        [
            "--tracking-uri",
            tracking_uri,
            "--experiment-name",
            "CLI-Test",
            "--run-name",
            "cli-check",
        ]
    )

    output = capsys.readouterr().out
    assert "Experiment: CLI-Test" in output
    assert "Run ID:" in output
    assert "timeout=10" not in output
    assert mlflow.active_run() is None

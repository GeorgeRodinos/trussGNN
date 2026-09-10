"""End-to-end tests for the MLflow experiment runner."""

from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

import trussgnn.experiments.run_experiment as runner
from trussgnn.data import GenerationConfig, generate_dataset, save_dataset
from trussgnn.data.generation import SPLIT_NAMES
from trussgnn.experiments.run_experiment import main, parse_args, run_experiment
from trussgnn.experiments.tracking import resolve_tracking_config
from trussgnn.training import TrainingConfig


@pytest.fixture(autouse=True)
def isolated_mlflow(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    if mlflow.active_run() is not None:
        mlflow.end_run()
    yield
    if mlflow.active_run() is not None:
        mlflow.end_run()


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("experiment-data")
    counts = {name: 1 for name in SPLIT_NAMES}
    save_dataset(generate_dataset(GenerationConfig(seed=31, split_counts=counts)), directory)
    return directory


@pytest.fixture
def tracking(tmp_path):
    return resolve_tracking_config(
        f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}",
        "Experiment-Test",
        environment={},
    )


def artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> set[str]:
    paths: set[str] = set()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            paths.update(artifact_paths(client, run_id, item.path))
        else:
            paths.add(item.path)
    return paths


@pytest.mark.parametrize("model_name", ["zero", "mlp", "gnn"])
def test_models_share_evaluation_contract_and_log_complete_runs(
    model_name, tiny_dataset, tracking
) -> None:
    result = run_experiment(
        tiny_dataset,
        model_name,
        tracking,
        TrainingConfig(max_epochs=1, patience=1, seed=7),
        batch_size=1,
        hidden_dim=4,
        layer_count=1,
    )
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    logged = client.get_run(result.run_id)
    artifacts = artifact_paths(client, result.run_id)

    assert result.model_name == model_name
    assert set(result.final_metrics) == {
        "validation", "iid_test", "geometry_ood", "topology_size_ood"
    }
    assert all(
        set(metrics) == {
            "loss", "mae_m", "mae_mm", "rmse_m", "rmse_mm",
            "mean_graph_relative_l2", "physics_loss", "mean_equilibrium_residual",
        }
        for metrics in result.final_metrics.values()
    )
    assert logged.info.status == "FINISHED"
    assert logged.info.run_name == f"{model_name}-experiment"
    assert logged.data.params["model"] == model_name
    for split in result.final_metrics:
        assert f"{split}/loss" in logged.data.metrics
        assert f"{split}/mae_mm" in logged.data.metrics
        assert f"{split}/rmse_mm" in logged.data.metrics
        assert f"{split}/mean_graph_relative_l2" in logged.data.metrics
        assert f"{split}/mean_equilibrium_residual" in logged.data.metrics
    assert {
        "resolved_config.json", "metadata.json", "normalization.json",
    } <= artifacts
    if model_name == "zero":
        assert result.best_epoch is None
        assert "best_checkpoint.pt" not in artifacts
    else:
        assert result.best_epoch == 1
        assert "best_checkpoint.pt" in artifacts
        assert client.get_metric_history(result.run_id, "train/loss")
        assert client.get_metric_history(result.run_id, "train/data_loss")
        assert client.get_metric_history(result.run_id, "train/physics_loss")
        assert client.get_metric_history(result.run_id, "train/total_loss")
        assert client.get_metric_history(result.run_id, "validation/data_loss")
        assert client.get_metric_history(result.run_id, "validation/physics_loss")
        assert client.get_metric_history(result.run_id, "validation/total_loss")
        assert client.get_metric_history(result.run_id, "validation/rmse_mm")
    assert logged.data.params["physics_loss_weight"] == "0.0"
    assert logged.data.params["physics_epsilon"] == "1e-12"
    config_path = client.download_artifacts(result.run_id, "resolved_config.json")
    logged_text = str(logged.data.params) + Path(config_path).read_text(encoding="utf-8")
    assert tracking.tracking_uri not in logged_text
    assert "tracking_uri" not in logged_text.lower()
    assert "password" not in logged_text.lower()
    assert "token" not in logged_text.lower()
    assert mlflow.active_run() is None


def test_fixed_seed_reproduces_final_metrics(tiny_dataset, tracking) -> None:
    kwargs = dict(
        dataset_dir=tiny_dataset,
        model_name="mlp",
        tracking=tracking,
        training=TrainingConfig(max_epochs=2, patience=2, seed=19),
        batch_size=1,
        hidden_dim=5,
        layer_count=1,
    )
    first = run_experiment(**kwargs)
    second = run_experiment(**kwargs)
    assert first.final_metrics == second.final_metrics


def test_learned_model_fits_before_final_split_evaluation(
    monkeypatch, tiny_dataset, tracking
) -> None:
    events = []
    real_fit = runner.fit_model
    real_evaluate = runner.evaluate_model

    def record_fit(*args, **kwargs):
        events.append("fit")
        return real_fit(*args, **kwargs)

    def record_evaluate(*args, **kwargs):
        events.append("evaluate")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(runner, "fit_model", record_fit)
    monkeypatch.setattr(runner, "evaluate_model", record_evaluate)
    run_experiment(
        tiny_dataset, "mlp", tracking,
        TrainingConfig(max_epochs=1, patience=1), batch_size=1,
        hidden_dim=4, layer_count=1,
    )
    assert events[0] == "fit"
    assert events[1:] == ["evaluate"] * 4


def test_failure_closes_active_run(monkeypatch, tiny_dataset, tracking) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("deliberate evaluation failure")

    monkeypatch.setattr(runner, "evaluate_model", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        run_experiment(
            tiny_dataset, "zero", tracking, TrainingConfig(max_epochs=1), batch_size=1
        )
    assert mlflow.active_run() is None


def test_cli_completes_against_temporary_store(tiny_dataset, tracking, capsys) -> None:
    main(
        [
            "--dataset-dir", str(tiny_dataset),
            "--model", "zero",
            "--tracking-uri", tracking.tracking_uri,
            "--experiment-name", tracking.experiment_name,
            "--max-epochs", "1",
            "--batch-size", "1",
        ]
    )
    output = capsys.readouterr().out
    assert "Experiment: Experiment-Test" in output
    assert "Run ID:" in output
    assert tracking.tracking_uri not in output
    assert mlflow.active_run() is None


def test_cli_physics_options_preserve_defaults_and_parse_explicit_values() -> None:
    defaults = parse_args(["--model", "gnn"])
    explicit = parse_args(
        [
            "--model",
            "gnn",
            "--physics-loss-weight",
            "0.25",
            "--physics-epsilon",
            "1e-9",
        ]
    )

    assert defaults.physics_loss_weight == 0.0
    assert defaults.physics_epsilon == 1e-12
    assert explicit.physics_loss_weight == 0.25
    assert explicit.physics_epsilon == 1e-9


@pytest.mark.parametrize(
    "arguments",
    [
        ["--model", "gnn", "--physics-loss-weight", "-1"],
        ["--model", "gnn", "--physics-loss-weight", "nan"],
        ["--model", "gnn", "--physics-loss-weight", "inf"],
        ["--model", "gnn", "--physics-epsilon", "0"],
        ["--model", "gnn", "--physics-epsilon", "nan"],
        ["--model", "gnn", "--physics-epsilon", "inf"],
    ],
)
def test_cli_rejects_invalid_physics_values(arguments) -> None:
    with pytest.raises(ValueError):
        main(arguments)


@pytest.mark.parametrize("model_name", ["zero", "mlp"])
def test_positive_physics_weight_is_rejected_for_non_gnn_models(
    model_name, tiny_dataset, tracking
) -> None:
    with pytest.raises(ValueError, match="only for the GNN"):
        run_experiment(
            tiny_dataset,
            model_name,
            tracking,
            TrainingConfig(max_epochs=1, physics_loss_weight=1e-12),
            batch_size=1,
        )


def test_positive_physics_gnn_run_logs_new_fields(tiny_dataset, tracking) -> None:
    result = run_experiment(
        tiny_dataset,
        "gnn",
        tracking,
        TrainingConfig(
            max_epochs=1,
            patience=1,
            seed=7,
            physics_loss_weight=1e-12,
            physics_epsilon=1e-9,
        ),
        batch_size=1,
        hidden_dim=4,
        layer_count=1,
    )
    client = MlflowClient(tracking_uri=tracking.tracking_uri)
    logged = client.get_run(result.run_id)

    assert logged.info.status == "FINISHED"
    assert logged.data.params["physics_loss_weight"] == "1e-12"
    assert logged.data.params["physics_epsilon"] == "1e-09"
    for key in (
        "train/data_loss",
        "train/physics_loss",
        "train/total_loss",
        "validation/data_loss",
        "validation/physics_loss",
        "validation/total_loss",
    ):
        assert client.get_metric_history(result.run_id, key)
    for split in ("validation", "iid_test", "geometry_ood", "topology_size_ood"):
        assert f"{split}/mean_equilibrium_residual" in logged.data.metrics
    assert mlflow.active_run() is None

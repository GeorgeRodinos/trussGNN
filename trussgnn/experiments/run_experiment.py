"""Run one tracked baseline, MLP, or GNN displacement experiment."""

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
import json
from io import StringIO
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

import mlflow
import torch
import torch_geometric
from torch import nn
from torch_geometric.loader import DataLoader

from trussgnn.data import (
    LoadedDataset,
    build_dataset_manifest,
    create_data_loaders,
    enforce_boundary_conditions,
    inverse_targets,
    load_dataset,
)
from trussgnn.models import EdgeAwareGNN, NodeMLP, ZeroDisplacementBaseline
from trussgnn.training import (
    TrainingConfig,
    evaluate_model,
    fit_model,
    seed_everything,
)

from .tracking import TrackingConfig, configure_experiment, resolve_tracking_config


EVALUATION_SPLITS = (
    "validation",
    "iid_test",
    "geometry_ood",
    "topology_size_ood",
)


@dataclass(frozen=True)
class ExperimentResult:
    """Identity and final split metrics from one completed MLflow run."""

    run_id: str
    experiment_id: str
    model_name: str
    best_epoch: int | None
    final_metrics: dict[str, dict[str, float]]


def _build_model(
    name: str,
    dataset: LoadedDataset,
    hidden_dim: int,
    layer_count: int,
    dropout: float,
) -> tuple[nn.Module, str]:
    if name == "zero":
        stats = dataset.normalization
        return ZeroDisplacementBaseline(stats.target_mean, stats.target_std), "baseline"
    if name == "mlp":
        return NodeMLP(hidden_dim, layer_count, dropout), "node_mlp"
    if name == "gnn":
        return EdgeAwareGNN(hidden_dim, layer_count, dropout), "edge_aware_gnn"
    raise ValueError("model_name must be one of: zero, mlp, gnn")


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parameters(
    model_name: str,
    model: nn.Module,
    model_family: str,
    training: TrainingConfig,
    batch_size: int,
    num_workers: int,
    hidden_dim: int,
    layer_count: int,
    dropout: float,
    dataset: LoadedDataset,
) -> dict[str, str | int | float]:
    values: dict[str, str | int | float] = {
        "model": model_name,
        "model_family": model_family,
        "architecture": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": training.seed,
        "max_epochs": training.max_epochs,
        "patience": training.patience,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "optimizer": "none" if model_name == "zero" else "Adam",
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "hidden_dim": hidden_dim,
        "layer_count": layer_count,
        "dropout": dropout,
        "dataset_seed": dataset.metadata["seed"],
        "normalization_source": dataset.normalization.source_split,
        "device": training.device,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "pyg_version": torch_geometric.__version__,
        "mlflow_version": mlflow.__version__,
    }
    for split, count in dataset.metadata["split_counts"].items():
        values[f"split_count_{split}"] = count
    return values


def _prediction_example(
    model: nn.Module,
    graph,
    dataset: LoadedDataset,
    device: torch.device,
) -> dict[str, object]:
    batch = next(iter(DataLoader([graph], batch_size=1))).to(device)
    model.eval()
    with torch.no_grad():
        normalized = model(batch)
        prediction = enforce_boundary_conditions(
            inverse_targets(normalized, dataset.normalization), batch.free_dof_mask
        )
        target = inverse_targets(batch.y, dataset.normalization)
    return {
        "graph_id": int(graph.graph_id),
        "prediction_m": prediction.detach().cpu().tolist(),
        "target_m": target.detach().cpu().tolist(),
        "free_dof_mask": batch.free_dof_mask.cpu().tolist(),
    }


def _resolved_run_configuration(
    model_name: str,
    training: TrainingConfig,
    batch_size: int,
    num_workers: int,
    hidden_dim: int,
    layer_count: int,
    dropout: float,
) -> dict[str, object]:
    """Return the reproducibility settings, deliberately excluding tracking details."""

    return {
        "model": model_name,
        "training": asdict(training),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "hidden_dim": hidden_dim,
        "layer_count": layer_count,
        "dropout": dropout,
    }


def run_experiment(
    dataset_dir: str | Path,
    model_name: str,
    tracking: TrackingConfig,
    training: TrainingConfig | None = None,
    *,
    batch_size: int = 32,
    num_workers: int = 0,
    hidden_dim: int = 64,
    layer_count: int = 2,
    dropout: float = 0.0,
    run_name: str | None = None,
) -> ExperimentResult:
    """Run, evaluate, and log one accepted model without leaking tracking details."""

    training = training or TrainingConfig()
    experiment = configure_experiment(tracking)
    dataset_dir = Path(dataset_dir)
    dataset = load_dataset(dataset_dir)
    manifest = build_dataset_manifest(dataset_dir)
    loaders = create_data_loaders(dataset.splits, batch_size, training.seed, num_workers)
    seed_everything(training.seed)
    model, model_family = _build_model(
        model_name, dataset, hidden_dim, layer_count, dropout
    )
    device = torch.device(training.device)
    model.to(device)

    best_epoch: int | None = None
    final_metrics: dict[str, dict[str, float]] = {}
    with TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        checkpoint_path = temporary / "best_checkpoint.pt"
        try:
            with redirect_stdout(StringIO()):
                with mlflow.start_run(
                    experiment_id=experiment.experiment_id,
                    run_name=run_name or f"{model_name}-phase-4d3",
                ) as run:
                    mlflow.log_params(
                        _parameters(
                            model_name, model, model_family, training, batch_size,
                            num_workers, hidden_dim, layer_count, dropout, dataset,
                        )
                    )
                    tags = {
                        "phase": "4",
                        "task": "node_displacement_regression",
                        "model_family": model_family,
                    }
                    commit = _git_commit()
                    if commit:
                        tags["git_commit"] = commit
                    mlflow.set_tags(tags)

                    training_result = None
                    if model_name != "zero":
                        training_result = fit_model(
                            model, loaders["train"], loaders["validation"],
                            dataset.normalization, training, checkpoint_path,
                        )
                        best_epoch = training_result.best_epoch
                        for record in training_result.history:
                            step = int(record["epoch"])
                            mlflow.log_metrics(
                                {
                                    "train/loss": float(record["train_loss"]),
                                    "validation/loss": float(record["loss"]),
                                    "validation/mae_mm": float(record["mae_mm"]),
                                    "validation/rmse_mm": float(record["rmse_mm"]),
                                },
                                step=step,
                            )

                    for split in EVALUATION_SPLITS:
                        metrics = evaluate_model(
                            model, loaders[split], device, dataset.normalization
                        )
                        final_metrics[split] = metrics
                        mlflow.log_metrics(
                            {f"{split}/{name}": value for name, value in metrics.items()},
                            step=(
                                training_result.epochs_completed + 1
                                if training_result is not None
                                else 0
                            ),
                        )

                    mlflow.log_dict(
                        _resolved_run_configuration(
                            model_name, training, batch_size, num_workers,
                            hidden_dim, layer_count, dropout,
                        ),
                        "resolved_config.json",
                    )
                    mlflow.log_dict(manifest, "dataset_manifest.json")
                    mlflow.log_artifact(dataset_dir / "metadata.json")
                    mlflow.log_artifact(dataset_dir / "normalization.json")
                    if training_result is not None:
                        history_path = temporary / "training_history.json"
                        history_path.write_text(
                            json.dumps(training_result.history, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        mlflow.log_artifact(history_path)
                        mlflow.log_artifact(checkpoint_path)
                    mlflow.log_dict(
                        _prediction_example(
                            model, dataset.splits["validation"][0], dataset, device
                        ),
                        "prediction_example.json",
                    )
                    run_id = run.info.run_id
        except Exception:
            if mlflow.active_run() is not None:
                mlflow.end_run(status="FAILED")
            raise

    return ExperimentResult(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        model_name=model_name,
        best_epoch=best_epoch,
        final_metrics=final_metrics,
    )


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--model", choices=("zero", "mlp", "gnn"), required=True)
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-name")
    parser.add_argument("--run-name")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    tracking = resolve_tracking_config(args.tracking_uri, args.experiment_name)
    training = TrainingConfig(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        seed=args.seed,
        device=args.device,
    )
    result = run_experiment(
        args.dataset_dir,
        args.model,
        tracking,
        training,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_dim=args.hidden_dim,
        layer_count=args.layers,
        dropout=args.dropout,
        run_name=args.run_name,
    )
    print(f"Experiment: {tracking.experiment_name}")
    print(f"Run ID: {result.run_id}")


if __name__ == "__main__":
    main()

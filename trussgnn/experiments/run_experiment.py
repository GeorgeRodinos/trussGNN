"""Run one tracked baseline, MLP, or GNN displacement experiment."""

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import mlflow
import torch
from torch import nn

from trussgnn.data import LoadedDataset, create_data_loaders, load_dataset
from trussgnn.models import EdgeAwareGNN, NodeMLP, ZeroDisplacementBaseline
from trussgnn.training import TrainingConfig, evaluate_model, fit_model, seed_everything

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


def build_model(
    name: str,
    dataset: LoadedDataset,
    hidden_dim: int,
    layer_count: int,
    dropout: float,
) -> nn.Module:
    """Construct one of the three accepted displacement models."""

    if name == "zero":
        stats = dataset.normalization
        return ZeroDisplacementBaseline(stats.target_mean, stats.target_std)
    if name == "mlp":
        return NodeMLP(hidden_dim, layer_count, dropout)
    if name == "gnn":
        return EdgeAwareGNN(hidden_dim, layer_count, dropout)
    raise ValueError("model_name must be one of: zero, mlp, gnn")


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
    """Load data, train one model when needed, evaluate it, and log the run."""

    training = training or TrainingConfig()
    dataset_dir = Path(dataset_dir)
    dataset = load_dataset(dataset_dir)
    loaders = create_data_loaders(dataset.splits, batch_size, training.seed, num_workers)

    seed_everything(training.seed)
    model = build_model(model_name, dataset, hidden_dim, layer_count, dropout)
    device = torch.device(training.device)
    model.to(device)

    experiment = configure_experiment(tracking)
    parameters = {
        "model": model_name,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "seed": training.seed,
        "device": training.device,
        "batch_size": batch_size,
        "max_epochs": training.max_epochs,
        "patience": training.patience,
        "learning_rate": training.learning_rate,
        "weight_decay": training.weight_decay,
        "hidden_dim": hidden_dim,
        "layer_count": layer_count,
        "dropout": dropout,
        "dataset_seed": dataset.metadata["seed"],
    }

    best_epoch = None
    final_metrics: dict[str, dict[str, float]] = {}
    with TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "best_checkpoint.pt"
        with redirect_stdout(StringIO()):
            with mlflow.start_run(
                experiment_id=experiment.experiment_id,
                run_name=run_name or f"{model_name}-phase-4",
            ) as run:
                mlflow.log_params(parameters)

                training_result = None
                if model_name != "zero":
                    training_result = fit_model(
                        model,
                        loaders["train"],
                        loaders["validation"],
                        dataset.normalization,
                        training,
                        checkpoint_path,
                    )
                    best_epoch = training_result.best_epoch
                    for epoch in training_result.history:
                        mlflow.log_metrics(
                            {
                                "train/loss": epoch["train_loss"],
                                "validation/loss": epoch["loss"],
                                "validation/rmse_mm": epoch["rmse_mm"],
                            },
                            step=int(epoch["epoch"]),
                        )

                final_step = (
                    training_result.epochs_completed + 1 if training_result else 0
                )
                for split in EVALUATION_SPLITS:
                    metrics = evaluate_model(
                        model, loaders[split], device, dataset.normalization
                    )
                    final_metrics[split] = metrics
                    mlflow.log_metrics(
                        {
                            f"{split}/loss": metrics["loss"],
                            f"{split}/mae_mm": metrics["mae_mm"],
                            f"{split}/rmse_mm": metrics["rmse_mm"],
                            f"{split}/mean_graph_relative_l2": metrics[
                                "mean_graph_relative_l2"
                            ],
                        },
                        step=final_step,
                    )

                mlflow.log_dict(
                    {"parameters": parameters, "training": asdict(training)},
                    "resolved_config.json",
                )
                mlflow.log_artifact(dataset_dir / "metadata.json")
                mlflow.log_artifact(dataset_dir / "normalization.json")
                if training_result is not None:
                    mlflow.log_artifact(checkpoint_path)
                run_id = run.info.run_id

    return ExperimentResult(
        run_id,
        experiment.experiment_id,
        model_name,
        best_epoch,
        final_metrics,
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

"""Deterministic training, evaluation, and early stopping."""

import copy
import random
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from trussgnn.data.loading import NormalizationStats

from .config import TrainingConfig
from .metrics import masked_mse, physical_metrics
from .physics import equilibrium_residual_loss, relative_equilibrium_residuals


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed fit and its restored best checkpoint."""

    history: list[dict[str, float | int]]
    best_epoch: int
    best_validation_rmse_m: float
    epochs_completed: int
    stopped_early: bool
    checkpoint_path: Path


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch and request deterministic operations."""

    if seed < 0:
        raise ValueError("seed cannot be negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _train_epoch(
    model: nn.Module,
    training_loader: Iterable[Data],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    normalization: NormalizationStats | None,
    config: TrainingConfig | None,
) -> dict[str, float]:
    """Train once and aggregate data by DOFs and physics by graphs.

    Each batch optimizes the sum of its two batch means. The epoch summary
    recombines the split-wide DOF-weighted data loss and graph-weighted physics
    loss, instead of averaging batch objectives of unequal sizes.
    """

    physics_weight = config.physics_loss_weight if config is not None else 0.0
    physics_epsilon = config.physics_epsilon if config is not None else 1e-12
    if physics_weight > 0 and normalization is None:
        raise ValueError("Positive physics_loss_weight requires normalization statistics")

    model.train()
    squared_error_sum = 0.0
    free_dof_count = 0
    physics_loss_sum = 0.0
    graph_count = 0
    for batch in training_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        prediction = model(batch)
        data_loss = masked_mse(prediction, batch.y, batch.free_dof_mask)
        if physics_weight > 0:
            physics_loss = equilibrium_residual_loss(
                prediction,
                batch,
                normalization,
                epsilon=physics_epsilon,
            )
            total_loss = data_loss + physics_weight * physics_loss
        else:
            physics_loss = None
            total_loss = data_loss
        total_loss.backward()
        optimizer.step()

        count = int(batch.free_dof_mask.sum().item())
        squared_error_sum += float(data_loss.detach().item()) * count
        free_dof_count += count
        if physics_loss is not None:
            batch_graphs = getattr(batch, "num_graphs", None)
            if batch_graphs is None:
                batch_graphs = 1
            physics_loss_sum += float(physics_loss.detach().item()) * batch_graphs
            graph_count += batch_graphs

    if free_dof_count == 0:
        raise ValueError("Training loader is empty or contains no free DOFs")
    data_epoch_loss = squared_error_sum / free_dof_count
    physics_epoch_loss = physics_loss_sum / graph_count if graph_count else 0.0
    return {
        "data_loss": data_epoch_loss,
        "physics_loss": physics_epoch_loss,
        "total_loss": data_epoch_loss + physics_weight * physics_epoch_loss,
    }


def train_one_epoch(
    model: nn.Module,
    training_loader: Iterable[Data],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
    normalization: NormalizationStats | None = None,
    config: TrainingConfig | None = None,
) -> float:
    """Train once and return the aggregated total loss.

    Omitting ``config`` preserves the original supervised-only path and return value.
    """

    return _train_epoch(
        model, training_loader, optimizer, device, normalization, config
    )["total_loss"]


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[Data],
    device: str | torch.device,
    normalization: NormalizationStats,
    physics_epsilon: float = 1e-12,
) -> dict[str, float]:
    """Evaluate one complete split without weighting batches equally."""

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    memberships: list[torch.Tensor] = []
    equilibrium_residuals: list[torch.Tensor] = []
    equilibrium_available: bool | None = None
    graph_offset = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            prediction = model(batch)
            has_physics_fields = all(
                getattr(batch, name, None) is not None
                for name in ("pos", "edge_index", "edge_attr")
            )
            if equilibrium_available is None:
                equilibrium_available = has_physics_fields
            elif equilibrium_available != has_physics_fields:
                raise ValueError("Evaluation batches have inconsistent physics fields")
            if has_physics_fields:
                equilibrium_residuals.append(
                    relative_equilibrium_residuals(
                        prediction,
                        batch,
                        normalization,
                        epsilon=physics_epsilon,
                    )
                )
            predictions.append(prediction)
            targets.append(batch.y)
            masks.append(batch.free_dof_mask)

            membership = getattr(batch, "batch", None)
            if membership is None:
                membership = torch.zeros(batch.x.shape[0], dtype=torch.long, device=device)
            memberships.append(membership + graph_offset)
            graph_offset += int(membership.max().item()) + 1

    if not predictions:
        raise ValueError("Evaluation loader is empty")

    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    combined = Data(batch=torch.cat(memberships))
    result = physical_metrics(prediction, target, mask, combined, normalization)
    result["loss"] = float(masked_mse(prediction, target, mask).item())
    if equilibrium_residuals:
        residuals = torch.cat(equilibrium_residuals)
        result["physics_loss"] = float(residuals.square().mean().item())
        result["mean_equilibrium_residual"] = float(residuals.mean().item())
    return {"loss": result.pop("loss"), **result}


def fit_model(
    model: nn.Module,
    training_loader: Iterable[Data],
    validation_loader: Iterable[Data],
    normalization: NormalizationStats,
    config: TrainingConfig,
    checkpoint_path: str | Path,
) -> TrainingResult:
    """Fit with Adam, stop on validation RMSE, and restore the best state."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("fit_model requires a model with trainable parameters")

    seed_everything(config.seed)
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(
        parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, config.max_epochs + 1):
        training_losses = _train_epoch(
            model,
            training_loader,
            optimizer,
            device,
            normalization,
            config,
        )
        validation = evaluate_model(
            model,
            validation_loader,
            device,
            normalization,
            config.physics_epsilon,
        )
        validation_physics_loss = validation.get("physics_loss", 0.0)
        if config.physics_loss_weight > 0 and "physics_loss" not in validation:
            raise ValueError(
                "Positive physics_loss_weight requires validation physics fields"
            )
        validation_total_loss = validation["loss"] + (
            config.physics_loss_weight * validation_physics_loss
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": training_losses["data_loss"],
                "train_data_loss": training_losses["data_loss"],
                "train_physics_loss": training_losses["physics_loss"],
                "train_total_loss": training_losses["total_loss"],
                "validation_data_loss": validation["loss"],
                "validation_physics_loss": validation_physics_loss,
                "validation_total_loss": validation_total_loss,
                **validation,
            }
        )

        if validation["rmse_m"] < best_rmse - config.min_delta:
            best_rmse = validation["rmse_m"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "epoch": epoch,
                    "best_validation_rmse_m": best_rmse,
                    "training_config": asdict(config),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                stopped_early = True
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return TrainingResult(
        history=history,
        best_epoch=best_epoch,
        best_validation_rmse_m=best_rmse,
        epochs_completed=len(history),
        stopped_early=stopped_early,
        checkpoint_path=checkpoint_path,
    )

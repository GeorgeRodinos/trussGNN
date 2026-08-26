"""Deterministic training, evaluation, and early stopping."""

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

from trussgnn.data.loading import NormalizationStats

from .config import TrainingConfig
from .metrics import masked_mse, physical_metrics


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


def train_one_epoch(
    model: nn.Module,
    training_loader: Iterable[Data],
    optimizer: torch.optim.Optimizer,
    device: str | torch.device,
) -> float:
    """Train once and return MSE weighted by the number of free DOFs."""

    model.train()
    squared_error_sum = 0.0
    free_dof_count = 0
    for batch in training_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        prediction = model(batch)
        loss = masked_mse(prediction, batch.y, batch.free_dof_mask)
        loss.backward()
        optimizer.step()

        count = int(batch.free_dof_mask.sum().item())
        squared_error_sum += float(loss.detach().item()) * count
        free_dof_count += count

    if free_dof_count == 0:
        raise ValueError("Training loader is empty or contains no free DOFs")
    return squared_error_sum / free_dof_count


def evaluate_model(
    model: nn.Module,
    data_loader: Iterable[Data],
    device: str | torch.device,
    normalization: NormalizationStats,
) -> dict[str, float]:
    """Evaluate one complete split without weighting batches equally."""

    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    memberships: list[torch.Tensor] = []
    graph_offset = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            prediction = model(batch)
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
        train_loss = train_one_epoch(model, training_loader, optimizer, device)
        validation = evaluate_model(model, validation_loader, device, normalization)
        history.append(
            {"epoch": epoch, "train_loss": train_loss, **validation}
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

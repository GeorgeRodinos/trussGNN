"""Masked optimization loss and physical-unit displacement metrics."""

import math

import torch
from torch_geometric.data import Data

from trussgnn.data.loading import (
    NormalizationStats,
    enforce_boundary_conditions,
    inverse_targets,
)


def _validate_inputs(
    prediction: torch.Tensor,
    target: torch.Tensor,
    free_dof_mask: torch.Tensor,
) -> None:
    if prediction.shape != target.shape or prediction.shape != free_dof_mask.shape:
        raise ValueError("prediction, target, and free_dof_mask must have the same shape")
    if free_dof_mask.dtype != torch.bool:
        raise ValueError("free_dof_mask must have dtype torch.bool")
    if not torch.any(free_dof_mask):
        raise ValueError("At least one free degree of freedom is required")


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    free_dof_mask: torch.Tensor,
) -> torch.Tensor:
    """Return normalized-space mean squared error over free DOFs only."""

    _validate_inputs(prediction, target, free_dof_mask)
    error = prediction[free_dof_mask] - target[free_dof_mask]
    return error.square().mean()


def physical_metrics(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    free_dof_mask: torch.Tensor,
    batch: Data,
    normalization: NormalizationStats,
    relative_epsilon: float = 1e-12,
) -> dict[str, float]:
    """Return free-DOF physical errors and mean per-graph relative L2."""

    _validate_inputs(normalized_prediction, normalized_target, free_dof_mask)
    if relative_epsilon <= 0:
        raise ValueError("relative_epsilon must be positive")

    prediction_m = inverse_targets(normalized_prediction, normalization)
    target_m = inverse_targets(normalized_target, normalization)
    prediction_m = enforce_boundary_conditions(prediction_m, free_dof_mask)

    free_error = prediction_m[free_dof_mask] - target_m[free_dof_mask]
    mae_m = free_error.abs().mean()
    rmse_m = free_error.square().mean().sqrt()

    graph_membership = getattr(batch, "batch", None)
    if graph_membership is None:
        graph_membership = torch.zeros(
            normalized_prediction.shape[0],
            dtype=torch.long,
            device=normalized_prediction.device,
        )

    relative_errors = []
    for graph_id in torch.unique(graph_membership, sorted=True):
        graph_mask = graph_membership == graph_id
        graph_free_dofs = free_dof_mask & graph_mask.unsqueeze(1)
        if not torch.any(graph_free_dofs):
            raise ValueError("Every graph must contain at least one free degree of freedom")
        graph_error = prediction_m[graph_free_dofs] - target_m[graph_free_dofs]
        denominator = torch.linalg.vector_norm(target_m[graph_free_dofs]).clamp_min(
            relative_epsilon
        )
        relative_errors.append(torch.linalg.vector_norm(graph_error) / denominator)

    mean_relative_l2 = torch.stack(relative_errors).mean()
    result = {
        "mae_m": float(mae_m.item()),
        "mae_mm": float((mae_m * 1_000).item()),
        "rmse_m": float(rmse_m.item()),
        "rmse_mm": float((rmse_m * 1_000).item()),
        "mean_graph_relative_l2": float(mean_relative_l2.item()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Physical metrics must be finite")
    return result

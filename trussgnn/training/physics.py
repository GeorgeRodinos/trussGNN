"""Differentiable free-DOF equilibrium residuals for truss predictions."""

import math

import torch
from torch_geometric.data import Data

from trussgnn.data.loading import (
    NormalizationStats,
    enforce_boundary_conditions,
    inverse_targets,
)


def _validate_epsilon(epsilon: float) -> None:
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise TypeError("epsilon must be a positive finite number in newtons")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a positive finite number in newtons")


def _validate_graph(
    normalized_prediction: torch.Tensor,
    batch: Data,
) -> tuple[torch.Tensor, int]:
    if not isinstance(normalized_prediction, torch.Tensor):
        raise TypeError("normalized_prediction must be a PyTorch tensor")
    if normalized_prediction.ndim != 2 or normalized_prediction.shape[1] != 2:
        raise ValueError("normalized_prediction must have shape [num_nodes, 2]")
    if normalized_prediction.shape[0] < 1:
        raise ValueError("At least one node is required")
    if not normalized_prediction.is_floating_point():
        raise ValueError("normalized_prediction must use a floating-point dtype")

    num_nodes = normalized_prediction.shape[0]
    required = {
        "batch.pos": (getattr(batch, "pos", None), (num_nodes, 2)),
        "batch.free_dof_mask": (
            getattr(batch, "free_dof_mask", None),
            (num_nodes, 2),
        ),
    }
    for name, (tensor, shape) in required.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")

    node_features = getattr(batch, "x", None)
    if (
        not isinstance(node_features, torch.Tensor)
        or node_features.ndim != 2
        or node_features.shape[0] != num_nodes
        or node_features.shape[1] < 4
    ):
        raise ValueError("batch.x must have shape [num_nodes, at least 4]")

    edge_index = getattr(batch, "edge_index", None)
    if (
        not isinstance(edge_index, torch.Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
    ):
        raise ValueError("batch.edge_index must be a torch.long tensor with shape [2, num_edges]")

    edge_attr = getattr(batch, "edge_attr", None)
    num_edges = edge_index.shape[1]
    if (
        not isinstance(edge_attr, torch.Tensor)
        or edge_attr.ndim != 2
        or edge_attr.shape[0] != num_edges
        or edge_attr.shape[1] < 5
    ):
        raise ValueError("batch.edge_attr must have shape [num_edges, at least 5]")

    floating_tensors = {
        "batch.x": node_features,
        "batch.pos": batch.pos,
        "batch.edge_attr": edge_attr,
    }
    device_tensors = {
        **floating_tensors,
        "batch.edge_index": edge_index,
        "batch.free_dof_mask": batch.free_dof_mask,
    }
    for name, tensor in floating_tensors.items():
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must use a floating-point dtype")
        if tensor.dtype != normalized_prediction.dtype:
            raise ValueError(f"{name} must use the prediction dtype")
    for name, tensor in device_tensors.items():
        if tensor.device != normalized_prediction.device:
            raise ValueError(f"{name} must be on the prediction device")

    if batch.free_dof_mask.dtype != torch.bool:
        raise ValueError("batch.free_dof_mask must have dtype torch.bool")
    if num_edges and (torch.any(edge_index < 0) or torch.any(edge_index >= num_nodes)):
        raise ValueError("batch.edge_index contains an invalid node index")

    membership = getattr(batch, "batch", None)
    if membership is None:
        membership = torch.zeros(
            num_nodes, dtype=torch.long, device=normalized_prediction.device
        )
        num_graphs = 1
    else:
        if (
            not isinstance(membership, torch.Tensor)
            or membership.shape != (num_nodes,)
            or membership.dtype != torch.long
        ):
            raise ValueError("batch.batch must be a torch.long tensor with shape [num_nodes]")
        if membership.device != normalized_prediction.device:
            raise ValueError("batch.batch must be on the prediction device")
        if torch.any(membership < 0) or torch.any(membership[1:] < membership[:-1]):
            raise ValueError("batch.batch must contain ordered, non-negative graph indices")
        unique_graphs = torch.unique(membership, sorted=True)
        expected_graphs = torch.arange(
            unique_graphs.numel(), device=membership.device, dtype=torch.long
        )
        if not torch.equal(unique_graphs, expected_graphs):
            raise ValueError("batch.batch graph indices must be contiguous from zero")
        num_graphs = unique_graphs.numel()
        declared_num_graphs = getattr(batch, "num_graphs", None)
        if declared_num_graphs is not None and declared_num_graphs != num_graphs:
            raise ValueError("batch.batch is inconsistent with batch.num_graphs")

    if num_edges:
        source, target = edge_index
        if torch.any(membership[source] != membership[target]):
            raise ValueError("Edges must not connect nodes from different graphs")

    for graph_id in range(num_graphs):
        graph_nodes = membership == graph_id
        if not torch.any(batch.free_dof_mask[graph_nodes]):
            raise ValueError("Every graph must contain at least one free degree of freedom")

    return membership, num_graphs


def _physical_inputs(
    normalized_prediction: torch.Tensor,
    batch: Data,
    normalization: NormalizationStats,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    displacement = enforce_boundary_conditions(
        inverse_targets(normalized_prediction, normalization),
        batch.free_dof_mask,
    )

    node_mean = normalization.node_mean.to(batch.x)
    node_std = normalization.safe_std(normalization.node_std).to(batch.x)
    physical_nodes = batch.x[:, :4] * node_std + node_mean
    force = physical_nodes[:, 2:4]

    edge_mean = normalization.edge_mean.to(batch.edge_attr)
    edge_std = normalization.safe_std(normalization.edge_std).to(batch.edge_attr)
    physical_edges = batch.edge_attr[:, :5] * edge_std + edge_mean
    youngs_modulus = physical_edges[:, 3]
    area = physical_edges[:, 4]

    if not torch.isfinite(displacement).all():
        raise ValueError("Recovered physical predictions must be finite")
    if not torch.isfinite(batch.pos).all():
        raise ValueError("Physical node positions must be finite")
    if not torch.isfinite(force).all():
        raise ValueError("Recovered physical forces must be finite")
    if not torch.isfinite(youngs_modulus).all() or torch.any(youngs_modulus <= 0):
        raise ValueError("Recovered physical Young's modulus values must be positive and finite")
    if not torch.isfinite(area).all() or torch.any(area <= 0):
        raise ValueError("Recovered physical area values must be positive and finite")

    return displacement, force, youngs_modulus, area


def relative_equilibrium_residuals(
    normalized_prediction: torch.Tensor,
    batch: Data,
    normalization: NormalizationStats,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return one relative free-DOF force-equilibrium residual per graph."""

    _validate_epsilon(epsilon)
    membership, num_graphs = _validate_graph(normalized_prediction, batch)
    displacement, force, youngs_modulus, area = _physical_inputs(
        normalized_prediction, batch, normalization
    )

    source, target = batch.edge_index
    direction = batch.pos[target] - batch.pos[source]
    length = torch.linalg.vector_norm(direction, dim=1)
    if not torch.isfinite(length).all() or torch.any(length <= 0):
        raise ValueError("Physical bar lengths must be positive and finite")

    unit_direction = direction / length.unsqueeze(1)
    relative_displacement = displacement[source] - displacement[target]
    axial_displacement = (relative_displacement * unit_direction).sum(dim=1)
    stiffness = youngs_modulus * area / length
    contributions = (
        stiffness * axial_displacement
    ).unsqueeze(1) * unit_direction
    internal_force = torch.zeros_like(displacement).index_add(
        0, source, contributions
    )

    residual = internal_force - force
    relative_residuals = []
    for graph_id in range(num_graphs):
        graph_free_dofs = batch.free_dof_mask & (membership == graph_id).unsqueeze(1)
        residual_norm = torch.linalg.vector_norm(residual[graph_free_dofs])
        force_norm = torch.linalg.vector_norm(force[graph_free_dofs])
        relative_residuals.append(residual_norm / force_norm.clamp_min(epsilon))
    return torch.stack(relative_residuals)


def equilibrium_residual_loss(
    normalized_prediction: torch.Tensor,
    batch: Data,
    normalization: NormalizationStats,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Return the equal-graph mean squared relative equilibrium residual."""

    residuals = relative_equilibrium_residuals(
        normalized_prediction, batch, normalization, epsilon
    )
    return residuals.square().mean()

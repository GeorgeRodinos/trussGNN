"""Focused tests for the differentiable equilibrium-residual loss."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from trussgnn.data.loading import NormalizationStats
from trussgnn.training import (
    equilibrium_residual_loss,
    relative_equilibrium_residuals,
)


def make_normalization(nontrivial: bool = False) -> NormalizationStats:
    if nontrivial:
        return NormalizationStats(
            node_mean=torch.tensor([2.0, -3.0, 5.0, -4.0]),
            node_std=torch.tensor([2.0, 3.0, 5.0, 2.0]),
            edge_mean=torch.tensor([0.5, 0.2, -0.3, 500.0, 0.5]),
            edge_std=torch.tensor([2.0, 0.5, 0.4, 250.0, 0.25]),
            target_mean=torch.tensor([0.003, -0.002]),
            target_std=torch.tensor([0.002, 0.004]),
            source_split="train",
        )
    return NormalizationStats(
        node_mean=torch.zeros(4),
        node_std=torch.ones(4),
        edge_mean=torch.zeros(5),
        edge_std=torch.ones(5),
        target_mean=torch.zeros(2),
        target_std=torch.ones(2),
        source_split="train",
    )


def normalize(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (values - mean.to(values)) / std.to(values)


def make_one_bar(
    displacement_m: float = 0.01,
    force_n: float = 10.0,
    *,
    normalization: NormalizationStats | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Data, torch.Tensor, NormalizationStats]:
    stats = normalization or make_normalization()
    physical_nodes = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, force_n, 0.0]], dtype=dtype
    )
    supports = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=dtype)
    node_values = normalize(physical_nodes, stats.node_mean, stats.node_std)
    edge_values = torch.tensor(
        [[1.0, 1.0, 0.0, 1000.0, 1.0], [1.0, -1.0, 0.0, 1000.0, 1.0]],
        dtype=dtype,
    )
    normalized_edges = normalize(edge_values, stats.edge_mean, stats.edge_std)
    graph = Data(
        x=torch.cat((node_values, supports), dim=1),
        pos=physical_nodes[:, :2].clone(),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=normalized_edges,
        y=torch.zeros((2, 2), dtype=dtype),
        free_dof_mask=supports == 0,
    )
    physical_prediction = torch.tensor(
        [[0.0, 0.0], [displacement_m, 0.0]], dtype=dtype
    )
    prediction = normalize(
        physical_prediction, stats.target_mean, stats.safe_std(stats.target_std)
    )
    return graph, prediction, stats


def test_exact_one_bar_equilibrium() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.01)

    residuals = relative_equilibrium_residuals(prediction, graph, stats)
    loss = equilibrium_residual_loss(prediction, graph, stats)

    assert residuals.shape == (1,)
    assert residuals.item() == pytest.approx(0.0, abs=1e-7)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)


def test_known_one_bar_imbalance() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.008)

    residuals = relative_equilibrium_residuals(prediction, graph, stats)
    loss = equilibrium_residual_loss(prediction, graph, stats)

    assert residuals.item() == pytest.approx(0.2)
    assert loss.item() == pytest.approx(0.04)


def test_support_reactions_are_excluded() -> None:
    graph, prediction, stats = make_one_bar()
    changed_reaction_load = graph.clone()
    changed_reaction_load.x[0, 2] = 1234.0

    expected = equilibrium_residual_loss(prediction, graph, stats)
    actual = equilibrium_residual_loss(prediction, changed_reaction_load, stats)

    assert actual == pytest.approx(expected)
    assert actual.item() == pytest.approx(0.0, abs=1e-12)


def test_constrained_predictions_are_ignored() -> None:
    graph, prediction, stats = make_one_bar()
    changed = prediction.clone()
    changed[0] = torch.tensor([50.0, -70.0])

    assert equilibrium_residual_loss(changed, graph, stats) == pytest.approx(
        equilibrium_residual_loss(prediction, graph, stats)
    )


def test_reverse_directed_edges_do_not_double_count() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.008)

    # Both 0 -> 1 and 1 -> 0 exist, but the free node receives only its own 8 N contribution.
    assert relative_equilibrium_residuals(prediction, graph, stats).item() == pytest.approx(0.2)


def test_nontrivial_normalization_recovers_physical_equilibrium() -> None:
    stats = make_normalization(nontrivial=True)
    graph, prediction, _ = make_one_bar(normalization=stats)

    assert relative_equilibrium_residuals(prediction, graph, stats).item() == pytest.approx(
        0.0, abs=2e-6
    )


def test_two_graph_batch_preserves_order_and_equal_graph_weighting() -> None:
    first, first_prediction, stats = make_one_bar(displacement_m=0.008)
    second, second_prediction, _ = make_one_bar(displacement_m=0.01, force_n=20.0)
    batch = Batch.from_data_list([first, second])
    prediction = torch.cat((first_prediction, second_prediction))

    residuals = relative_equilibrium_residuals(prediction, batch, stats)
    loss = equilibrium_residual_loss(prediction, batch, stats)

    assert residuals.tolist() == pytest.approx([0.2, 0.5])
    assert loss.item() == pytest.approx((0.2**2 + 0.5**2) / 2)


def test_physics_is_independent_of_displacement_labels() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.008)
    changed_labels = graph.clone()
    changed_labels.y = torch.full_like(changed_labels.y, 1e20)

    assert torch.equal(
        relative_equilibrium_residuals(prediction, graph, stats),
        relative_equilibrium_residuals(prediction, changed_labels, stats),
    )
    assert torch.equal(
        equilibrium_residual_loss(prediction, graph, stats),
        equilibrium_residual_loss(prediction, changed_labels, stats),
    )


def test_physics_loss_backpropagates_to_prediction() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.008)
    prediction.requires_grad_()

    equilibrium_residual_loss(prediction, graph, stats).backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[graph.free_dof_mask].abs().max() > 0


def test_exact_equilibrium_has_finite_zero_gradient() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.01)
    prediction.requires_grad_()

    equilibrium_residual_loss(prediction, graph, stats).backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad) == 0


def test_standalone_and_float64_preserve_dtype() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.008, dtype=torch.float64)

    residuals = relative_equilibrium_residuals(prediction, graph, stats)
    loss = equilibrium_residual_loss(prediction, graph, stats)

    assert residuals.dtype == torch.float64
    assert loss.dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_physics_preserves_cuda_device() -> None:
    graph, prediction, stats = make_one_bar(displacement_m=0.008)
    graph = graph.to("cuda")
    prediction = prediction.to("cuda")

    residuals = relative_equilibrium_residuals(prediction, graph, stats)
    loss = equilibrium_residual_loss(prediction, graph, stats)

    assert residuals.device.type == "cuda"
    assert loss.device.type == "cuda"


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_epsilon_raises(epsilon: float) -> None:
    graph, prediction, stats = make_one_bar()

    with pytest.raises((TypeError, ValueError), match="epsilon"):
        relative_equilibrium_residuals(prediction, graph, stats, epsilon)


def test_invalid_prediction_shape_raises() -> None:
    graph, prediction, stats = make_one_bar()

    with pytest.raises(ValueError, match="shape"):
        relative_equilibrium_residuals(prediction[:, :1], graph, stats)


def test_missing_or_non_boolean_free_dof_mask_raises() -> None:
    graph, prediction, stats = make_one_bar()
    missing = graph.clone()
    del missing.free_dof_mask
    non_boolean = graph.clone()
    non_boolean.free_dof_mask = non_boolean.free_dof_mask.float()

    with pytest.raises(ValueError, match="free_dof_mask"):
        relative_equilibrium_residuals(prediction, missing, stats)
    with pytest.raises(ValueError, match="torch.bool"):
        relative_equilibrium_residuals(prediction, non_boolean, stats)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [(3, 0.0, "Young's modulus"), (4, -1.0, "area")],
)
def test_non_positive_material_properties_raise(column: int, value: float, message: str) -> None:
    graph, prediction, stats = make_one_bar()
    graph.edge_attr[:, column] = value

    with pytest.raises(ValueError, match=message):
        relative_equilibrium_residuals(prediction, graph, stats)


@pytest.mark.parametrize("location", ["prediction", "force", "position", "edge"])
def test_non_finite_values_raise(location: str) -> None:
    graph, prediction, stats = make_one_bar()
    if location == "prediction":
        prediction[1, 0] = float("nan")
    elif location == "force":
        graph.x[1, 2] = float("inf")
    elif location == "position":
        graph.pos[1, 0] = float("nan")
    else:
        graph.edge_attr[0, 3] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        relative_equilibrium_residuals(prediction, graph, stats)


def test_zero_length_edge_raises() -> None:
    graph, prediction, stats = make_one_bar()
    graph.pos[1] = graph.pos[0]

    with pytest.raises(ValueError, match="bar lengths"):
        relative_equilibrium_residuals(prediction, graph, stats)


def test_invalid_edge_index_and_feature_shapes_raise() -> None:
    graph, prediction, stats = make_one_bar()
    invalid_index = graph.clone()
    invalid_index.edge_index[1, 0] = 2
    too_few_edge_features = graph.clone()
    too_few_edge_features.edge_attr = too_few_edge_features.edge_attr[:, :4]
    too_few_node_features = graph.clone()
    too_few_node_features.x = too_few_node_features.x[:, :3]

    with pytest.raises(ValueError, match="invalid node index"):
        relative_equilibrium_residuals(prediction, invalid_index, stats)
    with pytest.raises(ValueError, match="at least 5"):
        relative_equilibrium_residuals(prediction, too_few_edge_features, stats)
    with pytest.raises(ValueError, match="at least 4"):
        relative_equilibrium_residuals(prediction, too_few_node_features, stats)


def test_invalid_batch_membership_raises() -> None:
    graph, prediction, stats = make_one_bar()
    graph.batch = torch.tensor([0, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="contiguous"):
        relative_equilibrium_residuals(prediction, graph, stats)


def test_cross_graph_edge_raises() -> None:
    graph, prediction, stats = make_one_bar()
    graph.batch = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="different graphs"):
        relative_equilibrium_residuals(prediction, graph, stats)

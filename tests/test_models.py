"""Focused tests for the three Phase 4C node-level models."""

from io import BytesIO

import pytest
import torch
from torch_geometric.data import Batch, Data

from trussgnn.data.loading import NormalizationStats, inverse_targets
from trussgnn.models import EdgeAwareGNN, NodeMLP, ZeroDisplacementBaseline


def make_graph(node_count: int, offset: float = 0.0) -> Data:
    """Create a small deterministic graph with bidirectional chain edges."""

    x = torch.arange(node_count * 6, dtype=torch.float32).reshape(node_count, 6) / 10
    x = x + offset
    edge_pairs = []
    for node in range(node_count - 1):
        edge_pairs.extend([[node, node + 1], [node + 1, node]])
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.arange(len(edge_pairs) * 5, dtype=torch.float32).reshape(-1, 5) / 20
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.zeros((node_count, 2)),
    )
    graph.num_nodes = node_count
    graph.validate(raise_on_error=True)
    return graph


@pytest.fixture
def batch() -> Batch:
    return Batch.from_data_list([make_graph(3), make_graph(4, offset=0.25)])


@pytest.fixture
def target_stats() -> NormalizationStats:
    return NormalizationStats(
        node_mean=torch.zeros(4),
        node_std=torch.ones(4),
        edge_mean=torch.zeros(5),
        edge_std=torch.ones(5),
        target_mean=torch.tensor([2.0, -3.0]),
        target_std=torch.tensor([4.0, 0.0]),
        source_split="train",
    )


def test_zero_baseline_has_no_parameters_and_correct_shape(batch, target_stats) -> None:
    model = ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std)
    prediction = model(batch)

    assert list(model.parameters()) == []
    assert prediction.shape == (batch.x.shape[0], 2)


def test_zero_baseline_inverse_targets_are_physical_zeros(batch, target_stats) -> None:
    model = ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std)

    physical = inverse_targets(model(batch), target_stats)

    assert torch.equal(physical, torch.zeros_like(physical))


def test_zero_baseline_buffer_follows_dtype_conversion(target_stats) -> None:
    model = ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std).double()

    assert model.normalized_zero.dtype == torch.float64


def test_node_mlp_output_is_finite_and_has_one_row_per_node(batch) -> None:
    prediction = NodeMLP(hidden_dim=16, num_hidden_layers=2)(batch)

    assert prediction.shape == (batch.x.shape[0], 2)
    assert torch.isfinite(prediction).all()


def test_node_mlp_ignores_connectivity_and_edge_attributes() -> None:
    model = NodeMLP(hidden_dim=12).eval()
    original = make_graph(3)
    changed = original.clone()
    changed.edge_index = torch.tensor([[0, 2, 1, 2], [2, 0, 2, 1]])
    changed.edge_attr = changed.edge_attr + 100.0

    assert torch.equal(model(original), model(changed))


def test_edge_aware_gnn_output_is_finite_and_has_one_row_per_node(batch) -> None:
    prediction = EdgeAwareGNN(hidden_dim=16, num_message_passing_layers=2)(batch)

    assert prediction.shape == (batch.x.shape[0], 2)
    assert torch.isfinite(prediction).all()


def test_edge_attributes_change_gnn_predictions() -> None:
    torch.manual_seed(5)
    model = EdgeAwareGNN(hidden_dim=12, num_message_passing_layers=2).eval()
    original = make_graph(3)
    changed = original.clone()
    changed.edge_attr = changed.edge_attr + 2.0

    assert not torch.allclose(model(original), model(changed))


def test_connectivity_changes_gnn_predictions() -> None:
    torch.manual_seed(6)
    model = EdgeAwareGNN(hidden_dim=12, num_message_passing_layers=2).eval()
    original = make_graph(3)
    changed = original.clone()
    changed.edge_index = torch.tensor([[0, 2, 1, 2], [2, 0, 2, 1]])

    assert not torch.allclose(model(original), model(changed))


@pytest.mark.parametrize(
    "model",
    [NodeMLP(hidden_dim=10), EdgeAwareGNN(hidden_dim=10, num_message_passing_layers=2)],
)
def test_all_learned_parameters_receive_finite_gradients(model, batch) -> None:
    model(batch).square().mean().backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    "model",
    [
        NodeMLP(hidden_dim=8),
        EdgeAwareGNN(hidden_dim=8, num_message_passing_layers=2),
    ],
)
def test_learned_models_support_different_graph_sizes_in_one_batch(model, batch) -> None:
    assert model(batch).shape == (7, 2)


@pytest.mark.parametrize("model_type", [NodeMLP, EdgeAwareGNN])
def test_state_dict_round_trip_preserves_predictions(model_type, batch) -> None:
    torch.manual_seed(11)
    first = model_type(hidden_dim=8).eval()
    expected = first(batch)
    saved = BytesIO()
    torch.save(first.state_dict(), saved)
    saved.seek(0)

    second = model_type(hidden_dim=8).eval()
    second.load_state_dict(torch.load(saved, weights_only=True))

    assert torch.equal(second(batch), expected)


def test_zero_baseline_state_dict_round_trip_preserves_predictions(batch, target_stats) -> None:
    first = ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std)
    saved = BytesIO()
    torch.save(first.state_dict(), saved)
    saved.seek(0)

    second = ZeroDisplacementBaseline(torch.zeros(2), torch.ones(2))
    second.load_state_dict(torch.load(saved, weights_only=True))

    assert torch.equal(second(batch), first(batch))


@pytest.mark.parametrize("model_type", [NodeMLP, EdgeAwareGNN])
def test_identical_seeds_produce_identical_initialization_and_output(model_type, batch) -> None:
    torch.manual_seed(19)
    first = model_type(hidden_dim=8).eval()
    torch.manual_seed(19)
    second = model_type(hidden_dim=8).eval()

    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
    assert torch.equal(first(batch), second(batch))


def test_all_models_preserve_float64_with_matching_batch(batch, target_stats) -> None:
    double_batch = batch.clone()
    double_batch.x = double_batch.x.double()
    double_batch.edge_attr = double_batch.edge_attr.double()
    double_batch.y = double_batch.y.double()
    models = [
        ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std).double(),
        NodeMLP(hidden_dim=8).double(),
        EdgeAwareGNN(hidden_dim=8).double(),
    ]

    for model in models:
        assert model(double_batch).dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_models_and_predictions_remain_on_cuda(batch, target_stats) -> None:
    cuda_batch = batch.clone().to("cuda")
    models = [
        ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std).cuda(),
        NodeMLP(hidden_dim=8).cuda(),
        EdgeAwareGNN(hidden_dim=8).cuda(),
    ]

    for model in models:
        assert model(cuda_batch).device.type == "cuda"


def test_models_do_not_modify_input_batch(batch, target_stats) -> None:
    before = {name: value.clone() for name, value in batch if isinstance(value, torch.Tensor)}
    models = [
        ZeroDisplacementBaseline(target_stats.target_mean, target_stats.target_std),
        NodeMLP(hidden_dim=8),
        EdgeAwareGNN(hidden_dim=8),
    ]

    for model in models:
        model(batch)
    for name, value in before.items():
        assert torch.equal(batch[name], value)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: NodeMLP(hidden_dim=0), "hidden_dim"),
        (lambda: NodeMLP(num_hidden_layers=0), "num_hidden_layers"),
        (lambda: NodeMLP(dropout=-0.1), "dropout"),
        (lambda: NodeMLP(dropout=1.0), "dropout"),
        (lambda: EdgeAwareGNN(hidden_dim=0), "hidden_dim"),
        (
            lambda: EdgeAwareGNN(num_message_passing_layers=0),
            "num_message_passing_layers",
        ),
        (lambda: EdgeAwareGNN(dropout=-0.1), "dropout"),
        (lambda: EdgeAwareGNN(dropout=1.0), "dropout"),
    ],
)
def test_invalid_constructor_configuration_raises_clear_error(constructor, message) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()

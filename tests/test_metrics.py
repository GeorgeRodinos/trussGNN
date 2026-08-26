"""Tests for Phase 4D1 masked loss and physical displacement metrics."""

import pytest
import torch
from torch_geometric.data import Batch, Data

import trussgnn.training.metrics as metrics_module
from trussgnn.data.loading import NormalizationStats, enforce_boundary_conditions
from trussgnn.training import masked_mse, physical_metrics


@pytest.fixture
def normalization() -> NormalizationStats:
    return NormalizationStats(
        node_mean=torch.zeros(4),
        node_std=torch.ones(4),
        edge_mean=torch.zeros(5),
        edge_std=torch.ones(5),
        target_mean=torch.zeros(2),
        target_std=torch.ones(2),
        source_split="train",
    )


def test_masked_mse_matches_manual_calculation_and_ignores_constraints() -> None:
    prediction = torch.tensor([[1.0, 200.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False], [True, True]])

    loss = masked_mse(prediction, target, mask)
    prediction[0, 1] = 10_000.0

    assert loss == pytest.approx((1.0 + 9.0 + 16.0) / 3.0)
    assert masked_mse(prediction, target, mask) == loss


def test_masked_mse_rejects_empty_mask() -> None:
    values = torch.zeros((2, 2))

    with pytest.raises(ValueError, match="free degree of freedom"):
        masked_mse(values, values, torch.zeros_like(values, dtype=torch.bool))


def test_physical_mae_rmse_and_millimetres_match_manual_values(normalization) -> None:
    prediction = torch.tensor([[0.001, 10.0], [0.003, 0.004]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False], [True, True]])
    graph = Data(x=torch.zeros((2, 6)))

    result = physical_metrics(prediction, target, mask, graph, normalization)
    errors = torch.tensor([0.001, 0.003, 0.004])
    expected_mae = errors.abs().mean().item()
    expected_rmse = errors.square().mean().sqrt().item()

    assert result["mae_m"] == pytest.approx(expected_mae)
    assert result["mae_mm"] == pytest.approx(expected_mae * 1_000)
    assert result["rmse_m"] == pytest.approx(expected_rmse)
    assert result["rmse_mm"] == pytest.approx(expected_rmse * 1_000)


def test_relative_l2_is_averaged_per_graph_not_globally(normalization) -> None:
    prediction = torch.tensor([[2.0, 0.0], [11.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [10.0, 0.0]])
    mask = torch.tensor([[True, False], [True, False]])
    graphs = Batch.from_data_list([Data(x=torch.zeros((1, 6))), Data(x=torch.zeros((1, 6)))])

    result = physical_metrics(prediction, target, mask, graphs, normalization)

    assert result["mean_graph_relative_l2"] == pytest.approx((1.0 + 0.1) / 2.0)
    assert result["mean_graph_relative_l2"] != pytest.approx(2**0.5 / 101**0.5)


def test_single_graph_without_batch_vector_is_supported(normalization) -> None:
    prediction = torch.tensor([[2.0, 0.0], [4.0, 0.0]])
    target = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    mask = torch.tensor([[True, False], [True, False]])
    graph = Data(x=torch.zeros((2, 6)))

    result = physical_metrics(prediction, target, mask, graph, normalization)

    assert result["mean_graph_relative_l2"] == pytest.approx(1.0)


def test_zero_target_relative_error_is_finite(normalization) -> None:
    prediction = torch.tensor([[1.0, 0.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])

    result = physical_metrics(prediction, target, mask, Data(x=torch.zeros((1, 6))), normalization)

    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())
    assert result["mean_graph_relative_l2"] == pytest.approx(1e12)


def test_physical_predictions_are_zeroed_at_constraints(monkeypatch, normalization) -> None:
    captured = {}

    def capture(prediction, mask):
        result = enforce_boundary_conditions(prediction, mask)
        captured["prediction"] = result
        return result

    monkeypatch.setattr(metrics_module, "enforce_boundary_conditions", capture)
    prediction = torch.tensor([[5.0, 8.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[False, True], [True, False]])

    physical_metrics(prediction, target, mask, Data(x=torch.zeros((2, 6))), normalization)

    assert torch.count_nonzero(captured["prediction"][~mask]) == 0


def test_metric_functions_do_not_modify_inputs(normalization) -> None:
    prediction = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False], [True, True]])
    before = (prediction.clone(), target.clone(), mask.clone())

    masked_mse(prediction, target, mask)
    physical_metrics(prediction, target, mask, Data(x=torch.zeros((2, 6))), normalization)

    assert torch.equal(prediction, before[0])
    assert torch.equal(target, before[1])
    assert torch.equal(mask, before[2])


def test_float64_loss_and_metrics(normalization) -> None:
    prediction = torch.tensor([[1.0, 2.0]], dtype=torch.float64)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True]])

    loss = masked_mse(prediction, target, mask)
    result = physical_metrics(prediction, target, mask, Data(x=torch.zeros((1, 6))), normalization)

    assert loss.dtype == torch.float64
    assert all(isinstance(value, float) for value in result.values())
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_loss_and_metrics_support_cuda(normalization) -> None:
    prediction = torch.tensor([[1.0, 2.0]], device="cuda")
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, True]], device="cuda")
    graph = Data(x=torch.zeros((1, 6), device="cuda"))

    loss = masked_mse(prediction, target, mask)
    result = physical_metrics(prediction, target, mask, graph, normalization)

    assert loss.device.type == "cuda"
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())

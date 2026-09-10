"""Focused integration tests for physics-informed training and evaluation."""

import copy

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import trussgnn.training.engine as engine_module
from trussgnn.data import NormalizationStats
from trussgnn.training import (
    TrainingConfig,
    equilibrium_residual_loss,
    evaluate_model,
    fit_model,
    masked_mse,
    train_one_epoch,
)


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


def simple_graph(
    target: float = 1.0, graph_id: int = 0, *, with_physics: bool = False
) -> Data:
    graph = Data(
        x=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        y=torch.tensor([[target, 0.0], [target, 0.0]]),
        free_dof_mask=torch.tensor([[True, False], [True, False]]),
        graph_id=torch.tensor(graph_id),
    )
    if with_physics:
        graph.pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        graph.edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        graph.edge_attr = torch.ones((2, 5))
    return graph


def one_bar(displacement: float = 0.008) -> tuple[Data, torch.Tensor]:
    graph = Data(
        x=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 10.0, 0.0, 0.0, 1.0],
            ]
        ),
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor(
            [
                [1.0, 1.0, 0.0, 1000.0, 1.0],
                [1.0, -1.0, 0.0, 1000.0, 1.0],
            ]
        ),
        y=torch.tensor([[0.0, 0.0], [0.01, 0.0]]),
        free_dof_mask=torch.tensor([[False, False], [True, False]]),
    )
    prediction = torch.tensor([[0.0, 0.0], [displacement, 0.0]])
    return graph, prediction


class ScaleModel(nn.Module):
    def __init__(self, value: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value))

    def forward(self, batch: Data) -> torch.Tensor:
        return batch.x[:, :2] * self.scale


class OneBarModel(nn.Module):
    def __init__(self, displacement: float = 0.008) -> None:
        super().__init__()
        self.displacement = nn.Parameter(torch.tensor(displacement))

    def forward(self, batch: Data) -> torch.Tensor:
        return torch.stack(
            (batch.pos[:, 0] * self.displacement, torch.zeros_like(batch.pos[:, 1])),
            dim=1,
        )


def test_physics_configuration_defaults_and_positive_values() -> None:
    defaults = TrainingConfig()
    positive = TrainingConfig(physics_loss_weight=0.25, physics_epsilon=1e-9)

    assert defaults.physics_loss_weight == 0.0
    assert defaults.physics_epsilon == 1e-12
    assert positive.physics_loss_weight == 0.25
    assert positive.physics_epsilon == 1e-9


@pytest.mark.parametrize(
    "value",
    [-1.0, float("nan"), float("inf"), float("-inf"), True],
)
def test_invalid_physics_weights_raise(value: float) -> None:
    with pytest.raises(ValueError, match="physics_loss_weight"):
        TrainingConfig(physics_loss_weight=value)


@pytest.mark.parametrize(
    "value",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True],
)
def test_invalid_physics_epsilon_raises(value: float) -> None:
    with pytest.raises(ValueError, match="physics_epsilon"):
        TrainingConfig(physics_epsilon=value)


def test_zero_weight_matches_original_path_and_skips_physics(monkeypatch) -> None:
    original_model = ScaleModel(0.2)
    integrated_model = copy.deepcopy(original_model)
    original_optimizer = torch.optim.SGD(original_model.parameters(), lr=0.1)
    integrated_optimizer = torch.optim.SGD(integrated_model.parameters(), lr=0.1)

    def unexpected_physics_call(*_args, **_kwargs):
        raise AssertionError("physics loss must not run at zero weight")

    monkeypatch.setattr(engine_module, "equilibrium_residual_loss", unexpected_physics_call)
    original_loss = train_one_epoch(
        original_model,
        DataLoader([simple_graph()], batch_size=1),
        original_optimizer,
        "cpu",
    )
    integrated_loss = train_one_epoch(
        integrated_model,
        DataLoader([simple_graph()], batch_size=1),
        integrated_optimizer,
        "cpu",
        normalization=None,
        config=TrainingConfig(physics_loss_weight=0.0),
    )

    assert integrated_loss == original_loss
    assert torch.equal(integrated_model.scale, original_model.scale)


def test_zero_weight_epoch_summary_has_zero_physics_loss(monkeypatch) -> None:
    def unexpected_physics_call(*_args, **_kwargs):
        raise AssertionError("physics loss must not run at zero weight")

    monkeypatch.setattr(engine_module, "equilibrium_residual_loss", unexpected_physics_call)
    model = ScaleModel(0.2)
    losses = engine_module._train_epoch(
        model,
        DataLoader([simple_graph()], batch_size=1),
        torch.optim.SGD(model.parameters(), lr=0.0),
        "cpu",
        normalization(),
        TrainingConfig(physics_loss_weight=0.0),
    )

    assert losses["physics_loss"] == 0.0
    assert losses["total_loss"] == losses["data_loss"]


def test_zero_weight_fit_accepts_legacy_graphs_without_physics_fields(tmp_path) -> None:
    model = ScaleModel(0.2)
    graph = simple_graph()

    result = fit_model(
        model,
        DataLoader([graph], batch_size=1),
        DataLoader([graph], batch_size=1),
        normalization(),
        TrainingConfig(max_epochs=1, physics_loss_weight=0.0),
        tmp_path / "best.pt",
    )

    assert result.history[0]["train_physics_loss"] == 0.0
    assert result.history[0]["validation_physics_loss"] == 0.0
    assert result.history[0]["train_total_loss"] == result.history[0]["train_data_loss"]
    assert result.history[0]["validation_total_loss"] == result.history[0]["loss"]


def test_positive_weight_objective_and_gradient_are_correct() -> None:
    graph, _ = one_bar()
    model = OneBarModel()
    prediction = model(graph)
    data_loss = masked_mse(prediction, graph.y, graph.free_dof_mask)
    physics_loss = equilibrium_residual_loss(prediction, graph, normalization())
    weight = 0.5
    expected_total = data_loss + weight * physics_loss
    data_gradient = torch.autograd.grad(
        data_loss, model.displacement, retain_graph=True
    )[0]
    physics_gradient = torch.autograd.grad(
        physics_loss, model.displacement, retain_graph=True
    )[0]

    expected_total.backward()

    assert data_loss.item() == pytest.approx(4e-6)
    assert physics_loss.item() == pytest.approx(0.04)
    assert expected_total.item() == pytest.approx(0.020004)
    assert model.displacement.grad is not None
    assert torch.isfinite(model.displacement.grad)
    assert model.displacement.grad != 0
    assert physics_gradient != 0
    assert model.displacement.grad == pytest.approx(
        data_gradient + weight * physics_gradient
    )

    training_model = OneBarModel()
    reported_total = train_one_epoch(
        training_model,
        DataLoader([graph], batch_size=1),
        torch.optim.SGD(training_model.parameters(), lr=0.0),
        "cpu",
        normalization(),
        TrainingConfig(physics_loss_weight=weight),
    )
    assert reported_total == pytest.approx(expected_total.item())


def test_labels_change_data_loss_but_not_physics_loss() -> None:
    graph, prediction = one_bar()
    changed = graph.clone()
    changed.y = changed.y + 2.0

    assert masked_mse(prediction, graph.y, graph.free_dof_mask) != masked_mse(
        prediction, changed.y, changed.free_dof_mask
    )
    assert torch.equal(
        equilibrium_residual_loss(prediction, graph, normalization()),
        equilibrium_residual_loss(prediction, changed, normalization()),
    )


def test_positive_weight_requires_physics_fields() -> None:
    model = ScaleModel()

    with pytest.raises(ValueError, match="batch.pos"):
        train_one_epoch(
            model,
            DataLoader([simple_graph()], batch_size=1),
            torch.optim.SGD(model.parameters(), lr=0.0),
            "cpu",
            normalization(),
            TrainingConfig(physics_loss_weight=1.0),
        )


def test_epoch_losses_use_free_dofs_and_graph_counts(monkeypatch) -> None:
    graphs = [simple_graph(1.0), simple_graph(2.0), simple_graph(4.0)]
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    model = ScaleModel(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    def batch_physics_loss(prediction, batch, *_args, **_kwargs):
        value = 1.0 if batch.num_graphs == 2 else 9.0
        return prediction.sum() * 0.0 + value

    monkeypatch.setattr(engine_module, "equilibrium_residual_loss", batch_physics_loss)
    losses = engine_module._train_epoch(
        model,
        loader,
        optimizer,
        "cpu",
        normalization(),
        TrainingConfig(physics_loss_weight=2.0),
    )
    expected_data = torch.cat(
        [graph.y[graph.free_dof_mask] for graph in graphs]
    ).square().mean()
    expected_physics = (1.0 * 2 + 9.0) / 3

    assert losses["data_loss"] == pytest.approx(expected_data.item())
    assert losses["physics_loss"] == pytest.approx(expected_physics)
    assert losses["total_loss"] == pytest.approx(
        expected_data.item() + 2.0 * expected_physics
    )


def test_evaluation_averages_equilibrium_by_graph_not_batch(monkeypatch) -> None:
    graphs = [
        simple_graph(graph_id=value, with_physics=True) for value in (1, 3, 5)
    ]

    def graph_residuals(prediction, batch, *_args, **_kwargs):
        return batch.graph_id.to(prediction)

    monkeypatch.setattr(
        engine_module, "relative_equilibrium_residuals", graph_residuals
    )
    result = evaluate_model(
        ScaleModel(), DataLoader(graphs, batch_size=2), "cpu", normalization()
    )

    assert result["mean_equilibrium_residual"] == pytest.approx(3.0)
    assert result["mean_equilibrium_residual"] != pytest.approx(3.5)
    assert result["physics_loss"] == pytest.approx((1.0**2 + 3.0**2 + 5.0**2) / 3)


def test_checkpoint_selection_ignores_worsening_total_loss(
    monkeypatch, tmp_path
) -> None:
    model = ScaleModel()
    validation_results = iter(
        [
            {"loss": 2.0, "physics_loss": 1.0, "rmse_m": 2.0},
            {"loss": 1.0, "physics_loss": 10.0, "rmse_m": 1.0},
        ]
    )

    def fake_train(model, *_args):
        with torch.no_grad():
            model.scale.add_(1)
        return {"data_loss": 1.0, "physics_loss": 1.0, "total_loss": 2.0}

    def fake_evaluate(*_args):
        values = next(validation_results)
        return {
            **values,
            "mae_m": values["rmse_m"],
            "mae_mm": values["rmse_m"] * 1000,
            "rmse_mm": values["rmse_m"] * 1000,
            "mean_graph_relative_l2": values["rmse_m"],
            "mean_equilibrium_residual": values["physics_loss"],
        }

    monkeypatch.setattr(engine_module, "_train_epoch", fake_train)
    monkeypatch.setattr(engine_module, "evaluate_model", fake_evaluate)
    result = fit_model(
        model,
        [],
        [],
        normalization(),
        TrainingConfig(max_epochs=2, patience=2, physics_loss_weight=1.0),
        tmp_path / "best.pt",
    )

    assert result.history[0]["validation_total_loss"] == pytest.approx(3.0)
    assert result.history[1]["validation_total_loss"] == pytest.approx(11.0)
    assert result.best_epoch == 2
    assert model.scale.item() == pytest.approx(2.0)

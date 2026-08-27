"""Focused tests for deterministic Phase 4D2 training."""

from pathlib import Path

import pytest
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import trussgnn.training.engine as engine_module
from trussgnn.data.loading import NormalizationStats
from trussgnn.models import EdgeAwareGNN, NodeMLP, ZeroDisplacementBaseline
from trussgnn.training import (
    TrainingConfig,
    evaluate_model,
    fit_model,
    seed_everything,
    train_one_epoch,
)


def make_graph(value: float = 1.0, nodes: int = 2, dtype=torch.float32) -> Data:
    x = torch.zeros((nodes, 6), dtype=dtype)
    x[:, 0] = torch.arange(nodes, dtype=dtype) + value
    pairs = [[i, i + 1] for i in range(nodes - 1)]
    pairs += [[j, i] for i, j in pairs]
    edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    edge_attr = torch.full((len(pairs), 5), value, dtype=dtype)
    y = torch.stack((0.2 * x[:, 0], -0.1 * x[:, 0]), dim=1)
    mask = torch.ones((nodes, 2), dtype=torch.bool)
    mask[0, 1] = False
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, free_dof_mask=mask)


def make_normalization() -> NormalizationStats:
    return NormalizationStats(
        node_mean=torch.zeros(4),
        node_std=torch.ones(4),
        edge_mean=torch.zeros(5),
        edge_std=torch.ones(5),
        target_mean=torch.zeros(2),
        target_std=torch.ones(2),
        source_split="train",
    )


@pytest.fixture
def normalization() -> NormalizationStats:
    return make_normalization()


class ScaleModel(nn.Module):
    def __init__(self, value: float = 0.0, dtype=torch.float32) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(value, dtype=dtype))

    def forward(self, batch: Data) -> torch.Tensor:
        return batch.x[:, :2] * self.scale


def test_epoch_loss_weights_each_free_dof() -> None:
    graphs = [make_graph(1.0, 2), make_graph(2.0, 4)]
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    model = ScaleModel(0.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    expected = torch.cat([graph.y[graph.free_dof_mask] for graph in graphs]).square().mean()

    actual = train_one_epoch(model, loader, optimizer, "cpu")

    assert actual == pytest.approx(expected.item())


def test_evaluation_matches_complete_split_and_preserves_parameters(normalization) -> None:
    graphs = [make_graph(1.0, 2), make_graph(3.0, 3)]
    model = ScaleModel(0.5)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = evaluate_model(model, DataLoader(graphs, batch_size=1), "cpu", normalization)
    prediction = torch.cat([graph.x[:, :2] * 0.5 for graph in graphs])
    target = torch.cat([graph.y for graph in graphs])
    mask = torch.cat([graph.free_dof_mask for graph in graphs])
    error = prediction[mask] - target[mask]

    assert result["loss"] == pytest.approx(error.square().mean().item())
    assert result["mae_m"] == pytest.approx(error.abs().mean().item())
    assert result["rmse_m"] == pytest.approx(error.square().mean().sqrt().item())
    assert all(torch.equal(model.state_dict()[name], value) for name, value in before.items())


def test_optimizer_step_changes_parameters() -> None:
    model = ScaleModel()
    before = model.scale.detach().clone()
    train_one_epoch(model, DataLoader([make_graph()], batch_size=1),
                    torch.optim.Adam(model.parameters(), lr=0.1), "cpu")
    assert not torch.equal(model.scale, before)


@pytest.mark.parametrize(
    "model_name",
    ["mlp", "gnn"],
)
def test_tiny_learned_models_reduce_training_loss(model_name) -> None:
    seed_everything(17)
    model = (
        NodeMLP(hidden_dim=8, num_hidden_layers=1)
        if model_name == "mlp"
        else EdgeAwareGNN(hidden_dim=8, num_message_passing_layers=1)
    )
    loader = DataLoader([make_graph()], batch_size=1, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    first = evaluate_model(model, loader, "cpu", make_normalization())["loss"]
    for _ in range(30):
        train_one_epoch(model, loader, optimizer, "cpu")
    last = evaluate_model(model, loader, "cpu", make_normalization())["loss"]
    assert last < first


def test_fixed_seed_reproduces_fit_history_and_predictions(tmp_path, normalization) -> None:
    def run(name: str):
        seed_everything(17)
        model = NodeMLP(hidden_dim=6, num_hidden_layers=1)
        graphs = [make_graph(1.0), make_graph(2.0)]
        config = TrainingConfig(max_epochs=4, patience=4, seed=17)
        result = fit_model(model, DataLoader(graphs, batch_size=2, shuffle=False),
                           DataLoader(graphs, batch_size=2), normalization, config, tmp_path / name)
        return result.history, model(graphs[0]).detach()

    history_a, prediction_a = run("a.pt")
    history_b, prediction_b = run("b.pt")
    assert history_a == history_b
    assert torch.equal(prediction_a, prediction_b)


def test_best_validation_state_is_restored_and_early_stopping_obeys_patience(
    monkeypatch, tmp_path, normalization
) -> None:
    model = ScaleModel()
    validation_rmse = iter([3.0, 2.0, 2.05, 2.2])

    def fake_train(model, *_args):
        with torch.no_grad():
            model.scale.add_(1)
        return float(model.scale.detach().item())

    def fake_evaluate(model, *_args):
        rmse = next(validation_rmse)
        return {"loss": rmse, "mae_m": rmse, "mae_mm": rmse * 1000,
                "rmse_m": rmse, "rmse_mm": rmse * 1000, "mean_graph_relative_l2": rmse}

    monkeypatch.setattr(engine_module, "train_one_epoch", fake_train)
    monkeypatch.setattr(engine_module, "evaluate_model", fake_evaluate)
    config = TrainingConfig(max_epochs=10, patience=2, min_delta=0.1)
    result = fit_model(model, [], [], normalization, config, tmp_path / "best.pt")

    assert result.best_epoch == 2
    assert result.epochs_completed == 4
    assert result.stopped_early
    assert model.scale.item() == pytest.approx(2.0)


def test_checkpoint_contains_required_fields(tmp_path, normalization) -> None:
    path = tmp_path / "checkpoint.pt"
    model = NodeMLP(hidden_dim=4, num_hidden_layers=1)
    graph = make_graph()
    config = TrainingConfig(max_epochs=1)
    fit_model(model, DataLoader([graph]), DataLoader([graph]), normalization, config, path)
    checkpoint = torch.load(path, weights_only=True)

    assert set(checkpoint) == {
        "model_state_dict", "optimizer_state_dict", "epoch",
        "best_validation_rmse_m", "training_config",
    }
    assert checkpoint["training_config"] == {
        "max_epochs": 1, "learning_rate": 0.001, "weight_decay": 0.0,
        "patience": 10, "min_delta": 0.0, "seed": 42, "device": "cpu",
    }


def test_float64_evaluation(normalization) -> None:
    graph = make_graph(dtype=torch.float64)
    result = evaluate_model(ScaleModel(dtype=torch.float64), DataLoader([graph]), "cpu", normalization)
    assert all(isinstance(value, float) for value in result.values())


def test_empty_loaders_and_non_trainable_model_raise(normalization, tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(ScaleModel(), [], "cpu", normalization)
    with pytest.raises(ValueError, match="empty"):
        train_one_epoch(ScaleModel(), [], torch.optim.SGD(ScaleModel().parameters()), "cpu")
    baseline = ZeroDisplacementBaseline(torch.zeros(2), torch.ones(2))
    graph = make_graph()
    assert evaluate_model(baseline, DataLoader([graph]), "cpu", normalization)["loss"] >= 0
    with pytest.raises(ValueError, match="trainable parameters"):
        fit_model(baseline, [], [], normalization, TrainingConfig(max_epochs=1), tmp_path / "x.pt")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_training(normalization) -> None:
    model = ScaleModel().cuda()
    train_one_epoch(model, DataLoader([make_graph()]), torch.optim.Adam(model.parameters()), "cuda")
    result = evaluate_model(model, DataLoader([make_graph()]), "cuda", normalization)
    assert model.scale.device.type == "cuda"
    assert all(torch.isfinite(torch.tensor(value)) for value in result.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_epochs": 0}, {"learning_rate": 0}, {"weight_decay": -1},
        {"patience": -1}, {"min_delta": -1}, {"seed": -1}, {"device": "nonsense"},
    ],
)
def test_invalid_training_configuration(kwargs) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)

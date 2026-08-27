"""Focused tests for Phase 5B per-graph error analysis."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from trussgnn.analysis.error_analysis import (
    PerGraphError,
    assign_magnitude_groups,
    build_summary,
    load_verified_model,
    paired_comparisons,
    per_graph_metrics,
    write_analysis,
)
from trussgnn.data import LoadedDataset, NormalizationStats


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


def record(
    model: str,
    seed: int,
    graph_id: int,
    norm: float,
    rmse: float,
    *,
    split: str = "validation",
    panels: int = 2,
    group: str = "medium",
) -> PerGraphError:
    return PerGraphError(
        model, seed, f"{model}-{seed}", split, graph_id, 5, panels,
        norm, rmse, rmse / max(norm, 1e-12), group,
    )


def test_per_graph_metrics_match_manual_values_and_millimetres(normalization) -> None:
    prediction = torch.tensor([[0.002, 9.0], [0.006, 0.008]])
    target = torch.tensor([[0.001, 0.0], [0.002, 0.004]])
    mask = torch.tensor([[True, False], [True, True]])

    true_norm, rmse, relative = per_graph_metrics(
        prediction, target, mask, normalization
    )
    physical_target = torch.tensor([0.001, 0.002, 0.004])
    error = torch.tensor([0.001, 0.004, 0.004])

    assert true_norm == pytest.approx(torch.linalg.vector_norm(physical_target) * 1_000)
    assert rmse == pytest.approx(error.square().mean().sqrt() * 1_000)
    assert relative == pytest.approx(
        torch.linalg.vector_norm(error) / torch.linalg.vector_norm(physical_target)
    )


def test_constrained_dofs_do_not_affect_per_graph_metrics(normalization) -> None:
    target = torch.tensor([[0.001, 0.0]])
    mask = torch.tensor([[True, False]])

    first = per_graph_metrics(torch.tensor([[0.002, 1.0]]), target, mask, normalization)
    second = per_graph_metrics(torch.tensor([[0.002, 1e9]]), target, mask, normalization)

    assert first == second


def test_magnitude_groups_use_unique_split_quartiles_for_every_model() -> None:
    records = []
    for model, seed in (("zero", 42), ("mlp", 7), ("gnn", 7)):
        records.extend(record(model, seed, graph_id, norm, 1.0) for graph_id, norm in enumerate([1, 2, 3, 4]))

    grouped, quartiles = assign_magnitude_groups(records)
    groups = {(item.graph_id, item.magnitude_group) for item in grouped}

    assert quartiles["validation"] == {
        "q25_true_norm_mm": pytest.approx(1.75),
        "q75_true_norm_mm": pytest.approx(3.25),
    }
    assert groups == {(0, "low"), (1, "medium"), (2, "medium"), (3, "high")}


def test_panel_count_summary_is_separate_and_correct() -> None:
    records = [
        record("mlp", 7, 1, 2, 1, panels=6),
        record("gnn", 7, 1, 2, 0.5, panels=6),
        record("mlp", 7, 2, 3, 3, panels=7),
        record("gnn", 7, 2, 3, 2, panels=7),
    ]

    summary = [
        item for item in build_summary(records, {})["panel_counts"]
        if item["model"] == "mlp"
    ]

    assert [(item["num_panels"], item["mean_rmse_mm"]) for item in summary] == [(6, 1.0), (7, 3.0)]


def test_paired_comparison_uses_seed_split_and_graph_id() -> None:
    records = [
        record("mlp", 7, 1, 2, 4, group="low"),
        record("gnn", 7, 1, 2, 2, group="low"),
        record("mlp", 19, 1, 2, 1, group="low"),
        record("gnn", 19, 1, 2, 2, group="low"),
    ]

    result = paired_comparisons(records)["overall"][0]

    assert result["count"] == 2
    assert result["gnn_win_percentage"] == pytest.approx(50.0)
    assert result["mean_rmse_improvement_mm"] == pytest.approx(0.5)


def test_missing_pair_raises_clearly() -> None:
    with pytest.raises(ValueError, match="matching GNN"):
        paired_comparisons([record("mlp", 7, 1, 2, 1)])


def test_output_is_deterministic_and_contains_no_secrets(tmp_path) -> None:
    records = [
        record("gnn", 7, 2, 3, 1),
        record("zero", 42, 1, 2, 2),
        record("mlp", 7, 1, 2, 1),
        record("gnn", 7, 1, 2, 0.5),
        record("mlp", 7, 2, 3, 2),
    ]
    summary = build_summary(records, {"validation": {"q25": 1.0, "q75": 3.0}})
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_paths = write_analysis(records, summary, first)
    second_paths = write_analysis(list(reversed(records)), summary, second)

    for first_path, second_path in zip(first_paths, second_paths):
        assert first_path.read_bytes() == second_path.read_bytes()
        text = first_path.read_text(encoding="utf-8").lower()
        assert "tracking_uri" not in text
        assert "password" not in text
        assert "token" not in text
        assert "http://" not in text


class FakeClient:
    def __init__(self, *, status="FINISHED", model="zero", seed="42", experiment="1352"):
        self.run = SimpleNamespace(
            info=SimpleNamespace(status=status, experiment_id=experiment),
            data=SimpleNamespace(params={"model": model, "seed": seed}),
        )

    def get_run(self, _run_id):
        return self.run


def empty_dataset(normalization) -> LoadedDataset:
    return LoadedDataset({}, normalization, {"seed": 42})


@pytest.mark.parametrize(
    ("client", "message"),
    [
        (FakeClient(status="FAILED"), "not FINISHED"),
        (FakeClient(model="mlp"), "wrong model"),
        (FakeClient(seed="7"), "wrong training seed"),
        (FakeClient(experiment="9"), "not in experiment"),
    ],
)
def test_invalid_mlflow_run_raises_clear_error(client, message, normalization, tmp_path) -> None:
    with pytest.raises(ValueError, match=message):
        load_verified_model(
            client, "run", "zero", 42, "1352", empty_dataset(normalization), tmp_path
        )


def test_zero_model_is_reconstructed_without_training_or_checkpoint(
    normalization, tmp_path
) -> None:
    client = FakeClient()
    model = load_verified_model(
        client, "run", "zero", 42, "1352", empty_dataset(normalization), tmp_path
    )

    assert list(model.parameters()) == []
    assert not hasattr(client, "download_artifacts")


def test_analysis_module_does_not_import_training_functions() -> None:
    import trussgnn.analysis.error_analysis as module

    assert not hasattr(module, "fit_model")
    assert not hasattr(module, "train_one_epoch")

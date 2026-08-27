"""Focused offline tests for the three Phase 5C figures."""

from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

from trussgnn.analysis.plot_results import (
    phase5a_statistics,
    plot_error_analysis,
    plot_model_comparison,
    plot_representative_prediction,
    select_representative_graph,
    undirected_edges,
)
from trussgnn.data import NormalizationStats


class FakeClient:
    def __init__(self, metrics):
        self.metrics = metrics

    def get_run(self, run_id):
        return SimpleNamespace(
            info=SimpleNamespace(status="FINISHED"),
            data=SimpleNamespace(metrics=self.metrics[run_id]),
        )


def test_phase5a_statistics_use_mean_and_sample_standard_deviation() -> None:
    rows = []
    metrics = {}
    for model, seeds in (("zero", [42]), ("mlp", [7, 19, 42]), ("gnn", [7, 19, 42])):
        for index, seed in enumerate(seeds, start=1):
            run_id = f"{model}-{seed}"
            rows.append({"model": model, "training_seed": seed, "run_id": run_id})
            metrics[run_id] = {
                f"{split}/rmse_mm": float(index) for split in (
                    "validation", "iid_test", "geometry_ood", "topology_size_ood"
                )
            }

    result = phase5a_statistics(FakeClient(metrics), rows)

    assert result["zero"]["validation"] == {"mean": 1.0, "sample_std": 0.0}
    assert result["mlp"]["validation"]["mean"] == pytest.approx(2.0)
    assert result["mlp"]["validation"]["sample_std"] == pytest.approx(1.0)


def synthetic_summary() -> dict:
    magnitude = [
        {
            "model": model,
            "split": "iid_test",
            "magnitude_group": group,
            "mean_relative_l2": float(index + model_index),
        }
        for model_index, model in enumerate(("mlp", "gnn"))
        for index, group in enumerate(("low", "medium", "high"), start=1)
    ]
    panels = [
        {
            "model": model,
            "split": "topology_size_ood",
            "num_panels": panel,
            "mean_rmse_mm": 0.1 * panel + model_index,
        }
        for model_index, model in enumerate(("zero", "mlp", "gnn"))
        for panel in (6, 7, 8)
    ]
    return {"magnitude_groups": magnitude, "panel_counts": panels}


def test_plot_functions_create_three_non_empty_png_files(tmp_path) -> None:
    stats = {
        model: {
            split: {"mean": 0.1 + index, "sample_std": 0.01}
            for index, split in enumerate(
                ("validation", "iid_test", "geometry_ood", "topology_size_ood")
            )
        }
        for model in ("zero", "mlp", "gnn")
    }
    model_path = tmp_path / "model_comparison.png"
    error_path = tmp_path / "error_analysis.png"
    prediction_path = tmp_path / "representative_prediction.png"
    plot_model_comparison(stats, model_path)
    plot_error_analysis(synthetic_summary(), error_path)

    graph = Data(
        pos=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=torch.tensor([[0.0, 0.0], [0.001, 0.0]]),
        free_dof_mask=torch.tensor([[False, False], [True, True]]),
    )
    stats_object = NormalizationStats(
        torch.zeros(4), torch.ones(4), torch.zeros(5), torch.ones(5),
        torch.zeros(2), torch.ones(2), "train",
    )
    plot_representative_prediction(
        graph,
        torch.tensor([[0.0, 0.0], [0.0008, 0.0]]),
        stats_object,
        {"graph_id": 8, "rmse_mm": 0.1},
        prediction_path,
    )

    for path in (model_path, error_path, prediction_path):
        assert path.is_file()
        assert path.stat().st_size > 0
        assert path.read_bytes().startswith(b"\x89PNG")


def test_representative_selection_uses_median_error_eight_panel_graph() -> None:
    rows = [
        {"model": "gnn", "training_seed": 42, "split": "topology_size_ood", "num_panels": 8, "graph_id": 1, "rmse_mm": 0.1},
        {"model": "gnn", "training_seed": 42, "split": "topology_size_ood", "num_panels": 8, "graph_id": 2, "rmse_mm": 0.3},
        {"model": "gnn", "training_seed": 42, "split": "topology_size_ood", "num_panels": 8, "graph_id": 3, "rmse_mm": 0.2},
        {"model": "gnn", "training_seed": 42, "split": "topology_size_ood", "num_panels": 7, "graph_id": 4, "rmse_mm": 0.2},
        {"model": "mlp", "training_seed": 42, "split": "topology_size_ood", "num_panels": 8, "graph_id": 5, "rmse_mm": 0.2},
    ]

    selected = select_representative_graph(rows)

    assert selected["graph_id"] == 3


def test_duplicate_directed_edges_are_removed() -> None:
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 1], [1, 0, 2, 1, 1, 2]])

    assert undirected_edges(edge_index) == [(0, 1), (1, 2)]

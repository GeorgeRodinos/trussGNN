"""Create the three concise Phase 5 result figures."""

import argparse
import csv
import json
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlflow import MlflowClient
import numpy as np
import torch

from trussgnn.analysis.error_analysis import load_verified_model
from trussgnn.data import enforce_boundary_conditions, inverse_targets, load_dataset


SPLITS = ("validation", "iid_test", "geometry_ood", "topology_size_ood")
SPLIT_LABELS = ("Validation", "IID", "Geometry OOD", "Topology/size OOD")
MODELS = ("zero", "mlp", "gnn")
EXPERIMENT_ID = "1352"
AMPLIFICATION_FACTOR = 500.0


def read_per_graph_csv(path: str | Path) -> list[dict]:
    """Read the Phase 5B CSV into small typed dictionaries."""

    rows = []
    with Path(path).open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append(
                {
                    **row,
                    "training_seed": int(row["training_seed"]),
                    "graph_id": int(row["graph_id"]),
                    "num_nodes": int(row["num_nodes"]),
                    "num_panels": int(row["num_panels"]),
                    "true_norm_mm": float(row["true_norm_mm"]),
                    "rmse_mm": float(row["rmse_mm"]),
                    "relative_l2": float(row["relative_l2"]),
                }
            )
    return rows


def phase5a_statistics(client: MlflowClient, rows: list[dict]) -> dict:
    """Read split RMSE and calculate learned-model mean and sample deviation."""

    identities = sorted(
        {(row["model"], row["training_seed"], row["run_id"]) for row in rows}
    )
    values = {model: {split: [] for split in SPLITS} for model in MODELS}
    for model, _, run_id in identities:
        run = client.get_run(run_id)
        if run.info.status != "FINISHED":
            raise ValueError(f"MLflow run {run_id} is not FINISHED")
        for split in SPLITS:
            values[model][split].append(run.data.metrics[f"{split}/rmse_mm"])

    return {
        model: {
            split: {
                "mean": statistics.mean(values[model][split]),
                "sample_std": (
                    statistics.stdev(values[model][split])
                    if len(values[model][split]) > 1
                    else 0.0
                ),
            }
            for split in SPLITS
        }
        for model in MODELS
    }


def plot_model_comparison(statistics_by_model: dict, output_path: str | Path) -> None:
    """Plot Phase 5A split-level RMSE with training-seed error bars."""

    x = np.arange(len(SPLITS))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9, 5))
    for index, model in enumerate(MODELS):
        means = [statistics_by_model[model][split]["mean"] for split in SPLITS]
        deviations = [
            statistics_by_model[model][split]["sample_std"] for split in SPLITS
        ]
        axis.bar(
            x + (index - 1) * width,
            means,
            width,
            yerr=None if model == "zero" else deviations,
            capsize=4,
            label=model.upper(),
        )
    axis.set_ylabel("RMSE (mm)")
    axis.set_xticks(x, SPLIT_LABELS)
    axis.set_title("Displacement prediction by evaluation split")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _summary_lookup(items: list[dict], **keys) -> dict:
    return next(item for item in items if all(item[name] == value for name, value in keys.items()))


def plot_error_analysis(summary: dict, output_path: str | Path) -> None:
    """Plot IID relative error by magnitude and topology error by panel count."""

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5))
    groups = ("low", "medium", "high")
    x = np.arange(3)
    width = 0.35
    for index, model in enumerate(("mlp", "gnn")):
        values = [
            _summary_lookup(
                summary["magnitude_groups"],
                model=model,
                split="iid_test",
                magnitude_group=group,
            )["mean_relative_l2"]
            for group in groups
        ]
        left.bar(x + (index - 0.5) * width, values, width, label=model.upper())
    left.set_xticks(x, [name.title() for name in groups])
    left.set_ylabel("Mean graph-relative L2")
    left.set_xlabel("True displacement magnitude")
    left.set_title("IID relative error")
    left.legend()
    left.grid(axis="y", alpha=0.25)

    panels = (6, 7, 8)
    width = 0.25
    for index, model in enumerate(MODELS):
        values = [
            _summary_lookup(
                summary["panel_counts"],
                model=model,
                split="topology_size_ood",
                num_panels=panel,
            )["mean_rmse_mm"]
            for panel in panels
        ]
        right.bar(x + (index - 1) * width, values, width, label=model.upper())
    right.set_xticks(x, panels)
    right.set_xlabel("Number of panels")
    right.set_ylabel("Mean per-graph RMSE (mm)")
    right.set_title("Topology/size OOD error")
    right.legend()
    right.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def select_representative_graph(rows: list[dict]) -> dict:
    """Choose the seed-42 GNN eight-panel graph nearest median RMSE."""

    candidates = [
        row
        for row in rows
        if row["model"] == "gnn"
        and row["training_seed"] == 42
        and row["split"] == "topology_size_ood"
        and row["num_panels"] == 8
    ]
    if not candidates:
        raise ValueError("No seed-42 GNN eight-panel records were found")
    median = statistics.median(row["rmse_mm"] for row in candidates)
    return min(candidates, key=lambda row: (abs(row["rmse_mm"] - median), row["graph_id"]))


def undirected_edges(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Return each physical edge once from bidirectional PyG connectivity."""

    return sorted(
        {
            tuple(sorted((int(first), int(second))))
            for first, second in edge_index.t().tolist()
            if first != second
        }
    )


def plot_representative_prediction(
    graph,
    normalized_prediction: torch.Tensor,
    normalization,
    record: dict,
    output_path: str | Path,
    amplification_factor: float = AMPLIFICATION_FACTOR,
) -> None:
    """Plot original, true displaced, and predicted displaced truss geometry."""

    prediction = enforce_boundary_conditions(
        inverse_targets(normalized_prediction, normalization), graph.free_dof_mask
    )
    target = enforce_boundary_conditions(
        inverse_targets(graph.y, normalization), graph.free_dof_mask
    )
    positions = graph.pos.cpu().numpy()
    true_positions = positions + amplification_factor * target.cpu().numpy()
    predicted_positions = positions + amplification_factor * prediction.cpu().numpy()
    edges = undirected_edges(graph.edge_index)

    fig, axis = plt.subplots(figsize=(9, 4.5))
    for coordinates, color, label, style in (
        (positions, "0.65", "Original", "--"),
        (true_positions, "tab:blue", "True displacement", "-"),
        (predicted_positions, "tab:orange", "GNN prediction", "-"),
    ):
        for edge_index, (first, second) in enumerate(edges):
            axis.plot(
                coordinates[[first, second], 0],
                coordinates[[first, second], 1],
                color=color,
                linestyle=style,
                linewidth=1.5,
                label=label if edge_index == 0 else None,
            )
        axis.scatter(coordinates[:, 0], coordinates[:, 1], color=color, s=16)
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_title(
        f"Graph {record['graph_id']} | RMSE {record['rmse_mm']:.3f} mm | "
        f"displacements ×{amplification_factor:g}"
    )
    axis.legend()
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def generate_figures(
    dataset_dir: str | Path,
    per_graph_csv: str | Path,
    summary_json: str | Path,
    tracking_uri: str,
    output_dir: str | Path,
) -> dict:
    """Read verified results and create exactly three Phase 5 figures."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_per_graph_csv(per_graph_csv)
    summary = json.loads(Path(summary_json).read_text(encoding="utf-8"))
    client = MlflowClient(tracking_uri=tracking_uri)
    plot_model_comparison(
        phase5a_statistics(client, rows), output / "model_comparison.png"
    )
    plot_error_analysis(summary, output / "error_analysis.png")

    record = select_representative_graph(rows)
    dataset = load_dataset(dataset_dir)
    graph = next(
        graph
        for graph in dataset.splits["topology_size_ood"]
        if int(graph.graph_id) == record["graph_id"]
    )
    with TemporaryDirectory() as temporary_directory:
        model = load_verified_model(
            client,
            record["run_id"],
            "gnn",
            42,
            EXPERIMENT_ID,
            dataset,
            Path(temporary_directory),
        )
        model.eval()
        with torch.no_grad():
            prediction = model(graph)
    plot_representative_prediction(
        graph,
        prediction,
        dataset.normalization,
        record,
        output / "representative_prediction.png",
    )
    return record


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--per-graph-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    record = generate_figures(
        args.dataset_dir,
        args.per_graph_csv,
        args.summary_json,
        args.tracking_uri,
        args.output_dir,
    )
    print(f"Created three figures in {args.output_dir}")
    print(f"Representative graph: {record['graph_id']}")
    print(f"Displacement amplification: {AMPLIFICATION_FACTOR:g}")


if __name__ == "__main__":
    main()

"""Analyse verified Phase 5A checkpoints one graph at a time."""

import argparse
import csv
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from mlflow import MlflowClient
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from trussgnn.data import (
    LoadedDataset,
    NormalizationStats,
    enforce_boundary_conditions,
    inverse_targets,
    load_dataset,
)
from trussgnn.models import EdgeAwareGNN, NodeMLP, ZeroDisplacementBaseline


EVALUATION_SPLITS = (
    "validation",
    "iid_test",
    "geometry_ood",
    "topology_size_ood",
)
MODEL_ORDER = {"zero": 0, "mlp": 1, "gnn": 2}


@dataclass(frozen=True)
class PerGraphError:
    """Physical error and graph attributes for one model prediction."""

    model: str
    training_seed: int
    run_id: str
    split: str
    graph_id: int
    num_nodes: int
    num_panels: int
    true_norm_mm: float
    rmse_mm: float
    relative_l2: float
    magnitude_group: str = ""


def per_graph_metrics(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    free_dof_mask: torch.Tensor,
    normalization: NormalizationStats,
    relative_epsilon: float = 1e-12,
) -> tuple[float, float, float]:
    """Return target norm, RMSE in millimetres, and relative L2 for one graph."""

    prediction = inverse_targets(normalized_prediction, normalization)
    target = inverse_targets(normalized_target, normalization)
    prediction = enforce_boundary_conditions(prediction, free_dof_mask)
    target = enforce_boundary_conditions(target, free_dof_mask)
    error = prediction[free_dof_mask] - target[free_dof_mask]
    true_norm = torch.linalg.vector_norm(target[free_dof_mask])
    rmse = error.square().mean().sqrt()
    relative_l2 = torch.linalg.vector_norm(error) / true_norm.clamp_min(relative_epsilon)
    return (
        float((true_norm * 1_000).item()),
        float((rmse * 1_000).item()),
        float(relative_l2.item()),
    )


def evaluate_graphs(
    model: nn.Module,
    graphs: list,
    normalization: NormalizationStats,
    model_name: str,
    training_seed: int,
    run_id: str,
    split: str,
) -> list[PerGraphError]:
    """Evaluate one model on one split using batches of one graph."""

    model.eval()
    records = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=1, shuffle=False):
            prediction = model(batch)
            true_norm_mm, rmse_mm, relative_l2 = per_graph_metrics(
                prediction, batch.y, batch.free_dof_mask, normalization
            )
            records.append(
                PerGraphError(
                    model_name,
                    training_seed,
                    run_id,
                    split,
                    int(batch.graph_id.item()),
                    int(batch.num_nodes),
                    int(batch.num_panels.item()),
                    true_norm_mm,
                    rmse_mm,
                    relative_l2,
                )
            )
    return records


def assign_magnitude_groups(
    records: list[PerGraphError],
) -> tuple[list[PerGraphError], dict[str, dict[str, float]]]:
    """Assign split-level quartile groups shared by every model and seed."""

    graph_norms: dict[tuple[str, int], float] = {}
    for record in records:
        key = (record.split, record.graph_id)
        if key in graph_norms and not np.isclose(graph_norms[key], record.true_norm_mm):
            raise ValueError("A graph has inconsistent target norms across model records")
        graph_norms[key] = record.true_norm_mm

    quartiles = {}
    for split in EVALUATION_SPLITS:
        values = [norm for (name, _), norm in graph_norms.items() if name == split]
        if not values:
            continue
        low, high = np.quantile(values, [0.25, 0.75])
        quartiles[split] = {"q25_true_norm_mm": float(low), "q75_true_norm_mm": float(high)}

    grouped = []
    for record in records:
        limits = quartiles[record.split]
        if record.true_norm_mm <= limits["q25_true_norm_mm"]:
            group = "low"
        elif record.true_norm_mm <= limits["q75_true_norm_mm"]:
            group = "medium"
        else:
            group = "high"
        grouped.append(replace(record, magnitude_group=group))
    return grouped, quartiles


def _mean_summary(records: list[PerGraphError], keys: tuple[str, ...]) -> list[dict]:
    grouped: dict[tuple, list[PerGraphError]] = defaultdict(list)
    for record in records:
        grouped[tuple(getattr(record, key) for key in keys)].append(record)

    results = []
    for values, group in sorted(grouped.items()):
        results.append(
            {
                **dict(zip(keys, values)),
                "count": len(group),
                "mean_rmse_mm": float(np.mean([record.rmse_mm for record in group])),
                "mean_relative_l2": float(
                    np.mean([record.relative_l2 for record in group])
                ),
            }
        )
    return results


def paired_comparisons(records: list[PerGraphError]) -> dict[str, list[dict]]:
    """Compare paired MLP and GNN records overall, by magnitude, and by panel count."""

    pairs: dict[tuple[int, str, int], dict[str, PerGraphError]] = defaultdict(dict)
    for record in records:
        if record.model in {"mlp", "gnn"}:
            pairs[(record.training_seed, record.split, record.graph_id)][record.model] = record
    if any(set(pair) != {"mlp", "gnn"} for pair in pairs.values()):
        raise ValueError("Every MLP record must have a matching GNN record")

    def summarize(group_keys: tuple[str, ...]) -> list[dict]:
        groups: dict[tuple, list[tuple[PerGraphError, PerGraphError]]] = defaultdict(list)
        for pair in pairs.values():
            mlp, gnn = pair["mlp"], pair["gnn"]
            key = tuple(getattr(mlp, name) for name in group_keys)
            groups[key].append((mlp, gnn))
        return [
            {
                **dict(zip(group_keys, key)),
                "count": len(values),
                "gnn_win_percentage": 100.0
                * sum(gnn.rmse_mm < mlp.rmse_mm for mlp, gnn in values)
                / len(values),
                "mean_rmse_improvement_mm": float(
                    np.mean([mlp.rmse_mm - gnn.rmse_mm for mlp, gnn in values])
                ),
            }
            for key, values in sorted(groups.items())
        ]

    return {
        "overall": summarize(("split",)),
        "by_magnitude_group": summarize(("split", "magnitude_group")),
        "by_panel_count": summarize(("split", "num_panels")),
    }


def build_summary(records: list[PerGraphError], quartiles: dict) -> dict:
    """Build the concise JSON-ready grouped analysis."""

    return {
        "quartiles_by_split": quartiles,
        "model_and_split": _mean_summary(records, ("model", "split")),
        "magnitude_groups": _mean_summary(
            records, ("model", "split", "magnitude_group")
        ),
        "panel_counts": _mean_summary(records, ("model", "split", "num_panels")),
        "paired_gnn_vs_mlp": paired_comparisons(records),
    }


def write_analysis(
    records: list[PerGraphError], summary: dict, output_dir: str | Path
) -> tuple[Path, Path]:
    """Write deterministically ordered CSV and JSON analysis files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    split_order = {name: index for index, name in enumerate(EVALUATION_SPLITS)}
    ordered = sorted(
        records,
        key=lambda record: (
            MODEL_ORDER[record.model],
            record.training_seed,
            split_order[record.split],
            record.graph_id,
        ),
    )
    csv_path = output / "phase5b_per_graph.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(ordered[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in ordered)

    json_path = output / "phase5b_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def load_verified_model(
    client: MlflowClient,
    run_id: str,
    expected_model: str,
    expected_seed: int,
    experiment_id: str,
    dataset: LoadedDataset,
    checkpoint_dir: Path,
) -> nn.Module:
    """Verify one Phase 5A run and reconstruct its accepted model."""

    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise ValueError(f"MLflow run {run_id} is not FINISHED")
    if run.info.experiment_id != experiment_id:
        raise ValueError(f"MLflow run {run_id} is not in experiment {experiment_id}")
    if run.data.params.get("model") != expected_model:
        raise ValueError(f"MLflow run {run_id} has the wrong model")
    if run.data.params.get("seed") != str(expected_seed):
        raise ValueError(f"MLflow run {run_id} has the wrong training seed")

    if expected_model == "zero":
        stats = dataset.normalization
        return ZeroDisplacementBaseline(stats.target_mean, stats.target_std)

    hidden_dim = int(run.data.params["hidden_dim"])
    layer_count = int(run.data.params["layer_count"])
    dropout = float(run.data.params["dropout"])
    model = (
        NodeMLP(hidden_dim, layer_count, dropout)
        if expected_model == "mlp"
        else EdgeAwareGNN(hidden_dim, layer_count, dropout)
    )
    checkpoint_path = client.download_artifacts(
        run_id, "best_checkpoint.pt", dst_path=str(checkpoint_dir)
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def run_error_analysis(
    dataset_dir: str | Path,
    tracking_uri: str,
    output_dir: str | Path,
    zero_run_id: str,
    mlp_run_ids: Sequence[str],
    gnn_run_ids: Sequence[str],
    experiment_id: str = "1352",
) -> tuple[Path, Path, list[PerGraphError], dict]:
    """Verify seven runs, evaluate every graph, and write the Phase 5B analysis."""

    if len(mlp_run_ids) != 3 or len(gnn_run_ids) != 3:
        raise ValueError("Exactly three MLP and three GNN run IDs are required")
    dataset = load_dataset(dataset_dir)
    client = MlflowClient(tracking_uri=tracking_uri)
    run_specs = [("zero", 42, zero_run_id)]
    run_specs += [("mlp", seed, run_id) for seed, run_id in zip((7, 19, 42), mlp_run_ids)]
    run_specs += [("gnn", seed, run_id) for seed, run_id in zip((7, 19, 42), gnn_run_ids)]

    records = []
    with TemporaryDirectory() as temporary_directory:
        checkpoint_dir = Path(temporary_directory)
        for model_name, seed, run_id in run_specs:
            model = load_verified_model(
                client, run_id, model_name, seed, experiment_id, dataset, checkpoint_dir
            )
            for split in EVALUATION_SPLITS:
                records.extend(
                    evaluate_graphs(
                        model,
                        dataset.splits[split],
                        dataset.normalization,
                        model_name,
                        seed,
                        run_id,
                        split,
                    )
                )

    records, quartiles = assign_magnitude_groups(records)
    summary = build_summary(records, quartiles)
    csv_path, json_path = write_analysis(records, summary, output_dir)
    return csv_path, json_path, records, summary


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", default="1352")
    parser.add_argument("--zero-run-id", required=True)
    parser.add_argument("--mlp-run-ids", nargs=3, required=True)
    parser.add_argument("--gnn-run-ids", nargs=3, required=True)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    args = parse_args(arguments)
    csv_path, json_path, records, _ = run_error_analysis(
        args.dataset_dir,
        args.tracking_uri,
        args.output_dir,
        args.zero_run_id,
        args.mlp_run_ids,
        args.gnn_run_ids,
        args.experiment_id,
    )
    print(f"Saved {len(records)} per-graph rows to {csv_path}")
    print(f"Saved summary to {json_path}")


if __name__ == "__main__":
    main()

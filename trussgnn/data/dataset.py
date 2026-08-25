"""PyTorch Geometric conversion, statistics, and persistence for Phase 3."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from .generation import GeneratedSample, GenerationConfig, generate_samples


@dataclass
class DatasetBundle:
    """Generated graph splits and their human-readable audit information."""

    splits: dict[str, list[Data]]
    normalization: dict[str, object]
    metadata: dict[str, object]
    config: GenerationConfig


def sample_to_data(sample: GeneratedSample) -> Data:
    """Convert one solved physical truss to a validated raw-SI PyG graph."""

    node_rows = [
        [node.x, node.y, node.fx, node.fy, float(node.fixed_x), float(node.fixed_y)]
        for node in sample.truss.nodes
    ]
    directed_edges: list[list[int]] = []
    edge_rows: list[list[float]] = []

    for edge in sample.truss.edges:
        first = sample.truss.nodes[edge.node_i]
        second = sample.truss.nodes[edge.node_j]
        dx = second.x - first.x
        dy = second.y - first.y
        length = float(np.hypot(dx, dy))
        cosine = dx / length
        sine = dy / length

        directed_edges.extend([[edge.node_i, edge.node_j], [edge.node_j, edge.node_i]])
        edge_rows.extend(
            [
                [length, cosine, sine, edge.E, edge.A],
                [length, -cosine, -sine, edge.E, edge.A],
            ]
        )

    data = Data(
        x=torch.tensor(node_rows, dtype=torch.float32),
        pos=torch.tensor([row[:2] for row in node_rows], dtype=torch.float32),
        edge_index=torch.tensor(directed_edges, dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(edge_rows, dtype=torch.float32),
        y=torch.tensor(sample.solution.displacements, dtype=torch.float32),
        graph_id=torch.tensor(sample.graph_id, dtype=torch.long),
        base_id=torch.tensor(sample.base_id, dtype=torch.long),
        num_panels=torch.tensor(sample.num_panels, dtype=torch.long),
        panel_widths=torch.tensor(sample.panel_widths, dtype=torch.float32),
        top_heights=torch.tensor(sample.top_heights, dtype=torch.float32),
        condition_number=torch.tensor(sample.condition_number, dtype=torch.float64),
    )
    data.num_nodes = len(sample.truss.nodes)
    data.validate(raise_on_error=True)
    return data


def _statistics(values: torch.Tensor) -> dict[str, list[float]]:
    values = values.to(torch.float64)
    return {
        "mean": values.mean(dim=0).tolist(),
        "std": values.std(dim=0, unbiased=False).tolist(),
    }


def training_statistics(training_graphs: list[Data]) -> dict[str, object]:
    """Calculate raw feature/target statistics from training graphs only."""

    if not training_graphs:
        raise ValueError("At least one training graph is required for normalization statistics")
    nodes = torch.cat([graph.x[:, :4] for graph in training_graphs], dim=0)
    edges = torch.cat([graph.edge_attr for graph in training_graphs], dim=0)
    targets = torch.cat([graph.y for graph in training_graphs], dim=0)
    return {
        "source_split": "train",
        "zero_std_fallback": 1.0,
        "node_features": {"names": ["x", "y", "fx", "fy"], **_statistics(nodes)},
        "edge_features": {
            "names": ["length", "cos_theta", "sin_theta", "E", "A"],
            **_statistics(edges),
        },
        "targets": {"names": ["ux", "uy"], **_statistics(targets)},
    }


def generate_dataset(config: GenerationConfig | None = None) -> DatasetBundle:
    """Generate, solve, validate, convert, and summarize all five splits."""

    config = config or GenerationConfig()
    physical_splits, metadata = generate_samples(config)
    graph_splits = {
        name: [sample_to_data(sample) for sample in samples]
        for name, samples in physical_splits.items()
    }
    normalization = training_statistics(graph_splits["train"])
    return DatasetBundle(graph_splits, normalization, metadata, config)


def save_dataset(bundle: DatasetBundle, output_directory: str | Path) -> None:
    """Save one tensor file per split plus JSON normalization and metadata."""

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, graphs in bundle.splits.items():
        torch.save(graphs, output / f"{name}.pt")

    (output / "normalization.json").write_text(
        json.dumps(bundle.normalization, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {**bundle.metadata, "config": asdict(bundle.config)}
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def load_split(output_directory: str | Path, split: str) -> list[Data]:
    """Load one named split without depending on generation internals."""

    path = Path(output_directory) / f"{split}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split does not exist: {path}")
    graphs = torch.load(path, weights_only=False)
    for graph in graphs:
        graph.validate(raise_on_error=True)
    return graphs

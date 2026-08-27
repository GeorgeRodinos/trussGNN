"""Load, normalize, batch, and identify an accepted Phase 3 dataset."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .dataset import load_split
from .generation import SPLIT_NAMES


DATASET_FILES = tuple(f"{name}.pt" for name in SPLIT_NAMES) + (
    "normalization.json",
    "metadata.json",
)


@dataclass(frozen=True)
class NormalizationStats:
    """Training-only means and standard deviations for stored continuous tensors."""

    node_mean: torch.Tensor
    node_std: torch.Tensor
    edge_mean: torch.Tensor
    edge_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor
    source_split: str

    @staticmethod
    def safe_std(std: torch.Tensor) -> torch.Tensor:
        """Replace zero standard deviations with one."""

        return torch.where(std == 0, torch.ones_like(std), std)


@dataclass(frozen=True)
class LoadedDataset:
    """Prepared graph splits, metadata, and training statistics."""

    splits: dict[str, list[Data]]
    normalization: NormalizationStats
    metadata: dict[str, object]


def _require_dataset_files(directory: Path) -> None:
    missing = [name for name in DATASET_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")


def _tensor(values: object, expected_size: int, label: str) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.shape != (expected_size,):
        raise ValueError(f"{label} must contain {expected_size} values")
    return tensor


def load_normalization(directory: str | Path) -> NormalizationStats:
    """Load and validate the training-only Phase 3 normalization JSON."""

    path = Path(directory) / "normalization.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path.name}")
    values = json.loads(path.read_text(encoding="utf-8"))
    if values.get("source_split") != "train":
        raise ValueError("normalization.json must use source_split='train'")

    return NormalizationStats(
        node_mean=_tensor(values["node_features"]["mean"], 4, "node mean"),
        node_std=_tensor(values["node_features"]["std"], 4, "node std"),
        edge_mean=_tensor(values["edge_features"]["mean"], 5, "edge mean"),
        edge_std=_tensor(values["edge_features"]["std"], 5, "edge std"),
        target_mean=_tensor(values["targets"]["mean"], 2, "target mean"),
        target_std=_tensor(values["targets"]["std"], 2, "target std"),
        source_split="train",
    )


def prepare_graph(graph: Data, stats: NormalizationStats) -> Data:
    """Clone one raw graph, normalize continuous tensors, and attach its free-DOF mask."""

    prepared = graph.clone()
    support_flags = prepared.x[:, 4:6]
    if not torch.all((support_flags == 0) | (support_flags == 1)):
        raise ValueError("Support flags must be exactly 0.0 or 1.0")

    node_values = prepared.x[:, :4]
    node_mean = stats.node_mean.to(node_values)
    node_std = stats.safe_std(stats.node_std).to(node_values)
    prepared.x[:, :4] = (node_values - node_mean) / node_std

    edge_mean = stats.edge_mean.to(prepared.edge_attr)
    edge_std = stats.safe_std(stats.edge_std).to(prepared.edge_attr)
    prepared.edge_attr = (prepared.edge_attr - edge_mean) / edge_std

    target_mean = stats.target_mean.to(prepared.y)
    target_std = stats.safe_std(stats.target_std).to(prepared.y)
    prepared.y = (prepared.y - target_mean) / target_std
    prepared.free_dof_mask = support_flags == 0
    prepared.validate(raise_on_error=True)
    return prepared


def load_dataset(directory: str | Path) -> LoadedDataset:
    """Load and validate all five raw splits, then prepare normalized clones."""

    directory = Path(directory)
    _require_dataset_files(directory)
    normalization = load_normalization(directory)
    raw_splits = {name: load_split(directory, name) for name in SPLIT_NAMES}
    splits = {
        name: [prepare_graph(graph, normalization) for graph in graphs]
        for name, graphs in raw_splits.items()
    }
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    return LoadedDataset(splits, normalization, metadata)


def inverse_targets(values: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    """Convert normalized displacements back to physical metres."""

    target_mean = stats.target_mean.to(values)
    target_std = stats.safe_std(stats.target_std).to(values)
    return values * target_std + target_mean


def enforce_boundary_conditions(
    physical_predictions: torch.Tensor, free_dof_mask: torch.Tensor
) -> torch.Tensor:
    """Return physical predictions with constrained components set exactly to zero."""

    if physical_predictions.shape != free_dof_mask.shape:
        raise ValueError("Predictions and free_dof_mask must have the same shape")
    if free_dof_mask.dtype != torch.bool:
        raise ValueError("free_dof_mask must have dtype torch.bool")
    return torch.where(free_dof_mask, physical_predictions, torch.zeros_like(physical_predictions))


def create_data_loaders(
    splits: dict[str, list[Data]],
    batch_size: int,
    seed: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Create a seeded shuffled training loader and ordered evaluation loaders."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    loaders: dict[str, DataLoader] = {}
    for name in SPLIT_NAMES:
        if name not in splits:
            raise ValueError(f"Missing prepared split: {name}")
        generator = torch.Generator().manual_seed(seed) if name == "train" else None
        loaders[name] = DataLoader(
            splits[name],
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            generator=generator,
        )
    return loaders

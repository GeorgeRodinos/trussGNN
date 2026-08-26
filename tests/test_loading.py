"""Verification of Phase 4B loading, normalization, batching, and identity."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from trussgnn.data.dataset import generate_dataset, save_dataset
from trussgnn.data.generation import SPLIT_NAMES, GenerationConfig
from trussgnn.data.loading import (
    DATASET_FILES,
    build_dataset_manifest,
    create_data_loaders,
    enforce_boundary_conditions,
    inverse_targets,
    load_dataset,
    load_normalization,
    prepare_graph,
)


@pytest.fixture(scope="module")
def dataset_directory(tmp_path_factory) -> Path:
    """Generate one tiny temporary Phase 3 dataset for all loading tests."""

    directory = tmp_path_factory.mktemp("phase4b-dataset")
    counts = {name: (4 if name == "train" else 2) for name in SPLIT_NAMES}
    bundle = generate_dataset(GenerationConfig(seed=31, split_counts=counts))
    save_dataset(bundle, directory)
    return directory


@pytest.fixture(scope="module")
def loaded(dataset_directory):
    return load_dataset(dataset_directory)


def graph_order(loader) -> list[int]:
    return [int(graph_id) for batch in loader for graph_id in batch.graph_id]


def test_all_splits_load_with_expected_counts_and_validate(loaded) -> None:
    expected = {name: (4 if name == "train" else 2) for name in SPLIT_NAMES}

    assert {name: len(graphs) for name, graphs in loaded.raw_splits.items()} == expected
    for graphs in loaded.raw_splits.values():
        for graph in graphs:
            assert graph.validate(raise_on_error=True)


def test_continuous_nodes_and_support_flags_are_prepared_correctly(loaded) -> None:
    raw = loaded.raw_splits["train"][0]
    prepared = loaded.splits["train"][0]
    stats = loaded.normalization
    expected = (raw.x[:, :4] - stats.node_mean) / stats.safe_std(stats.node_std)

    assert torch.allclose(prepared.x[:, :4], expected)
    assert torch.equal(prepared.x[:, 4:6], raw.x[:, 4:6])
    assert torch.all((prepared.x[:, 4:6] == 0) | (prepared.x[:, 4:6] == 1))


def test_edge_features_and_targets_match_direct_normalization(loaded) -> None:
    raw = loaded.raw_splits["train"][0]
    prepared = loaded.splits["train"][0]
    stats = loaded.normalization

    expected_edges = (raw.edge_attr - stats.edge_mean) / stats.safe_std(stats.edge_std)
    expected_targets = (raw.y - stats.target_mean) / stats.safe_std(stats.target_std)
    assert torch.allclose(prepared.edge_attr, expected_edges)
    assert torch.allclose(prepared.y, expected_targets)


def test_positions_and_graph_metadata_remain_raw(loaded) -> None:
    raw = loaded.raw_splits["train"][0]
    prepared = loaded.splits["train"][0]

    for name in (
        "pos",
        "graph_id",
        "base_id",
        "num_panels",
        "panel_widths",
        "top_heights",
        "condition_number",
    ):
        assert torch.equal(prepared[name], raw[name])


def test_inverse_targets_recovers_raw_displacements(loaded) -> None:
    prepared = loaded.splits["train"][0]
    raw = loaded.raw_splits["train"][0]

    recovered = inverse_targets(prepared.y, loaded.normalization)
    assert torch.allclose(recovered, raw.y)


def test_zero_standard_deviation_fallback_is_finite(loaded) -> None:
    stats = replace(
        loaded.normalization,
        node_std=torch.zeros(4),
        edge_std=torch.zeros(5),
        target_std=torch.zeros(2),
    )

    prepared = prepare_graph(loaded.raw_splits["train"][0], stats)
    assert torch.isfinite(prepared.x).all()
    assert torch.isfinite(prepared.edge_attr).all()
    assert torch.isfinite(prepared.y).all()


def test_preparation_does_not_modify_raw_graph(loaded) -> None:
    raw = loaded.raw_splits["train"][0]
    before = {name: value.clone() for name, value in raw if isinstance(value, torch.Tensor)}

    prepare_graph(raw, loaded.normalization)

    for name, value in before.items():
        assert torch.equal(raw[name], value)


def test_free_dof_mask_has_expected_shape_dtype_and_values(loaded) -> None:
    raw = loaded.raw_splits["train"][0]
    prepared = loaded.splits["train"][0]

    assert prepared.free_dof_mask.shape == (raw.num_nodes, 2)
    assert prepared.free_dof_mask.dtype == torch.bool
    assert torch.equal(prepared.free_dof_mask, raw.x[:, 4:6] == 0)


def test_boundary_conditions_zero_constraints_without_mutating_input(loaded) -> None:
    mask = loaded.splits["train"][0].free_dof_mask
    predictions = torch.arange(mask.numel(), dtype=torch.float32).reshape(mask.shape) + 1
    before = predictions.clone()

    constrained = enforce_boundary_conditions(predictions, mask)

    assert torch.equal(predictions, before)
    assert torch.equal(constrained[mask], predictions[mask])
    assert torch.count_nonzero(constrained[~mask]) == 0


def test_multi_graph_batch_is_valid(loaded) -> None:
    loaders = create_data_loaders(loaded.splits, batch_size=2, seed=9)
    batch = next(iter(loaders["train"]))

    assert batch.num_graphs == 2
    assert batch.validate(raise_on_error=True)
    assert batch.free_dof_mask.shape == batch.y.shape


def test_training_loader_order_is_reproducible(loaded) -> None:
    first = create_data_loaders(loaded.splits, batch_size=2, seed=123)["train"]
    second = create_data_loaders(loaded.splits, batch_size=2, seed=123)["train"]

    assert graph_order(first) == graph_order(second)


def test_evaluation_loader_order_is_unchanged_and_deterministic(loaded) -> None:
    first = create_data_loaders(loaded.splits, batch_size=2, seed=1)
    second = create_data_loaders(loaded.splits, batch_size=1, seed=999)

    for name in SPLIT_NAMES[1:]:
        expected = [int(graph.graph_id) for graph in loaded.splits[name]]
        assert graph_order(first[name]) == expected
        assert graph_order(second[name]) == expected


def test_manifest_file_identity_and_dataset_information(dataset_directory) -> None:
    manifest = build_dataset_manifest(dataset_directory)
    entries = {entry["filename"]: entry for entry in manifest["files"]}

    assert set(entries) == set(DATASET_FILES)
    for filename, entry in entries.items():
        content = (dataset_directory / filename).read_bytes()
        assert entry["size_bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    assert manifest["dataset_seed"] == 31
    assert manifest["split_counts"] == {name: (4 if name == "train" else 2) for name in SPLIT_NAMES}
    assert manifest["normalization_source_split"] == "train"


def test_manifest_is_deterministic(dataset_directory) -> None:
    assert build_dataset_manifest(dataset_directory) == build_dataset_manifest(dataset_directory)


def test_normalization_matches_phase3_training_statistics(loaded) -> None:
    training_nodes = torch.cat([graph.x[:, :4] for graph in loaded.raw_splits["train"]])
    stats = loaded.normalization

    assert stats.source_split == "train"
    assert stats.node_mean == pytest.approx(training_nodes.double().mean(0).tolist())
    assert stats.node_std == pytest.approx(
        training_nodes.double().std(0, unbiased=False).tolist()
    )


def test_missing_dataset_files_raise_clear_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing dataset files"):
        load_dataset(tmp_path)


def test_non_training_normalization_source_is_rejected(tmp_path) -> None:
    values = {"source_split": "validation"}
    (tmp_path / "normalization.json").write_text(json.dumps(values), encoding="utf-8")

    with pytest.raises(ValueError, match="source_split='train'"):
        load_normalization(tmp_path)

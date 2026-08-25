"""Focused verification of the deterministic Phase 3 dataset pipeline."""

import copy

import numpy as np
import pytest
import torch

from trussgnn.data.dataset import (
    generate_dataset,
    load_split,
    sample_to_data,
    save_dataset,
    training_statistics,
)
from trussgnn.data.generation import (
    SPLIT_NAMES,
    GenerationConfig,
    build_triangular_chain,
    generate_samples,
)


def tiny_config(seed: int = 17) -> GenerationConfig:
    """Use two graphs per split so tests never create the full dataset."""

    return GenerationConfig(seed=seed, split_counts={name: 2 for name in SPLIT_NAMES})


def assert_same_graph(first, second) -> None:
    assert set(first.keys()) == set(second.keys())
    for key in first.keys():
        first_value = first[key]
        second_value = second[key]
        if isinstance(first_value, torch.Tensor):
            assert torch.equal(first_value, second_value)
        else:
            assert first_value == second_value


@pytest.mark.parametrize("num_panels", [2, 4])
def test_truss_construction_counts_supports_and_load(num_panels: int) -> None:
    truss = build_triangular_chain(
        panel_widths=np.ones(num_panels),
        top_heights=np.ones(num_panels),
        loaded_top_node=1,
        fx=100.0,
        fy=-1_000.0,
        youngs_modulus=200e9,
        edge_areas=np.full(4 * num_panels - 1, 1e-3),
    )

    assert len(truss.nodes) == 2 * num_panels + 1
    assert len(truss.edges) == 4 * num_panels - 1
    assert truss.nodes[0].fixed_x and truss.nodes[0].fixed_y
    assert truss.nodes[num_panels].fixed_y
    assert not truss.nodes[num_panels].fixed_x
    loaded = [node for node in truss.nodes[num_panels + 1 :] if node.fx != 0 or node.fy != 0]
    assert len(loaded) == 1


def test_generation_is_reproducible() -> None:
    first = generate_dataset(tiny_config(seed=4))
    second = generate_dataset(tiny_config(seed=4))

    assert first.metadata == second.metadata
    assert first.normalization == second.normalization
    for split in SPLIT_NAMES:
        for first_graph, second_graph in zip(first.splits[split], second.splits[split]):
            assert_same_graph(first_graph, second_graph)


def test_graph_conversion_shapes_dtypes_and_validation() -> None:
    bundle = generate_dataset(tiny_config())
    graph = bundle.splits["train"][0]
    panels = int(graph.num_panels)
    physical_edges = 4 * panels - 1

    assert graph.x.shape == (2 * panels + 1, 6)
    assert graph.pos.shape == (2 * panels + 1, 2)
    assert graph.edge_index.shape == (2, 2 * physical_edges)
    assert graph.edge_attr.shape == (2 * physical_edges, 5)
    assert graph.y.shape == (2 * panels + 1, 2)
    assert graph.x.dtype == graph.pos.dtype == graph.edge_attr.dtype == graph.y.dtype == torch.float32
    assert graph.edge_index.dtype == torch.long
    assert graph.validate(raise_on_error=True)


def test_every_physical_edge_has_correct_reverse_features() -> None:
    graph = generate_dataset(tiny_config()).splits["train"][0]

    for column in range(0, graph.edge_index.shape[1], 2):
        forward = graph.edge_index[:, column]
        reverse = graph.edge_index[:, column + 1]
        forward_features = graph.edge_attr[column]
        reverse_features = graph.edge_attr[column + 1]

        assert torch.equal(forward, reverse.flip(0))
        assert torch.equal(forward_features[[0, 3, 4]], reverse_features[[0, 3, 4]])
        assert torch.equal(forward_features[[1, 2]], -reverse_features[[1, 2]])


def test_graph_targets_equal_fem_solution_and_residual_is_small() -> None:
    physical_splits, _ = generate_samples(tiny_config())
    sample = physical_splits["train"][0]
    graph = sample_to_data(sample)

    assert graph.y.numpy() == pytest.approx(sample.solution.displacements)
    assert sample.solution.normalized_free_residual() < 1e-10


def test_split_base_ids_and_parameter_ranges_are_disjoint() -> None:
    config = tiny_config()
    bundle = generate_dataset(config)
    seen_base_ids: set[int] = set()

    for split, graphs in bundle.splits.items():
        split_ids = {int(graph.base_id) for graph in graphs}
        assert seen_base_ids.isdisjoint(split_ids)
        seen_base_ids.update(split_ids)

        for graph in graphs:
            panels = int(graph.num_panels)
            widths = graph.panel_widths.numpy()
            heights = graph.top_heights.numpy()
            if split == "topology_size_ood":
                assert config.topology_ood_panels[0] <= panels <= config.topology_ood_panels[1]
                assert np.all((widths >= config.train_panel_width[0]) & (widths <= config.train_panel_width[1]))
                assert np.all((heights >= config.train_height[0]) & (heights <= config.train_height[1]))
            elif split == "geometry_ood":
                assert config.train_panels[0] <= panels <= config.train_panels[1]
                assert np.all(widths >= config.geometry_ood_panel_width[0])
                assert np.all(heights >= config.geometry_ood_height[0])
            else:
                assert config.train_panels[0] <= panels <= config.train_panels[1]
                assert np.all((widths >= config.train_panel_width[0]) & (widths <= config.train_panel_width[1]))
                assert np.all((heights >= config.train_height[0]) & (heights <= config.train_height[1]))


def test_statistics_use_training_split_only_and_match_direct_calculation() -> None:
    bundle = generate_dataset(tiny_config())
    statistics = bundle.normalization
    training_nodes = torch.cat([graph.x[:, :4] for graph in bundle.splits["train"]]).double()

    assert statistics["source_split"] == "train"
    assert statistics["node_features"]["mean"] == pytest.approx(training_nodes.mean(0).tolist())
    assert statistics["node_features"]["std"] == pytest.approx(
        training_nodes.std(0, unbiased=False).tolist()
    )

    before = copy.deepcopy(statistics)
    bundle.splits["iid_test"][0].x += 1_000_000.0
    after = training_statistics(bundle.splits["train"])
    assert after == before


def test_saved_split_round_trip_preserves_tensors_and_metadata(tmp_path) -> None:
    bundle = generate_dataset(tiny_config())
    save_dataset(bundle, tmp_path)
    loaded = load_split(tmp_path, "train")

    assert len(loaded) == len(bundle.splits["train"])
    for original, restored in zip(bundle.splits["train"], loaded):
        assert_same_graph(original, restored)
    assert (tmp_path / "normalization.json").is_file()
    assert (tmp_path / "metadata.json").is_file()

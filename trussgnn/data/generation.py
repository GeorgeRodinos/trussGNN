"""Deterministic generation and FEM validation of triangular-chain trusses."""

from dataclasses import dataclass, field

import numpy as np

from trussgnn.fem import (
    Edge,
    InvalidEdgeError,
    Node,
    Solution,
    Truss,
    UnstableStructureError,
    solve_truss,
)


SPLIT_NAMES = (
    "train",
    "validation",
    "iid_test",
    "geometry_ood",
    "topology_size_ood",
)

REJECTION_REASONS = (
    "self_loop",
    "duplicate_edge",
    "non_finite_input",
    "non_positive_edge_length",
    "invalid_edge",
    "unstable_structure",
    "non_finite_displacement",
    "equilibrium_residual",
    "large_displacement",
    "condition_number",
)


@dataclass
class GenerationConfig:
    """All adjustable ranges and counts for deterministic dataset generation."""

    seed: int = 42
    split_counts: dict[str, int] = field(
        default_factory=lambda: {
            "train": 1200,
            "validation": 200,
            "iid_test": 200,
            "geometry_ood": 200,
            "topology_size_ood": 200,
        }
    )
    train_panels: tuple[int, int] = (2, 5)
    topology_ood_panels: tuple[int, int] = (6, 8)
    train_panel_width: tuple[float, float] = (0.8, 1.5)
    geometry_ood_panel_width: tuple[float, float] = (1.6, 2.2)
    train_height: tuple[float, float] = (0.8, 1.5)
    geometry_ood_height: tuple[float, float] = (1.6, 2.2)
    horizontal_load: tuple[float, float] = (-2_000.0, 2_000.0)
    vertical_load: tuple[float, float] = (-10_000.0, -1_000.0)
    youngs_modulus: tuple[float, float] = (70e9, 210e9)
    edge_area: tuple[float, float] = (5e-4, 2e-3)
    geometry_perturbation: float = 0.05
    variations_per_base: int = 2
    residual_limit: float = 1e-10
    displacement_span_limit: float = 0.01
    condition_number_limit: float = 1e12
    max_attempts_per_graph: int = 100

    def __post_init__(self) -> None:
        if set(self.split_counts) != set(SPLIT_NAMES):
            raise ValueError(f"split_counts must contain exactly: {', '.join(SPLIT_NAMES)}")
        if any(count < 0 for count in self.split_counts.values()):
            raise ValueError("Split counts cannot be negative")
        if self.variations_per_base < 1 or self.max_attempts_per_graph < 1:
            raise ValueError("variations_per_base and max_attempts_per_graph must be positive")


@dataclass
class BaseStructure:
    """Geometry and topology assigned to one split before variations are made."""

    base_id: int
    num_panels: int
    panel_widths: np.ndarray
    height: float


@dataclass
class GeneratedSample:
    """A solved physical truss and its audit metadata."""

    truss: Truss
    solution: Solution
    graph_id: int
    base_id: int
    num_panels: int
    panel_widths: np.ndarray
    top_heights: np.ndarray
    condition_number: float


def build_triangular_chain(
    panel_widths: np.ndarray,
    top_heights: np.ndarray,
    loaded_top_node: int,
    fx: float,
    fy: float,
    youngs_modulus: float,
    edge_areas: np.ndarray,
) -> Truss:
    """Build one triangular-chain bridge from explicit physical parameters."""

    num_panels = len(panel_widths)
    if num_panels < 2 or len(top_heights) != num_panels:
        raise ValueError("A triangular chain needs matching widths/heights for at least two panels")
    if loaded_top_node < 0 or loaded_top_node >= num_panels:
        raise ValueError("loaded_top_node is outside the top-node range")
    if len(edge_areas) != 4 * num_panels - 1:
        raise ValueError("edge_areas must contain one value per physical edge")

    bottom_x = np.concatenate(([0.0], np.cumsum(panel_widths)))
    nodes = [Node(float(x), 0.0) for x in bottom_x]

    for panel in range(num_panels):
        top_x = (bottom_x[panel] + bottom_x[panel + 1]) / 2.0
        load_x = fx if panel == loaded_top_node else 0.0
        load_y = fy if panel == loaded_top_node else 0.0
        nodes.append(Node(float(top_x), float(top_heights[panel]), load_x, load_y))

    nodes[0].fixed_x = True
    nodes[0].fixed_y = True
    nodes[num_panels].fixed_y = True

    endpoint_pairs: list[tuple[int, int]] = []
    endpoint_pairs.extend((panel, panel + 1) for panel in range(num_panels))
    endpoint_pairs.extend(
        (num_panels + 1 + panel, num_panels + 2 + panel)
        for panel in range(num_panels - 1)
    )
    for panel in range(num_panels):
        top = num_panels + 1 + panel
        endpoint_pairs.extend(((panel, top), (top, panel + 1)))

    edges = [
        Edge(i, j, float(youngs_modulus), float(area))
        for (i, j), area in zip(endpoint_pairs, edge_areas)
    ]
    return Truss(nodes, edges)


def _ranges_for_split(
    split: str, config: GenerationConfig
) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float]]:
    if split == "topology_size_ood":
        return config.topology_ood_panels, config.train_panel_width, config.train_height
    if split == "geometry_ood":
        return config.train_panels, config.geometry_ood_panel_width, config.geometry_ood_height
    return config.train_panels, config.train_panel_width, config.train_height


def _new_base(
    base_id: int, split: str, config: GenerationConfig, rng: np.random.Generator
) -> BaseStructure:
    panel_range, width_range, height_range = _ranges_for_split(split, config)
    num_panels = int(rng.integers(panel_range[0], panel_range[1] + 1))
    widths = rng.uniform(*width_range, size=num_panels)
    height = float(rng.uniform(*height_range))
    return BaseStructure(base_id, num_panels, widths, height)


def _variation(
    base: BaseStructure, split: str, config: GenerationConfig, rng: np.random.Generator
) -> Truss:
    _, width_range, height_range = _ranges_for_split(split, config)
    change = config.geometry_perturbation
    widths = base.panel_widths * rng.uniform(1.0 - change, 1.0 + change, base.num_panels)
    widths = np.clip(widths, *width_range)
    heights = base.height * rng.uniform(1.0 - change, 1.0 + change, base.num_panels)
    heights = np.clip(heights, *height_range)

    edge_count = 4 * base.num_panels - 1
    fy = float(rng.uniform(*config.vertical_load))
    fx_low = max(config.horizontal_load[0], -0.5 * abs(fy))
    fx_high = min(config.horizontal_load[1], 0.5 * abs(fy))
    return build_triangular_chain(
        panel_widths=widths,
        top_heights=heights,
        loaded_top_node=int(rng.integers(base.num_panels)),
        fx=float(rng.uniform(fx_low, fx_high)),
        fy=fy,
        youngs_modulus=float(rng.uniform(*config.youngs_modulus)),
        edge_areas=rng.uniform(*config.edge_area, size=edge_count),
    )


def _validate_candidate(
    truss: Truss, config: GenerationConfig
) -> tuple[Solution | None, float | None, str | None]:
    pairs = [(edge.node_i, edge.node_j) for edge in truss.edges]
    undirected_pairs = [tuple(sorted(pair)) for pair in pairs]
    if any(i == j for i, j in pairs):
        return None, None, "self_loop"
    if len(set(undirected_pairs)) != len(undirected_pairs):
        return None, None, "duplicate_edge"

    node_values = [[node.x, node.y, node.fx, node.fy] for node in truss.nodes]
    edge_values = [[edge.E, edge.A] for edge in truss.edges]
    if not np.all(np.isfinite(node_values)) or not np.all(np.isfinite(edge_values)):
        return None, None, "non_finite_input"

    lengths = []
    for edge in truss.edges:
        first, second = truss.nodes[edge.node_i], truss.nodes[edge.node_j]
        lengths.append(np.hypot(second.x - first.x, second.y - first.y))
    if any(length <= 0 for length in lengths):
        return None, None, "non_positive_edge_length"

    try:
        solution = solve_truss(truss)
    except InvalidEdgeError:
        return None, None, "invalid_edge"
    except UnstableStructureError:
        return None, None, "unstable_structure"

    if not np.all(np.isfinite(solution.displacements)):
        return None, None, "non_finite_displacement"
    if solution.normalized_free_residual() >= config.residual_limit:
        return None, None, "equilibrium_residual"

    positions = np.array([[node.x, node.y] for node in truss.nodes])
    span = float(positions[:, 0].max() - positions[:, 0].min())
    displacement_magnitudes = np.linalg.norm(solution.displacements, axis=1)
    if float(displacement_magnitudes.max()) > config.displacement_span_limit * span:
        return None, None, "large_displacement"

    reduced = solution.stiffness[np.ix_(solution.free_dofs, solution.free_dofs)]
    condition_number = float(np.linalg.cond(reduced))
    if not np.isfinite(condition_number) or condition_number > config.condition_number_limit:
        return None, None, "condition_number"
    return solution, condition_number, None


def generate_samples(
    config: GenerationConfig,
) -> tuple[dict[str, list[GeneratedSample]], dict[str, object]]:
    """Generate all requested splits from one explicit seeded NumPy generator."""

    rng = np.random.default_rng(config.seed)
    splits: dict[str, list[GeneratedSample]] = {}
    rejection_counts: dict[str, dict[str, int]] = {}
    next_base_id = 0
    next_graph_id = 0

    for split in SPLIT_NAMES:
        requested = config.split_counts[split]
        samples: list[GeneratedSample] = []
        split_rejections = {reason: 0 for reason in REJECTION_REASONS}

        while len(samples) < requested:
            base = _new_base(next_base_id, split, config, rng)
            next_base_id += 1
            variations = min(config.variations_per_base, requested - len(samples))

            for _ in range(variations):
                for _attempt in range(config.max_attempts_per_graph):
                    truss = _variation(base, split, config, rng)
                    solution, condition_number, rejection = _validate_candidate(truss, config)
                    if rejection is not None:
                        split_rejections[rejection] += 1
                        continue

                    bottom_count = base.num_panels + 1
                    panel_widths = np.diff([node.x for node in truss.nodes[:bottom_count]])
                    top_heights = np.array([node.y for node in truss.nodes[bottom_count:]])
                    samples.append(
                        GeneratedSample(
                            truss=truss,
                            solution=solution,
                            graph_id=next_graph_id,
                            base_id=base.base_id,
                            num_panels=base.num_panels,
                            panel_widths=panel_widths,
                            top_heights=top_heights,
                            condition_number=float(condition_number),
                        )
                    )
                    next_graph_id += 1
                    break
                else:
                    raise RuntimeError(
                        f"Could not generate a valid graph for {split} after "
                        f"{config.max_attempts_per_graph} attempts"
                    )

        splits[split] = samples
        rejection_counts[split] = split_rejections

    total_rejections: dict[str, int] = {}
    for counts in rejection_counts.values():
        for reason, count in counts.items():
            total_rejections[reason] = total_rejections.get(reason, 0) + count

    metadata: dict[str, object] = {
        "seed": config.seed,
        "split_counts": {name: len(samples) for name, samples in splits.items()},
        "base_ids": {
            name: sorted({sample.base_id for sample in samples})
            for name, samples in splits.items()
        },
        "rejection_counts": rejection_counts,
        "total_rejection_counts": total_rejections,
    }
    return splits, metadata

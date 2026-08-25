"""Verification examples for the Phase 2 linear truss solver."""

import numpy as np
import pytest

from trussgnn import Edge, InvalidEdgeError, Node, Truss, UnstableStructureError, solve_truss


def triangular_truss() -> Truss:
    """Return the deterministic, fully stable three-bar verification case."""

    youngs_modulus, area = 210.0e9, 1.0e-3
    return Truss(
        nodes=[
            Node(0.0, 0.0, fixed_x=True, fixed_y=True),
            Node(2.0, 0.0, fixed_y=True),
            Node(1.0, 1.0, fx=2_500.0, fy=-10_000.0),
        ],
        edges=[
            Edge(0, 1, youngs_modulus, area),
            Edge(0, 2, youngs_modulus, area),
            Edge(1, 2, youngs_modulus, area),
        ],
    )


def test_single_horizontal_bar_matches_analytical_displacement() -> None:
    length, youngs_modulus, area, load = 2.0, 200.0e9, 3.0e-4, 12_000.0
    truss = Truss(
        nodes=[Node(0.0, 0.0, fixed_x=True, fixed_y=True), Node(length, 0.0, fx=load, fixed_y=True)],
        edges=[Edge(0, 1, youngs_modulus, area)],
    )

    solution = solve_truss(truss)

    assert solution.displacements[1, 0] == pytest.approx(load * length / (youngs_modulus * area))
    assert solution.displacements[[0, 0, 1], [0, 1, 1]] == pytest.approx(0.0)
    assert solution.reactions[0, 0] == pytest.approx(-load)


def test_triangular_truss_is_finite_constrained_and_balanced() -> None:
    solution = solve_truss(triangular_truss())

    assert np.all(np.isfinite(solution.displacements))
    assert solution.displacements.reshape(-1)[solution.constrained_dofs] == pytest.approx(0.0)
    applied = solution.force.reshape((-1, 2)).sum(axis=0)
    reactions = solution.reactions.sum(axis=0)
    assert applied + reactions == pytest.approx(np.zeros(2), abs=1e-10)


def test_equilibrium_residual_is_small_on_free_degrees_of_freedom() -> None:
    solution = solve_truss(triangular_truss())

    assert solution.normalized_free_residual() < 1e-12


def test_unstable_structure_raises_clear_exception() -> None:
    truss = Truss(
        nodes=[Node(0.0, 0.0, fixed_x=True, fixed_y=True), Node(1.0, 0.0)],
        edges=[Edge(0, 1, 200.0e9, 1.0e-3)],
    )

    with pytest.raises(UnstableStructureError, match="singular or unstable"):
        solve_truss(truss)


def test_zero_length_edge_raises_clear_exception() -> None:
    truss = Truss(
        nodes=[Node(1.0, 1.0, fixed_x=True, fixed_y=True), Node(1.0, 1.0, fixed_y=True)],
        edges=[Edge(0, 1, 200.0e9, 1.0e-3)],
    )

    with pytest.raises(InvalidEdgeError, match="zero length"):
        solve_truss(truss)

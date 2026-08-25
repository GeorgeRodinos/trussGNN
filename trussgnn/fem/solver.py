"""A minimal finite-element solver for linear, static 2D trusses."""

from dataclasses import dataclass

import numpy as np


class InvalidEdgeError(ValueError):
    """An edge has invalid nodes or physical properties."""


class UnstableStructureError(ValueError):
    """The truss can move because it is not sufficiently supported or braced."""


@dataclass
class Node:
    """A truss joint with coordinates, load, and support conditions."""

    x: float
    y: float
    fx: float = 0.0
    fy: float = 0.0
    fixed_x: bool = False
    fixed_y: bool = False


@dataclass
class Edge:
    """An axial bar whose endpoints are local indices into ``Truss.nodes``."""

    node_i: int
    node_j: int
    E: float
    A: float


@dataclass
class Truss:
    """Nodes and the edges connecting them."""

    nodes: list[Node]
    edges: list[Edge]


@dataclass
class Solution:
    """Displacements, reactions, and the assembled equation ``K u = f``."""

    displacements: np.ndarray
    reactions: np.ndarray
    stiffness: np.ndarray
    force: np.ndarray
    free_dofs: np.ndarray
    constrained_dofs: np.ndarray

    def normalized_free_residual(self, epsilon: float = 1e-15) -> float:
        """Measure how closely the free degrees of freedom satisfy ``K u = f``."""

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        displacement_vector = self.displacements.flatten()
        residual = self.stiffness @ displacement_vector - self.force
        free_residual = residual[self.free_dofs]
        free_force = self.force[self.free_dofs]
        denominator = max(np.linalg.norm(free_force), epsilon)
        return float(np.linalg.norm(free_residual) / denominator)


def edge_stiffness(node_i: Node, node_j: Node, E: float, A: float) -> np.ndarray:
    """Return the 4x4 global stiffness matrix for one axial truss bar.

    The degree-of-freedom order is ``[u_ix, u_iy, u_jx, u_jy]``.
    """

    values = [node_i.x, node_i.y, node_j.x, node_j.y, E, A]
    if not np.all(np.isfinite(values)):
        raise InvalidEdgeError("Edge coordinates, E, and A must be finite")
    if E <= 0 or A <= 0:
        raise InvalidEdgeError("Edge E and A must be positive")

    dx = node_j.x - node_i.x
    dy = node_j.y - node_i.y
    length = np.hypot(dx, dy)
    if length == 0:
        raise InvalidEdgeError("Edge has zero length")

    cosine = dx / length
    sine = dy / length
    c = cosine
    s = sine

    # This projects the axial stiffness E*A/L into global x and y directions.
    return (E * A / length) * np.array(
        [
            [c * c, c * s, -c * c, -c * s],
            [c * s, s * s, -c * s, -s * s],
            [-c * c, -c * s, c * c, c * s],
            [-c * s, -s * s, c * s, s * s],
        ]
    )


def assemble_global_stiffness(truss: Truss) -> np.ndarray:
    """Add every edge stiffness matrix to the global stiffness matrix."""

    number_of_nodes = len(truss.nodes)
    if number_of_nodes == 0:
        raise ValueError("A truss must contain at least one node")

    K = np.zeros((2 * number_of_nodes, 2 * number_of_nodes))

    for edge in truss.edges:
        i = edge.node_i
        j = edge.node_j

        if isinstance(i, bool) or not isinstance(i, (int, np.integer)):
            raise InvalidEdgeError("Edge node_i must be a local integer index")
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)):
            raise InvalidEdgeError("Edge node_j must be a local integer index")
        if i < 0 or i >= number_of_nodes or j < 0 or j >= number_of_nodes:
            raise InvalidEdgeError("Edge endpoint index is outside the local node range")
        if i == j:
            raise InvalidEdgeError("Edge endpoints must be different nodes")

        k_edge = edge_stiffness(
            truss.nodes[i], truss.nodes[j], edge.E, edge.A
        )
        edge_dofs = [2 * i, 2 * i + 1, 2 * j, 2 * j + 1]

        # Add each of the 16 edge entries to its global matrix position.
        for local_row, global_row in enumerate(edge_dofs):
            for local_column, global_column in enumerate(edge_dofs):
                K[global_row, global_column] += k_edge[local_row, local_column]

    return K


def solve_truss(truss: Truss) -> Solution:
    """Assemble and solve a truss, then calculate its support reactions."""

    K = assemble_global_stiffness(truss)
    number_of_dofs = 2 * len(truss.nodes)

    f = np.zeros(number_of_dofs)
    constrained_dofs = []

    for node_index, node in enumerate(truss.nodes):
        if not np.all(np.isfinite([node.x, node.y, node.fx, node.fy])):
            raise ValueError("Node coordinates and loads must be finite")

        x_dof = 2 * node_index
        y_dof = x_dof + 1
        f[x_dof] = node.fx
        f[y_dof] = node.fy

        if node.fixed_x:
            constrained_dofs.append(x_dof)
        if node.fixed_y:
            constrained_dofs.append(y_dof)

    constrained = np.array(constrained_dofs, dtype=int)
    all_dofs = np.arange(number_of_dofs)
    free = np.setdiff1d(all_dofs, constrained)

    u = np.zeros(number_of_dofs)
    if len(free) > 0:
        K_free = K[np.ix_(free, free)]
        f_free = f[free]

        # A non-positive eigenvalue means a rigid motion or internal mechanism.
        eigenvalues = np.linalg.eigvalsh(K_free)
        largest = np.max(np.abs(eigenvalues))
        tolerance = np.finfo(float).eps * len(free) * largest
        if eigenvalues[0] <= tolerance:
            raise UnstableStructureError(
                "Reduced stiffness matrix is singular or unstable; "
                "check connectivity and supports"
            )

        # Solve directly. The stiffness matrix is never inverted explicitly.
        try:
            u[free] = np.linalg.solve(K_free, f_free)
        except np.linalg.LinAlgError as error:
            raise UnstableStructureError(
                "Reduced stiffness matrix is singular or unstable; "
                "check connectivity and supports"
            ) from error

    residual = K @ u - f
    reactions = np.zeros(number_of_dofs)
    reactions[constrained] = residual[constrained]

    return Solution(
        displacements=u.reshape((-1, 2)),
        reactions=reactions.reshape((-1, 2)),
        stiffness=K,
        force=f,
        free_dofs=free,
        constrained_dofs=constrained,
    )

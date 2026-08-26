"""Zero-displacement and node-independent baseline models."""

import torch
from torch import nn
from torch_geometric.data import Data


def _validate_network_options(hidden_dim: int, layers: int, dropout: float) -> None:
    if hidden_dim < 1:
        raise ValueError("hidden_dim must be at least 1")
    if layers < 1:
        raise ValueError("num_hidden_layers must be at least 1")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


class ZeroDisplacementBaseline(nn.Module):
    """Return physical zero displacement represented in normalized target space."""

    def __init__(self, target_mean: torch.Tensor, target_std: torch.Tensor) -> None:
        super().__init__()
        if target_mean.shape != (2,) or target_std.shape != (2,):
            raise ValueError("target_mean and target_std must have shape [2]")
        target_std = target_std.to(target_mean)
        safe_std = torch.where(target_std == 0, torch.ones_like(target_std), target_std)
        self.register_buffer("normalized_zero", -target_mean / safe_std)

    def forward(self, batch: Data) -> torch.Tensor:
        """Return one normalized zero-displacement prediction per node."""

        return self.normalized_zero.expand(batch.x.shape[0], 2)


class NodeMLP(nn.Module):
    """Predict each node independently from its six node features."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _validate_network_options(hidden_dim, num_hidden_layers, dropout)

        layers: list[nn.Module] = []
        input_dim = 6
        for _ in range(num_hidden_layers):
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, batch: Data) -> torch.Tensor:
        """Apply the same MLP to every node without reading graph edges."""

        return self.network(batch.x)

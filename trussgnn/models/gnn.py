"""A small edge-aware graph neural network for nodal displacement."""

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv


class EdgeAwareGNN(nn.Module):
    """Exchange messages through physical edges and predict two values per node."""

    def __init__(
        self,
        hidden_dim: int = 64,
        num_message_passing_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be at least 1")
        if num_message_passing_layers < 1:
            raise ValueError("num_message_passing_layers must be at least 1")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.node_encoder = nn.Linear(6, hidden_dim)
        self.convolutions = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    ),
                    edge_dim=5,
                )
                for _ in range(num_message_passing_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.node_decoder = nn.Linear(hidden_dim, 2)

    def forward(self, batch: Data) -> torch.Tensor:
        """Return normalized displacement predictions for every input node."""

        hidden = torch.relu(self.node_encoder(batch.x))
        for convolution in self.convolutions:
            hidden = convolution(hidden, batch.edge_index, batch.edge_attr)
            hidden = self.dropout(torch.relu(hidden))
        return self.node_decoder(hidden)

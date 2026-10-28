"""Graph convolutional experience sharing between SECs.

The ``GCN`` module is the one from the thesis code: two linear layers, ReLU, dropout 0.5, and the adjacency
aggregation applied to the output. The two stubs of the original are completed here. ``compute_adjacency_matrix``
returns the symmetric normalised adjacency with self loops that Kipf and Welling use, built from the
neighbours of each SEC. ``share_experience`` passes the per-SEC features through the GCN and returns one
shared reward per SEC, which the training loop adds to the Q target of that SEC's next transitions. The GCN
is trained to predict each SEC's next reward from its neighbourhood, so the shared reward is a learned,
neighbourhood-informed estimate rather than a random projection.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x))
        x = self.dropout(x)
        x = self.gc2(x)
        return torch.matmul(adj_matrix, x)


def compute_adjacency_matrix(neighbours: list[list[int]] | dict[int, list[int]]) -> torch.Tensor:
    """Symmetric normalised adjacency with self loops, D^-1/2 (A + I) D^-1/2, from a neighbours list or dict.

    A dict is read with its keys as node ids, either zero based or one based (the original ``Neighbors``
    dictionaries were one based).
    """
    if isinstance(neighbours, dict):
        keys = sorted(neighbours)
        offset = min(keys)
        n = len(keys)
        lists = [[int(v) - offset for v in neighbours[k]] for k in keys]
    else:
        n = len(neighbours)
        lists = [[int(v) for v in row] for row in neighbours]
    a = np.eye(n, dtype=np.float64)
    for i, row in enumerate(lists):
        for j in row:
            a[i, j] = 1.0
            a[j, i] = 1.0
    d = a.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    a_hat = d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    return torch.as_tensor(a_hat, dtype=torch.float32)


def share_experience(gcn: nn.Module, features: np.ndarray | torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    """Shared reward per SEC, shape ``(num_secs,)``, from the per-SEC feature rows."""
    x = torch.as_tensor(features, dtype=torch.float32)
    out = gcn(x, adjacency)
    return out.squeeze(-1)


class ExperienceSharer:
    """Trains the GCN online to predict the next reward of each SEC from the neighbourhood features, and
    hands the detached prediction to the agents as the shared reward term (scaled by ``beta``)."""

    def __init__(self, gcn: nn.Module, adjacency: torch.Tensor, beta: float = 0.1, learning_rate: float = 0.001, reward_scale: float = 1.0) -> None:
        self.gcn = gcn
        self.adjacency = adjacency
        self.beta = beta
        self.reward_scale = reward_scale
        self.optimizer = torch.optim.Adam(gcn.parameters(), lr=learning_rate)
        self.last_prediction: torch.Tensor | None = None
        self.shared = np.zeros(adjacency.shape[0], dtype=np.float64)
        self.losses: list[float] = []

    def update(self, features: np.ndarray, realised_rewards: np.ndarray) -> float | None:
        """Call once per round. Fits the previous prediction to the rewards realised in this round, then
        predicts the next round from the current features."""
        loss_value = None
        target = torch.as_tensor(realised_rewards / self.reward_scale, dtype=torch.float32)
        if self.last_prediction is not None:
            loss = F.mse_loss(self.last_prediction, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.item())
            self.losses.append(loss_value)
        self.gcn.train()
        self.last_prediction = share_experience(self.gcn, features, self.adjacency)
        self.shared = self.beta * self.last_prediction.detach().numpy() * self.reward_scale
        return loss_value

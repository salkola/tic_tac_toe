"""Policy-value network for MCTS move selection and leaf evaluation."""

import torch
from torch import nn


class PolicyValueNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = 9
        for _ in range(4):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(hidden_dim, 9)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        policy_logits = self.policy_head(features)
        value = torch.tanh(self.value_head(features)).squeeze(-1)
        return policy_logits, value

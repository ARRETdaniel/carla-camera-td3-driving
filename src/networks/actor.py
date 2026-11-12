"""Deterministic Policy Network (Actor) for TD3/DDPG"""

import torch
import torch.nn as nn
import numpy as np


class Actor(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        max_action: float = 1.0,
        hidden_size: int = 256,
    ):
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(
                layer.weight, -1.0 / np.sqrt(layer.in_features),
                1.0 / np.sqrt(layer.in_features)
            )
            if layer.bias is not None:
                nn.init.uniform_(
                    layer.bias, -1.0 / np.sqrt(layer.in_features),
                    1.0 / np.sqrt(layer.in_features)
                )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))

        a = self.tanh(self.fc3(x))
        a = a * self.max_action

        return a

    def pre_tanh(self, state: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

"""Critic Q-Networks for TD3/DDPG"""

import torch
import torch.nn as nn
import numpy as np


class Critic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_size: int = 256,
    ):
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.relu = nn.ReLU()

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

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        sa = torch.cat([state, action], dim=1)

        x = self.relu(self.fc1(sa))
        x = self.relu(self.fc2(x))

        q = self.fc3(x)

        return q


class TwinCritic(nn.Module):

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 2,
        hidden_size: int = 256,
    ):
        super(TwinCritic, self).__init__()

        self.Q1 = Critic(state_dim, action_dim, hidden_size)
        self.Q2 = Critic(state_dim, action_dim, hidden_size)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple:
        q1 = self.Q1(state, action)
        q2 = self.Q2(state, action)
        return q1, q2

    def Q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.Q1(state, action)

    def Q2_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.Q2(state, action)


class CriticLoss:

    @staticmethod
    def compute_td3_loss(
        q1: torch.Tensor,
        q2: torch.Tensor,
        target_q: torch.Tensor,
    ) -> tuple:
        loss_q1 = torch.nn.functional.mse_loss(q1, target_q)
        loss_q2 = torch.nn.functional.mse_loss(q2, target_q)
        loss = loss_q1 + loss_q2

        return loss_q1, loss_q2, loss

    @staticmethod
    def compute_ddpg_loss(
        q: torch.Tensor,
        target_q: torch.Tensor,
    ) -> torch.Tensor:
        loss = torch.nn.functional.mse_loss(q, target_q)
        return loss

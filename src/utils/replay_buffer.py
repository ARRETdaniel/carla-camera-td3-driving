"""Experience Replay Buffer for TD3/DDPG"""

import numpy as np
import torch
from typing import Tuple


class ReplayBuffer:

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_size: int = int(1e6),
        device: str = None
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        reward: float,
        done: bool
    ) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with {self.size} transitions"
            )

        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def __len__(self) -> int:
        return self.size

    def is_full(self) -> bool:
        return self.size >= self.max_size

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0


if __name__ == "__main__":
    print("Testing ReplayBuffer...")

    buffer = ReplayBuffer(state_dim=535, action_dim=2, max_size=1000)

    for i in range(100):
        state = np.random.randn(535)
        action = np.random.randn(2)
        next_state = np.random.randn(535)
        reward = np.random.randn()
        done = (i % 10 == 0)

        buffer.add(state, action, next_state, reward, done)

    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer full: {buffer.is_full()}")

    batch = buffer.sample(batch_size=32)
    states, actions, next_states, rewards, not_dones = batch

    print(f"\nSampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Not dones: {not_dones.shape}")
    print(f"  Device: {states.device}")

    print("\n✓ ReplayBuffer tests passed!")

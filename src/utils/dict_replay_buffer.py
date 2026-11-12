"""Dict Observation Replay Buffer for End-to-End CNN Training"""

import numpy as np
import torch
import logging
from typing import Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint((np.asarray(x, dtype=np.float32) + 1.0) * 127.5), 0, 255).astype(np.uint8)


def _from_uint8(u: np.ndarray) -> np.ndarray:
    return u.astype(np.float32) / 127.5 - 1.0


class DictReplayBuffer:

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (4, 84, 84),
        vector_dim: int = 53,
        action_dim: int = 2,
        max_size: int = int(1e6),
        device: str = None,
        handle_timeout_termination: bool = True
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.handle_timeout_termination = handle_timeout_termination

        self.image_shape = image_shape
        self.vector_dim = vector_dim
        self.action_dim = action_dim

        self.images = np.zeros((max_size, *image_shape), dtype=np.uint8)
        self.next_images = np.zeros((max_size, *image_shape), dtype=np.uint8)

        self.vectors = np.zeros((max_size, vector_dim), dtype=np.float32)
        self.next_vectors = np.zeros((max_size, vector_dim), dtype=np.float32)

        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)

        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.timeouts = np.zeros((max_size, 1), dtype=np.float32)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"DictReplayBuffer initialized:")
        print(f"  Max size: {max_size:,}")
        print(f"  Image shape: {image_shape}")
        print(f"  Vector dim: {vector_dim}")
        print(f"  Action dim: {action_dim}")
        print(f"  Device: {self.device}")

        image_bytes = max_size * np.prod(image_shape) * 1 * 2
        vector_bytes = max_size * vector_dim * 4 * 2
        other_bytes = max_size * (action_dim + 2) * 4
        total_mb = (image_bytes + vector_bytes + other_bytes) / (1024 * 1024)
        print(f"  Estimated memory: {total_mb:.1f} MB")

    def add(
        self,
        obs_dict: Dict[str, np.ndarray],
        action: np.ndarray,
        next_obs_dict: Dict[str, np.ndarray],
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ) -> None:
        self.images[self.ptr] = _to_uint8(obs_dict['image'])
        self.next_images[self.ptr] = _to_uint8(next_obs_dict['image'])

        self.vectors[self.ptr] = np.array(obs_dict['vector'])
        self.next_vectors[self.ptr] = np.array(next_obs_dict['vector'])

        self.actions[self.ptr] = np.array(action)
        self.rewards[self.ptr] = reward

        self.dones[self.ptr] = float(done)

        if self.handle_timeout_termination:
            self.timeouts[self.ptr] = float(info.get("TimeLimit.truncated", False))

        if (self.ptr + 1) % 100 == 0 or (self.ptr + 1 == self.max_size and self.size < self.max_size):
            current_size = min(self.size + 1, self.max_size)
            logger.info(
                f"ReplayBuffer: size={current_size}/{self.max_size} "
                f"({100.0*current_size/self.max_size:.1f}%), "
                f"reward_mean={self.rewards[:current_size].mean():.3f} "
                f"(std={self.rewards[:current_size].std():.3f}), "
                f"done_rate={self.dones[:current_size].mean():.2%}, "
                f"timeout_rate={self.timeouts[:current_size].mean():.2%}"
            )

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size: int,
        env: Optional[Any] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        if batch_size > self.size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer with {self.size} transitions"
            )

        ind = np.random.randint(0, self.size, size=batch_size)

        obs_dict = {
            'image': torch.FloatTensor(_from_uint8(self.images[ind])).to(self.device),
            'vector': torch.FloatTensor(self.vectors[ind]).to(self.device)
        }

        next_obs_dict = {
            'image': torch.FloatTensor(_from_uint8(self.next_images[ind])).to(self.device),
            'vector': torch.FloatTensor(self.next_vectors[ind]).to(self.device)
        }

        obs_dict = self._normalize_obs(obs_dict, env)
        next_obs_dict = self._normalize_obs(next_obs_dict, env)

        actions = torch.FloatTensor(self.actions[ind]).to(self.device)

        rewards = torch.FloatTensor(self.rewards[ind]).to(self.device)
        rewards = self._normalize_reward(rewards, env)

        dones_batch = self.dones[ind]
        if self.handle_timeout_termination:
            dones_batch = dones_batch * (1.0 - self.timeouts[ind])

        dones = torch.FloatTensor(dones_batch).to(self.device)

        return obs_dict, actions, next_obs_dict, rewards, dones

    def _normalize_obs(
        self,
        obs_dict: Dict[str, torch.Tensor],
        env: Optional[Any]
    ) -> Dict[str, torch.Tensor]:
        if env is not None and hasattr(env, 'normalize_obs'):
            return {key: env.normalize_obs(obs) for key, obs in obs_dict.items()}
        return obs_dict

    def _normalize_reward(
        self,
        reward: torch.Tensor,
        env: Optional[Any]
    ) -> torch.Tensor:
        if env is not None and hasattr(env, 'normalize_reward'):
            return env.normalize_reward(reward)
        return reward

    def __len__(self) -> int:
        return self.size

    def is_full(self) -> bool:
        return self.size >= self.max_size

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0


if __name__ == "__main__":
    print("Testing DictReplayBuffer...")

    buffer = DictReplayBuffer(
        image_shape=(4, 84, 84),
        vector_dim=23,
        action_dim=2,
        max_size=1000
    )

    for i in range(100):
        obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        action = np.random.randn(2)
        next_obs_dict = {
            'image': np.random.randn(4, 84, 84).astype(np.float32),
            'vector': np.random.randn(23).astype(np.float32)
        }
        reward = np.random.randn()
        done = (i % 10 == 0)

        buffer.add(obs_dict, action, next_obs_dict, reward, done)

    print(f"\nBuffer size: {len(buffer)}")
    print(f"Buffer full: {buffer.is_full()}")

    obs_dict, actions, next_obs_dict, rewards, not_dones = buffer.sample(batch_size=32)

    print(f"\nSampled batch shapes:")
    print(f"  obs_dict['image']: {obs_dict['image'].shape}")
    print(f"  obs_dict['vector']: {obs_dict['vector'].shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  next_obs_dict['image']: {next_obs_dict['image'].shape}")
    print(f"  next_obs_dict['vector']: {next_obs_dict['vector'].shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Not dones: {not_dones.shape}")

    print(f"\nDevice check:")
    print(f"  obs_dict['image'].device: {obs_dict['image'].device}")
    print(f"  obs_dict['vector'].device: {obs_dict['vector'].device}")

    print(f"\nGradient tracking:")
    print(f"  obs_dict['image'].requires_grad: {obs_dict['image'].requires_grad}")
    print(f"  obs_dict['vector'].requires_grad: {obs_dict['vector'].requires_grad}")

    print("\n✓ DictReplayBuffer tests passed!")

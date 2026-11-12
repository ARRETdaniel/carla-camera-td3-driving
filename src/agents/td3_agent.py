"""Twin Delayed Deep Deterministic Policy Gradient (TD3) Agent"""

import copy
import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from src.networks.actor import Actor
from src.networks.critic import TwinCritic
from src.utils.replay_buffer import ReplayBuffer
from src.utils.dict_replay_buffer import DictReplayBuffer


class TD3Agent:

    def __init__(
        self,
        state_dim: int = 565,
        action_dim: int = 2,
        max_action: float = 1.0,
        cnn_extractor: Optional[torch.nn.Module] = None,
        actor_cnn: Optional[torch.nn.Module] = None,
        critic_cnn: Optional[torch.nn.Module] = None,
        use_dict_buffer: bool = True,
        config: Optional[Dict] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ):
        if config is None:
            if config_path is None:
                config_path = "config/td3_config.yaml"
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.config = config

        algo_config = config['algorithm']
        self.discount = algo_config.get('gamma', algo_config.get('discount', 0.99))
        self.tau = algo_config['tau']
        self.policy_noise = algo_config['policy_noise']
        self.noise_clip = algo_config['noise_clip']
        self.policy_freq = algo_config['policy_freq']
        self.actor_lr = algo_config.get('actor_lr', algo_config.get('learning_rate', 0.0003))
        self.critic_lr = algo_config.get('critic_lr', algo_config.get('learning_rate', 0.0003))

        _caps = algo_config.get('caps') or {}
        self.caps_enabled = bool(_caps.get('enabled', False))
        self.caps_lambda_temporal = float(_caps.get('lambda_temporal', 0.0))
        self.caps_lambda_spatial = float(_caps.get('lambda_spatial', 0.0))
        self.caps_sigma_vector = float(_caps.get('sigma_vector', 0.0))
        if self.caps_enabled:
            assert self.caps_lambda_temporal >= 0.0 and self.caps_lambda_spatial >= 0.0, _caps
            assert self.caps_lambda_spatial == 0.0 or self.caps_sigma_vector > 0.0, _caps
            print(f"  CAPS ON: lambda_temporal={self.caps_lambda_temporal}, lambda_spatial={self.caps_lambda_spatial}, "
                  f"sigma_vector={self.caps_sigma_vector} (noise on the observation vector only)")
        self.pre_activation_weight = float(algo_config.get('pre_activation_weight', 0.0) or 0.0)
        assert self.pre_activation_weight >= 0.0, self.pre_activation_weight
        if self.pre_activation_weight > 0.0:
            print(f"  PRE-ACTIVATION PENALTY ON: weight={self.pre_activation_weight} (rlkit DDPG form)")
        _aug = algo_config.get('augmentation') or {}
        self.image_aug = None
        if _aug.get('type'):
            assert _aug['type'] == 'random_shift', _aug
            from src.utils.augmentation import RandomShiftsAug
            self.image_aug = RandomShiftsAug(pad=int(_aug.get('pad', 4)))
            print(f"  IMAGE AUGMENTATION ON: random_shift, pad={self.image_aug.pad} (DrQ-v2)")

        training_config = config.get('training', {})
        algo_config_training = config.get('algorithm', {})

        self.batch_size = training_config.get('batch_size', algo_config_training.get('batch_size', 256))
        self.buffer_size = training_config.get('buffer_size', algo_config_training.get('buffer_size', 1000000))
        print(f"[DEBUG] Buffer size from config: {self.buffer_size}")
        self.start_timesteps = training_config.get('start_timesteps', training_config.get('learning_starts',
                                                   algo_config_training.get('learning_starts', 25000)))
        print(f"[DEBUG] Start timesteps from config: {self.start_timesteps}")

        exploration_config = config.get('exploration', {})
        self.expl_noise = exploration_config.get('expl_noise', algo_config.get('exploration_noise', 0.1))

        if device is not None:
            self.device = torch.device(device)
            print(f"TD3Agent initialized on device: {self.device} (manually specified)")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"TD3Agent initialized on device: {self.device}")

        network_config = config.get('networks', {}).get('actor', {})
        hidden_layers = network_config.get('hidden_sizes', network_config.get('hidden_layers', [256, 256]))
        hidden_size = hidden_layers[0] if isinstance(hidden_layers, list) else hidden_layers
        self.actor = Actor(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            hidden_size=hidden_size
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor)


        if cnn_extractor is not None and (actor_cnn is None or critic_cnn is None):
            print("  WARNING: cnn_extractor parameter is DEPRECATED!")
            print("  Using cnn_extractor for both actor and critic (NOT RECOMMENDED)")
            print("  This can cause gradient interference. Use actor_cnn/critic_cnn instead.")
            if actor_cnn is None:
                actor_cnn = cnn_extractor
            if critic_cnn is None:
                critic_cnn = cnn_extractor

        self.actor_cnn = actor_cnn
        self.critic_cnn = critic_cnn
        self.use_dict_buffer = use_dict_buffer


        self.logger = logging.getLogger(__name__)

        if self.actor_cnn is not None:
            actor_params = list(self.actor.parameters())
            self.logger.info(f"  Actor optimizer: {len(list(self.actor.parameters()))} MLP params "
                             f"(encoder excluded - DrQ-v2 shared-encoder design)")
        else:
            actor_params = list(self.actor.parameters())
            self.logger.info(f"  Actor optimizer: {len(list(self.actor.parameters()))} MLP params (no CNN)")

        _clip_cfg = algo_config.get("gradient_clipping", {})
        _clip_on = _clip_cfg.get("enabled", True)
        _a = _clip_cfg.get("actor_max_norm", 1.0)
        _c = _clip_cfg.get("critic_max_norm", 10.0)
        self.actor_max_norm = float(_a) if (_clip_on and _a is not None) else float("inf")
        self.critic_max_norm = float(_c) if (_clip_on and _c is not None) else float("inf")
        self.clip_norm_type = float(_clip_cfg.get("norm_type", 2.0))
        self.logger.info(
            f"  Gradient clipping: actor_max_norm={self.actor_max_norm}, "
            f"critic_max_norm={self.critic_max_norm} (inf = disabled)"
        )

        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=self.actor_lr,
            weight_decay=1e-4
        )
        self.logger.info(f"  Actor optimizer created: lr={self.actor_lr}, weight_decay=1e-4, total_params={sum(p.numel() for p in actor_params)}")

        critic_config = config.get('networks', {}).get('critic', {})

        self.use_twin_critic = int(critic_config.get('n_critics', 2)) >= 2
        self.target_min = bool(critic_config.get('target_min', self.use_twin_critic))
        if self.target_min and not self.use_twin_critic:
            raise ValueError("networks.critic.target_min: true needs n_critics >= 2 (a min over two trained heads)")
        if self.use_twin_critic and not self.target_min:
            print("  AHE-2H MODE: two critic heads fitted, target Q1' only (networks.critic.target_min: false)")

        hidden_layers = critic_config.get('hidden_sizes', critic_config.get('hidden_layers', [256, 256]))
        hidden_size = hidden_layers[0] if isinstance(hidden_layers, list) else hidden_layers
        self.critic = TwinCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size
        ).to(self.device)

        self.critic_target = copy.deepcopy(self.critic)

        if self.critic_cnn is not None:
            critic_params = list(self.critic.parameters()) + list(self.critic_cnn.parameters())
            self.logger.info(f"  Critic optimizer: {len(list(self.critic.parameters()))} MLP params + {len(list(self.critic_cnn.parameters()))} CNN params")
        else:
            critic_params = list(self.critic.parameters())
            self.logger.info(f"  Critic optimizer: {len(list(self.critic.parameters()))} MLP params (no CNN)")

        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=self.critic_lr,
            weight_decay=1e-4
        )
        self.logger.info(f"  Critic optimizer created: lr={self.critic_lr}, weight_decay=1e-4, total_params={sum(p.numel() for p in critic_params)}")

        self.actor_cnn_optimizer = None
        self.critic_cnn_optimizer = None

        if self.actor_cnn is not None:
            self.actor_cnn.train()
            self.logger.info(f"  Actor CNN set to training mode (gradients will flow through actor_optimizer)")

        if self.critic_cnn is not None:
            self.critic_cnn.train()
            self.logger.info(f"  Critic CNN set to training mode (gradients will flow through critic_optimizer)")

        if use_dict_buffer and (self.actor_cnn is None or self.critic_cnn is None):
            print("   WARNING: DictReplayBuffer enabled but CNN(s) missing!")
            if self.actor_cnn is None:
                print("      actor_cnn is None - actor will use zero features")
            if self.critic_cnn is None:
                print("      critic_cnn is None - critic will use zero features")

        if self.actor_cnn is not None and self.critic_cnn is not None:
            if id(self.actor_cnn) == id(self.critic_cnn):
                print("      CRITICAL WARNING: Actor and critic share the SAME CNN instance!")
                print("      This causes gradient interference and training instability.")
                print("      Create separate CNN instances: actor_cnn = CNN(), critic_cnn = CNN()")
            else:
                print(f"     Actor and critic use SEPARATE CNN instances (recommended)")
                print(f"     Actor CNN id: {id(self.actor_cnn)}")
                print(f"     Critic CNN id: {id(self.critic_cnn)}")

        if use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
            num_waypoints = self.config.get("route", {}).get("num_waypoints_ahead", 25)
            if vector_dim is None:
                vector_dim = 3 + (num_waypoints * 2)

            self.replay_buffer = DictReplayBuffer(
                image_shape=(4, 84, 84),
                vector_dim=vector_dim,
                action_dim=action_dim,
                max_size=self.buffer_size,
                device=self.device
            )
            print(f"  Using DictReplayBuffer for end-to-end CNN training")
            print(f"  Vector dimension: {vector_dim} (3 kinematic + {num_waypoints} waypoints × 2"
                  f"{f' + {vector_dim - 3 - 2 * num_waypoints} extra' if vector_dim != 3 + 2 * num_waypoints else ''})")
        else:
            self.replay_buffer = ReplayBuffer(
                state_dim=state_dim,
                action_dim=action_dim,
                max_size=self.buffer_size,
                device=self.device
            )
            print(f"WARNING: using standard ReplayBuffer (CNN not trained)")

        self.total_it = 0

        self.cnn_diagnostics = None


        self.action_buffer = []
        self.action_buffer_size = 100

        print(f"TD3Agent initialized with:")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Actor hidden size: {network_config.get('hidden_sizes', network_config.get('hidden_layers', [256, 256]))}")
        print(f"  Critic hidden size: {critic_config.get('hidden_sizes', critic_config.get('hidden_layers', [256, 256]))}")
        print(f"  Discount γ: {self.discount}, Tau τ: {self.tau}")
        print(f"  Policy freq: {self.policy_freq}, Policy noise: {self.policy_noise}")
        print(f"  Exploration noise: {self.expl_noise}")
        print(f"  Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
        print(f"  Action statistics tracking: enabled (buffer size: {self.action_buffer_size})")

    def select_action(
        self,
        state: Union[np.ndarray, Dict[str, np.ndarray]],
        noise: Optional[float] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        if isinstance(state, dict):
            obs_dict_tensor = {
                'image': torch.FloatTensor(state['image']).unsqueeze(0).to(self.device),
                'vector': torch.FloatTensor(state['vector']).unsqueeze(0).to(self.device)
            }

            with torch.no_grad():
                state_tensor = self.extract_features(
                    obs_dict_tensor,
                    enable_grad=False,
                    use_actor_cnn=False
                )
        else:
            state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()

        if not deterministic and noise is not None and noise > 0:
            action_clean = action.copy()
            noise_sample = np.random.normal(0, noise, size=self.action_dim)
            action = action + noise_sample

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[NOISE_VERIFICATION] "
                    f"clean=[{action_clean[0]:+.4f}, {action_clean[1]:+.4f}] + "
                    f"noise=[{noise_sample[0]:+.4f}, {noise_sample[1]:+.4f}] = "
                    f"noisy=[{action[0]:+.4f}, {action[1]:+.4f}] (before clip)"
                )

            action = np.clip(action, -self.max_action, self.max_action)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"[NOISE_VERIFICATION] "
                    f"final=[{action[0]:+.4f}, {action[1]:+.4f}] (after clip to [{-self.max_action}, {self.max_action}])"
                )

        self.action_buffer.append(action.copy())
        if len(self.action_buffer) > self.action_buffer_size:
            self.action_buffer.pop(0)

        return action

    def extract_features(
        self,
        obs_dict: Dict[str, torch.Tensor],
        enable_grad: bool = True,
        use_actor_cnn: bool = True
    ) -> torch.Tensor:
        cnn = self.actor_cnn if use_actor_cnn else self.critic_cnn

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FEATURE EXTRACTION - INPUT:\n"
                f"   Mode: {'ACTOR' if use_actor_cnn else 'CRITIC'}\n"
                f"   Gradient: {'ENABLED' if enable_grad else 'DISABLED'}\n"
                f"   Image shape: {obs_dict['image'].shape}\n"
                f"   Image range: [{obs_dict['image'].min().item():.3f}, {obs_dict['image'].max().item():.3f}]\n"
                f"   Vector shape: {obs_dict['vector'].shape}\n"
                f"   Vector range: [{obs_dict['vector'].min().item():.3f}, {obs_dict['vector'].max().item():.3f}]"
            )

        if cnn is None:
            batch_size = obs_dict['vector'].shape[0]
            image_features = torch.zeros(batch_size, 512, device=self.device)
            print(f"WARNING: extract_features called but {'actor' if use_actor_cnn else 'critic'}_cnn is None!")
        elif enable_grad:
            image_features = cnn(obs_dict['image'])
        else:
            with torch.no_grad():
                image_features = cnn(obs_dict['image'])

        if self.logger.isEnabledFor(logging.DEBUG):
            cnn_l2_norm = image_features.norm(dim=1).mean().item()
            self.logger.debug(
                f"   FEATURE EXTRACTION - IMAGE FEATURES:\n"
                f"   Shape: {image_features.shape}\n"
                f"   Range: [{image_features.min().item():.3f}, {image_features.max().item():.3f}]\n"
                f"   Mean: {image_features.mean().item():.3f}, Std: {image_features.std().item():.3f}\n"
                f"   L2 norm: {cnn_l2_norm:.3f}\n"
                f"   Requires grad: {image_features.requires_grad}"
            )


        state = torch.cat([image_features, obs_dict['vector']], dim=1)

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FEATURE EXTRACTION - OUTPUT:\n"
                f"   State shape: {state.shape} (previous it was 512 image + 23 vector = 535)\n"
                f"   Range: [{state.min().item():.3f}, {state.max().item():.3f}]\n"
                f"   Mean: {state.mean().item():.3f}, Std: {state.std().item():.3f}\n"
                f"   Requires grad: {state.requires_grad}\n"
                f"   Has NaN: {torch.isnan(state).any().item()}\n"
                f"   Has Inf: {torch.isinf(state).any().item()}\n"
                f"   State quality: {'GOOD' if not (torch.isnan(state).any() or torch.isinf(state).any()) else 'BAD'}"
            )

        return state

    def enable_diagnostics(self) -> None:
        if self.actor_cnn is not None or self.critic_cnn is not None:
            try:
                from src.utils.cnn_diagnostics import CNNDiagnostics
                self.actor_cnn_diagnostics = None
                self.critic_cnn_diagnostics = None

                if self.actor_cnn is not None:
                    self.actor_cnn_diagnostics = CNNDiagnostics(self.actor_cnn)
                    print("[CNN DIAGNOSTICS] Enabled actor CNN diagnostics tracking")

                if self.critic_cnn is not None:
                    self.critic_cnn_diagnostics = CNNDiagnostics(self.critic_cnn)
                    print("[CNN DIAGNOSTICS] Enabled critic CNN diagnostics tracking")
            except ImportError:
                print("[CNN DIAGNOSTICS] WARNING: Could not import CNNDiagnostics")
                self.actor_cnn_diagnostics = None
                self.critic_cnn_diagnostics = None
        else:
            print("[CNN DIAGNOSTICS] WARNING: No CNN extractors available")
            self.actor_cnn_diagnostics = None
            self.critic_cnn_diagnostics = None

    def get_diagnostics_summary(self, last_n: int = 100) -> Optional[Dict]:
        summary = {}

        if hasattr(self, 'actor_cnn_diagnostics') and self.actor_cnn_diagnostics is not None:
            summary['actor'] = self.actor_cnn_diagnostics.get_summary(last_n=last_n)

        if hasattr(self, 'critic_cnn_diagnostics') and self.critic_cnn_diagnostics is not None:
            summary['critic'] = self.critic_cnn_diagnostics.get_summary(last_n=last_n)

        return summary if summary else None

    def print_diagnostics(self, last_n: int = 100) -> None:
        if self.cnn_diagnostics is not None:
            self.cnn_diagnostics.print_summary(last_n=last_n)
        else:
            print("[CNN DIAGNOSTICS] Not enabled. Call agent.enable_diagnostics() first.")

    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        self.total_it += 1

        if batch_size is None:
            batch_size = self.batch_size

        if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
            obs_dict, action, next_obs_dict, reward, dones = self.replay_buffer.sample(batch_size)
            if self.image_aug is not None:
                obs_dict = dict(obs_dict, image=self.image_aug(obs_dict['image']))
                next_obs_dict = dict(next_obs_dict, image=self.image_aug(next_obs_dict['image']))

            if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
                self.logger.debug(
                    f"\n   TRAINING STEP {self.total_it} - BATCH SAMPLED:\n"
                    f"   Batch size: {batch_size}\n"
                    f"   Reward range: [{reward.min().item():.2f}, {reward.max().item():.2f}]\n"
                    f"   Reward mean: {reward.mean().item():.2f}, Std: {reward.std().item():.2f}\n"
                    f"   Action range: [{action.min().item():.3f}, {action.max().item():.3f}]\n"
                    f"   Done count: {dones.bool().sum().item()}/{batch_size}"
                )

            state = self.extract_features(
                obs_dict,
                enable_grad=True,
                use_actor_cnn=False
            )
        else:
            state, action, next_state, reward, not_done = self.replay_buffer.sample(batch_size)
            dones = 1.0 - not_done

        with torch.no_grad():
            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                next_state_actor = self.extract_features(
                    next_obs_dict, enable_grad=False, use_actor_cnn=False
                )
                next_state_critic = self.extract_features(
                    next_obs_dict, enable_grad=False, use_actor_cnn=False
                )
            else:
                next_state_actor = next_state_critic = next_state

            noise = torch.randn_like(action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_action = self.actor_target(next_state_actor) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state_critic, next_action)
            if self.target_min:
                target_Q_min = torch.min(target_Q1, target_Q2)
            else:
                target_Q_min = target_Q1

            self._twin_q_abs_diff = (target_Q1 - target_Q2).abs().mean().item()
            self._clipped_double_q_gap = (target_Q1 - target_Q_min).mean().item()
            self._twin_q_disagree_frac = (target_Q2 < target_Q1).float().mean().item()

            target_Q = reward + (1.0 - dones) * self.discount * target_Q_min

        current_Q1, current_Q2 = self.critic(state, action)

        if self.use_twin_critic:
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        else:
            critic_loss = F.mse_loss(current_Q1, target_Q)

        if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
            self.logger.debug(
                f"   TRAINING STEP {self.total_it} - CRITIC UPDATE:\n"
                f"   Current Q1: mean={current_Q1.mean().item():.2f}, std={current_Q1.std().item():.2f}, "
                f"min={current_Q1.min().item():.2f}, max={current_Q1.max().item():.2f}\n"
                f"   Current Q2: mean={current_Q2.mean().item():.2f}, std={current_Q2.std().item():.2f}, "
                f"min={current_Q2.min().item():.2f}, max={current_Q2.max().item():.2f}\n"
                f"   Target Q: mean={target_Q.mean().item():.2f}, std={target_Q.std().item():.2f}, "
                f"min={target_Q.min().item():.2f}, max={target_Q.max().item():.2f}\n"
                f"   Critic loss: {critic_loss.item():.4f}\n"
                f"   TD error Q1: {(current_Q1 - target_Q).abs().mean().item():.4f}\n"
                f"   TD error Q2: {(current_Q2 - target_Q).abs().mean().item():.4f}\n"
                f"   Reward stats: mean={reward.mean().item():.2f}, std={reward.std().item():.2f}, "
                f"min={reward.min().item():.2f}, max={reward.max().item():.2f}\n"
                f"   Next Q stats: mean={target_Q1.mean().item():.2f}, min_Q={target_Q.mean().item():.2f}\n"
                f"   Discount applied: {self.discount:.4f}, Done ratio: {dones.bool().sum().item()}/{batch_size}"
            )

        self.critic_optimizer.zero_grad()
        if self.critic_cnn_optimizer is not None:
            self.critic_cnn_optimizer.zero_grad()

        critic_loss.backward()

        if self.critic_cnn is not None:
            critic_grad_norm_before = torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()) + list(self.critic_cnn.parameters()),
                max_norm=float('inf'),
                norm_type=2.0
            ).item()

            if self.total_it % 100 == 0:
                self.logger.debug(f"  Critic gradient norm BEFORE clip: {critic_grad_norm_before:.4f}")

            torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()) + list(self.critic_cnn.parameters()),
                max_norm=self.critic_max_norm,
                norm_type=2.0
            )

            critic_grad_norm_after = sum(
                p.grad.norm().item() ** 2 for p in list(self.critic.parameters()) + list(self.critic_cnn.parameters()) if p.grad is not None
            ) ** 0.5

            if self.total_it % 100 == 0:
                self.logger.debug(
                    f"  Critic gradient norm AFTER clip: {critic_grad_norm_after:.4f} "
                    f"(max={self.critic_max_norm})")
                if (np.isfinite(self.critic_max_norm)
                        and critic_grad_norm_after > self.critic_max_norm * 1.01):
                    self.logger.warning(
                        f"CLIPPING FAILED! Critic grad {critic_grad_norm_after:.4f} > "
                        f"{self.critic_max_norm}")
        else:
            critic_grad_norm_before = 0.0
            critic_grad_norm_after = 0.0
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self.critic_max_norm,
                norm_type=2.0
            )
        if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
            critic_grad_norm = sum(
                p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None
            )
            if self.critic_cnn is not None:
                cnn_grad_norm = sum(
                    p.grad.norm().item() for p in self.critic_cnn.parameters() if p.grad is not None
                )
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - GRADIENTS (AFTER CLIPPING):\n"
                    f"   Critic grad norm: {critic_grad_norm:.4f}\n"
                    f"   Critic CNN grad norm: {cnn_grad_norm:.4f}"
                )
            else:
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - GRADIENTS (AFTER CLIPPING):\n"
                    f"   Critic grad norm: {critic_grad_norm:.4f}"
                )

        if self.cnn_diagnostics is not None:
            self.cnn_diagnostics.capture_gradients()

            if self.total_it % 100 == 0:
                self._log_detailed_gradient_flow(self.critic_cnn, "critic_cnn")

        if self.cnn_diagnostics is not None and self.use_dict_buffer and self.critic_cnn is not None:
            with torch.no_grad():
                sample_features = self.critic_cnn(obs_dict['image'])
                self.cnn_diagnostics.capture_features(sample_features, name="critic_update")

                if self.total_it % 100 == 0:
                    self._log_feature_diversity(sample_features, "critic_cnn")

        self.critic_optimizer.step()

        if self.cnn_diagnostics is not None and self.critic_cnn is not None:
            self.cnn_diagnostics.capture_weights()

            if self.total_it % 100 == 0:
                self._log_weight_statistics(self.critic_cnn, "critic_cnn")

        metrics = {
            'critic_loss': critic_loss.item(),
            'q1_value': current_Q1.mean().item(),
            'q2_value': current_Q2.mean().item(),
            'debug/twin_q_abs_diff': self._twin_q_abs_diff,
            'debug/clipped_double_q_gap': self._clipped_double_q_gap,
            'debug/twin_q_disagree_frac': self._twin_q_disagree_frac,
            'debug/twin_q_abs_diff_online': (current_Q1 - current_Q2).abs().mean().item(),
            'debug/critic_grad_norm_BEFORE_clip': critic_grad_norm_before,
            'debug/critic_grad_norm_AFTER_clip': critic_grad_norm_after,
            'debug/critic_grad_clip_ratio': critic_grad_norm_after / max(critic_grad_norm_before, 1e-8),
            'debug/q1_std': current_Q1.std().item(),
            'debug/q1_min': current_Q1.min().item(),
            'debug/q1_max': current_Q1.max().item(),
            'debug/q2_std': current_Q2.std().item(),
            'debug/q2_min': current_Q2.min().item(),
            'debug/q2_max': current_Q2.max().item(),
            'debug/q1_q2_diff': torch.abs(current_Q1 - current_Q2).mean().item(),
            'debug/q1_q2_max_diff': torch.abs(current_Q1 - current_Q2).max().item(),
            'debug/target_q_mean': target_Q.mean().item(),
            'debug/target_q_std': target_Q.std().item(),
            'debug/target_q_min': target_Q.min().item(),
            'debug/target_q_max': target_Q.max().item(),
            'debug/td_error_q1': (current_Q1 - target_Q).abs().mean().item(),
            'debug/td_error_q2': (current_Q2 - target_Q).abs().mean().item(),
            'debug/reward_mean': reward.mean().item(),
            'debug/reward_std': reward.std().item(),
            'debug/reward_min': reward.min().item(),
            'debug/reward_max': reward.max().item(),
            'debug/done_ratio': (dones.bool()).sum().item() / batch_size,
            'debug/effective_discount': (dones * self.discount).mean().item(),
        }


        if self.critic_cnn is not None:
            critic_cnn_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_cnn.parameters(),
                max_norm=float('inf'),
                norm_type=2.0
            ).item()
            metrics['critic_cnn_grad_norm'] = critic_cnn_grad_norm
            metrics['debug/critic_cnn_grad_norm_AFTER_clip'] = critic_cnn_grad_norm

        critic_mlp_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=float('inf'),
            norm_type=2.0
        ).item()
        metrics['critic_mlp_grad_norm'] = critic_mlp_grad_norm
        metrics['debug/critic_mlp_grad_norm_AFTER_clip'] = critic_mlp_grad_norm

        if self.total_it % self.policy_freq == 0:
            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                state_for_actor = self.extract_features(
                    obs_dict,
                    enable_grad=False,
                    use_actor_cnn=False
                )
            else:
                state_for_actor = state.detach() if hasattr(state, "detach") else state

            if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                state_for_critic_eval = self.extract_features(
                    obs_dict, enable_grad=False, use_actor_cnn=False
                )
            else:
                state_for_critic_eval = state.detach()

            pi_s = self.actor(state_for_actor)
            actor_q_values = self.critic.Q1(state_for_critic_eval, pi_s)
            actor_loss = -actor_q_values.mean()

            caps_metrics = {}
            if self.caps_enabled:
                if self.use_dict_buffer and (self.actor_cnn is not None or self.critic_cnn is not None):
                    next_state_for_actor = self.extract_features(
                        next_obs_dict, enable_grad=False, use_actor_cnn=False
                    )
                    vector_dim = obs_dict['vector'].shape[1]
                else:
                    next_state_for_actor = next_state.detach() if hasattr(next_state, "detach") else next_state
                    vector_dim = state_for_actor.shape[1]
                caps_temporal, caps_spatial = self._caps_terms(
                    pi_s, state_for_actor, next_state_for_actor, vector_dim
                )
                actor_loss = (actor_loss
                              + self.caps_lambda_temporal * caps_temporal
                              + self.caps_lambda_spatial * caps_spatial)
                caps_metrics['actor_reg/caps_temporal'] = caps_temporal.item()
                caps_metrics['actor_reg/caps_spatial'] = caps_spatial.item()
            if self.pre_activation_weight > 0.0:
                pre_act = self.actor.pre_tanh(state_for_actor).pow(2).sum(dim=1).mean()
                actor_loss = actor_loss + self.pre_activation_weight * pre_act
                caps_metrics['actor_reg/pre_activation'] = pre_act.item()
            if caps_metrics:
                with torch.no_grad():
                    _u_steer = self.actor.pre_tanh(state_for_actor)[:, 0].abs()
                caps_metrics['actor_reg/q_term'] = -actor_q_values.mean().item()
                caps_metrics['actor_reg/pre_tanh_steer_abs_median'] = _u_steer.median().item()
                caps_metrics['actor_reg/steer_abs_gt_0_8_frac'] = (pi_s[:, 0].abs() > 0.8).float().mean().item()

            if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
                self.logger.debug(
                    f"   TRAINING STEP {self.total_it} - ACTOR UPDATE (delayed, freq={self.policy_freq}):\n"
                    f"   Actor loss: {actor_loss.item():.4f}\n"
                    f"   Q-value under current policy: {-actor_loss.item():.2f}\n"
                    f"      ACTUAL Q-values driving policy:\n"
                    f"      mean={actor_q_values.mean().item():.2f}, std={actor_q_values.std().item():.2f}\n"
                    f"      min={actor_q_values.min().item():.2f}, max={actor_q_values.max().item():.2f}\n"
                    f"   (If mean ≈ +2.4M and actor_loss ≈ -2.4M → Critic overestimation confirmed)\n"
                    f"   (If mean ≈ +90 → Scaling/logging issue, not critic problem)"
                )

            self.actor_optimizer.zero_grad()
            if self.actor_cnn_optimizer is not None:
                self.actor_cnn_optimizer.zero_grad()

            actor_loss.backward()

            if self.actor_cnn is not None:
                actor_grad_norm_before = torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.actor_cnn.parameters()),
                    max_norm=float('inf'),
                    norm_type=2.0
                ).item()

                if self.total_it % 100 == 0:
                    self.logger.debug(f"  Actor gradient norm BEFORE clip: {actor_grad_norm_before:.4f}")

                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.actor_cnn.parameters()),
                    max_norm=self.actor_max_norm,
                    norm_type=2.0
                )

                actor_grad_norm_after = sum(
                    p.grad.norm().item() ** 2 for p in list(self.actor.parameters()) + list(self.actor_cnn.parameters()) if p.grad is not None
                ) ** 0.5

                if self.total_it % 100 == 0:
                    self.logger.debug(
                        f"  Actor gradient norm AFTER clip: {actor_grad_norm_after:.4f} "
                        f"(max={self.actor_max_norm})")
                    if (np.isfinite(self.actor_max_norm)
                            and actor_grad_norm_after > self.actor_max_norm * 1.01):
                        self.logger.warning(
                            f"  CLIPPING FAILED! Actor grad {actor_grad_norm_after:.4f} > "
                            f"{self.actor_max_norm}")
            else:
                actor_grad_norm_before = 0.0
                actor_grad_norm_after = 0.0
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.actor_max_norm,
                    norm_type=2.0
                )
            if self.logger.isEnabledFor(logging.DEBUG) and self.total_it % 100 == 0:
                actor_grad_norm = sum(
                    p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None
                )
                if self.actor_cnn is not None:
                    actor_cnn_grad_norm = sum(
                        p.grad.norm().item() for p in self.actor_cnn.parameters() if p.grad is not None
                    )
                    self.logger.debug(
                        f"   TRAINING STEP {self.total_it} - ACTOR GRADIENTS (AFTER CLIPPING max_norm=1.0):\n"
                        f"   Actor grad norm: {actor_grad_norm:.4f} (expected: <1.0)\n"
                        f"   Actor CNN grad norm: {actor_cnn_grad_norm:.4f} (expected: <1.0)"
                    )
                else:
                    self.logger.debug(
                        f"   TRAINING STEP {self.total_it} - ACTOR GRADIENTS (AFTER CLIPPING max_norm=1.0):\n"
                        f"   Actor grad norm: {actor_grad_norm:.4f} (expected: <1.0)"
                    )

            metrics['debug/actor_grad_norm_BEFORE_clip'] = actor_grad_norm_before
            metrics['debug/actor_grad_norm_AFTER_clip'] = actor_grad_norm_after
            metrics['debug/actor_grad_clip_ratio'] = actor_grad_norm_after / max(actor_grad_norm_before, 1e-8)

            if self.actor_cnn is not None:
                actor_cnn_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_cnn.parameters(),
                    max_norm=float('inf'),
                    norm_type=2.0
                ).item()
                metrics['actor_cnn_grad_norm'] = actor_cnn_grad_norm
                metrics['debug/actor_cnn_grad_norm_AFTER_clip'] = actor_cnn_grad_norm

            actor_mlp_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=float('inf'),
                norm_type=2.0
            ).item()
            metrics['actor_mlp_grad_norm'] = actor_mlp_grad_norm
            metrics['debug/actor_mlp_grad_norm_AFTER_clip'] = actor_mlp_grad_norm

            if self.cnn_diagnostics is not None:
                self.cnn_diagnostics.capture_gradients()

                if self.total_it % 100 == 0:
                    self._log_detailed_gradient_flow(self.actor_cnn, "actor_cnn")

            if self.cnn_diagnostics is not None and self.use_dict_buffer and self.actor_cnn is not None:
                with torch.no_grad():
                    sample_features = self.actor_cnn(obs_dict['image'])
                    self.cnn_diagnostics.capture_features(sample_features, name="actor_update")

                    if self.total_it % 100 == 0:
                        self._log_feature_diversity(sample_features, "actor_cnn")

            self.actor_optimizer.step()

            if self.cnn_diagnostics is not None and self.actor_cnn is not None:
                self.cnn_diagnostics.capture_weights()

                if self.total_it % 1000 == 0:
                    self._log_weight_statistics(self.actor_cnn, "actor_cnn")

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )

            metrics['actor_loss'] = actor_loss.item()
            metrics.update(caps_metrics)

            metrics['debug/actor_q_mean'] = actor_q_values.mean().item()
            metrics['debug/actor_q_std'] = actor_q_values.std().item()
            metrics['debug/actor_q_min'] = actor_q_values.min().item()
            metrics['debug/actor_q_max'] = actor_q_values.max().item()

        return metrics

    def _caps_terms(
        self,
        pi_s: torch.Tensor,
        state_for_actor: torch.Tensor,
        next_state_for_actor: torch.Tensor,
        vector_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        zero = pi_s.new_zeros(())
        temporal = zero
        if self.caps_lambda_temporal > 0.0:
            pi_next = self.actor(next_state_for_actor)
            temporal = 0.5 * (pi_next - pi_s).pow(2).sum(dim=1).mean()
        spatial = zero
        if self.caps_lambda_spatial > 0.0:
            noise = torch.zeros_like(state_for_actor)
            noise[:, -vector_dim:] = torch.randn_like(state_for_actor[:, -vector_dim:]) * self.caps_sigma_vector
            pi_bar = self.actor(state_for_actor + noise)
            spatial = 0.5 * (pi_bar - pi_s).pow(2).sum(dim=1).mean()
        return temporal, spatial

    def save_checkpoint(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'total_it': self.total_it,

            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),

            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),

            'config': self.config,
            'use_dict_buffer': self.use_dict_buffer,

            'discount': self.discount,
            'tau': self.tau,
            'policy_freq': self.policy_freq,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'max_action': self.max_action,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }

        if self.use_dict_buffer:
            if self.actor_cnn is not None:
                checkpoint['actor_cnn_state_dict'] = self.actor_cnn.state_dict()
                print(f"  Saving actor CNN state ({len(checkpoint['actor_cnn_state_dict'])} layers)")
            else:
                checkpoint['actor_cnn_state_dict'] = None

            if self.critic_cnn is not None:
                checkpoint['critic_cnn_state_dict'] = self.critic_cnn.state_dict()
                print(f"  Saving critic CNN state ({len(checkpoint['critic_cnn_state_dict'])} layers)")
            else:
                checkpoint['critic_cnn_state_dict'] = None

            if self.actor_cnn_optimizer is not None:
                checkpoint['actor_cnn_optimizer_state_dict'] = self.actor_cnn_optimizer.state_dict()
                print(f"  Saving actor CNN optimizer state")
            else:
                checkpoint['actor_cnn_optimizer_state_dict'] = None

            if self.critic_cnn_optimizer is not None:
                checkpoint['critic_cnn_optimizer_state_dict'] = self.critic_cnn_optimizer.state_dict()
                print(f"  Saving critic CNN optimizer state")
            else:
                checkpoint['critic_cnn_optimizer_state_dict'] = None

        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        if self.use_dict_buffer:
            print(f"  Includes SEPARATE actor_cnn and critic_cnn states (Phase 21 fix)")

    def load_checkpoint(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        if checkpoint.get('use_dict_buffer', False):
            if 'actor_cnn_state_dict' in checkpoint and checkpoint['actor_cnn_state_dict'] is not None:
                if self.actor_cnn is not None:
                    self.actor_cnn.load_state_dict(checkpoint['actor_cnn_state_dict'])
                    print(f"Actor CNN state restored ({len(checkpoint['actor_cnn_state_dict'])} layers)")
                else:
                    print(f"Actor CNN state in checkpoint but agent.actor_cnn is None")
            else:
                print(f"No actor CNN state in checkpoint")

            if 'critic_cnn_state_dict' in checkpoint and checkpoint['critic_cnn_state_dict'] is not None:
                if self.critic_cnn is not None:
                    self.critic_cnn.load_state_dict(checkpoint['critic_cnn_state_dict'])
                    print(f"Critic CNN state restored ({len(checkpoint['critic_cnn_state_dict'])} layers)")
                else:
                    print(f"Critic CNN state in checkpoint but agent.critic_cnn is None")
            else:
                print(f"No critic CNN state in checkpoint")

            if 'actor_cnn_optimizer_state_dict' in checkpoint and checkpoint['actor_cnn_optimizer_state_dict'] is not None:
                if self.actor_cnn_optimizer is not None:
                    self.actor_cnn_optimizer.load_state_dict(checkpoint['actor_cnn_optimizer_state_dict'])
                    print(f"Actor CNN optimizer restored")
                else:
                    print(f"Actor CNN optimizer state in checkpoint but agent.actor_cnn_optimizer is None")
            else:
                print(f"No actor CNN optimizer state in checkpoint")

            if 'critic_cnn_optimizer_state_dict' in checkpoint and checkpoint['critic_cnn_optimizer_state_dict'] is not None:
                if self.critic_cnn_optimizer is not None:
                    self.critic_cnn_optimizer.load_state_dict(checkpoint['critic_cnn_optimizer_state_dict'])
                    print(f"Critic CNN optimizer restored")
                else:
                    print(f"Critic CNN optimizer state in checkpoint but agent.critic_cnn_optimizer is None")
            else:
                print(f"No critic CNN optimizer state in checkpoint")

        self.total_it = checkpoint['total_it']

        print(f"Checkpoint loaded from {filepath}")
        print(f"  Resumed at iteration: {self.total_it}")
        if checkpoint.get('use_dict_buffer', False):
            print(f"  SEPARATE CNNs restored (Phase 21 fix)")

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            'total_iterations': self.total_it,
            'is_training': self.total_it >= self.start_timesteps,
            'exploration_phase': self.total_it < self.start_timesteps,

            'replay_buffer_size': len(self.replay_buffer),
            'replay_buffer_full': self.replay_buffer.is_full(),
            'buffer_utilization': len(self.replay_buffer) / self.replay_buffer.max_size,
            'buffer_max_size': self.replay_buffer.max_size,
            'use_dict_buffer': self.use_dict_buffer,

            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr'],

            'discount': self.discount,
            'tau': self.tau,
            'policy_freq': self.policy_freq,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'max_action': self.max_action,
            'start_timesteps': self.start_timesteps,
            'batch_size': self.batch_size,
            'actor_param_mean': self._get_param_stat(self.actor.parameters(), 'mean'),
            'actor_param_std': self._get_param_stat(self.actor.parameters(), 'std'),
            'actor_param_max': self._get_param_stat(self.actor.parameters(), 'max'),
            'actor_param_min': self._get_param_stat(self.actor.parameters(), 'min'),

            'critic_param_mean': self._get_param_stat(self.critic.parameters(), 'mean'),
            'critic_param_std': self._get_param_stat(self.critic.parameters(), 'std'),
            'critic_param_max': self._get_param_stat(self.critic.parameters(), 'max'),
            'critic_param_min': self._get_param_stat(self.critic.parameters(), 'min'),

            'target_actor_param_mean': self._get_param_stat(self.actor_target.parameters(), 'mean'),
            'target_critic_param_mean': self._get_param_stat(self.critic_target.parameters(), 'mean'),

            'device': str(self.device)
        }

        if self.use_dict_buffer:
            stats.update({
                'actor_cnn_lr': self.actor_cnn_optimizer.param_groups[0]['lr'] if self.actor_cnn_optimizer else None,
                'critic_cnn_lr': self.critic_cnn_optimizer.param_groups[0]['lr'] if self.critic_cnn_optimizer else None,

                'actor_cnn_param_mean': self._get_param_stat(self.actor_cnn.parameters(), 'mean') if self.actor_cnn else None,
                'actor_cnn_param_std': self._get_param_stat(self.actor_cnn.parameters(), 'std') if self.actor_cnn else None,
                'actor_cnn_param_max': self._get_param_stat(self.actor_cnn.parameters(), 'max') if self.actor_cnn else None,
                'actor_cnn_param_min': self._get_param_stat(self.actor_cnn.parameters(), 'min') if self.actor_cnn else None,

                'critic_cnn_param_mean': self._get_param_stat(self.critic_cnn.parameters(), 'mean') if self.critic_cnn else None,
                'critic_cnn_param_std': self._get_param_stat(self.critic_cnn.parameters(), 'std') if self.critic_cnn else None,
                'critic_cnn_param_max': self._get_param_stat(self.critic_cnn.parameters(), 'max') if self.critic_cnn else None,
                'critic_cnn_param_min': self._get_param_stat(self.critic_cnn.parameters(), 'min') if self.critic_cnn else None,
            })

        return stats

    def get_action_stats(self) -> Dict[str, float]:
        if len(self.action_buffer) == 0:
            return {
                'action_steering_mean': 0.0,
                'action_steering_std': 0.0,
                'action_steering_min': 0.0,
                'action_steering_max': 0.0,
                'action_throttle_mean': 0.0,
                'action_throttle_std': 0.0,
                'action_throttle_min': 0.0,
                'action_throttle_max': 0.0,
            }

        actions = np.array(self.action_buffer)

        return {
            'action_steering_mean': float(actions[:, 0].mean()),
            'action_steering_std': float(actions[:, 0].std()),
            'action_steering_min': float(actions[:, 0].min()),
            'action_steering_max': float(actions[:, 0].max()),

            'action_throttle_mean': float(actions[:, 1].mean()),
            'action_throttle_std': float(actions[:, 1].std()),
            'action_throttle_min': float(actions[:, 1].min()),
            'action_throttle_max': float(actions[:, 1].max()),
        }

    def _get_param_stat(self, parameters, stat_type: str = 'mean') -> float:
        params_flat = torch.cat([p.data.flatten() for p in parameters if p.requires_grad])

        if len(params_flat) == 0:
            return 0.0

        if stat_type == 'mean':
            return float(torch.mean(params_flat).item())
        elif stat_type == 'std':
            return float(torch.std(params_flat).item())
        elif stat_type == 'max':
            return float(torch.max(params_flat).item())
        elif stat_type == 'min':
            return float(torch.min(params_flat).item())
        else:
            raise ValueError(f"Unknown stat_type: {stat_type}")

    def _log_detailed_gradient_flow(self, cnn: torch.nn.Module, network_name: str) -> None:
        if cnn is None:
            return

        gradient_norms = []
        vanishing_threshold = 1e-6
        exploding_threshold = 10.0

        for name, param in cnn.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)

                if grad_norm < vanishing_threshold:
                    status = "VANISHING"
                elif grad_norm > exploding_threshold:
                    status = "EXPLODING"
                else:
                    status = "OK"

                self.logger.debug(
                    f"[{network_name}] Gradient {name}: {grad_norm:.6f} {status}"
                )

        if gradient_norms:
            min_norm = min(gradient_norms)
            max_norm = max(gradient_norms)
            avg_norm = np.mean(gradient_norms)

            flow_ratio = gradient_norms[0] / gradient_norms[-1] if len(gradient_norms) > 1 else 1.0

            issues = []
            if min_norm < vanishing_threshold:
                issues.append("Vanishing gradients")
            if max_norm > exploding_threshold:
                issues.append("Exploding gradients")
            if flow_ratio < 0.1 or flow_ratio > 10.0:
                issues.append(f"Poor flow ratio ({flow_ratio:.2f})")

            health_status = " | ".join(issues) if issues else "HEALTHY"
            self.logger.debug(
                f"   [{network_name}] Gradient Flow Summary (Step {self.total_it}):\n"
                f"   Min: {min_norm:.6f}, Max: {max_norm:.6f}, Avg: {avg_norm:.6f}\n"
                f"   Flow ratio (first/last): {flow_ratio:.3f}\n"
                f"   Status: {health_status}"
            )

    def _log_feature_diversity(self, features: torch.Tensor, network_name: str) -> None:
        if features.shape[0] < 2:
            return

        try:
            features_cpu = features.detach().cpu()

            feature_corr = torch.corrcoef(features_cpu.T)

            n_features = feature_corr.shape[0]
            avg_corr = (feature_corr.abs().sum() - n_features) / (n_features * (n_features - 1))
            avg_corr = avg_corr.item()

            sparsity_threshold = 0.1
            sparsity = (features_cpu.abs() < sparsity_threshold).float().mean().item()

            _, s, _ = torch.svd(features_cpu)
            s_normalized = s / s.sum()
            entropy = -(s_normalized * torch.log(s_normalized + 1e-8)).sum().item()
            effective_rank = torch.exp(torch.tensor(entropy)).item()

            correlation_threshold = 0.3
            issues = []
            if avg_corr > 0.7:
                issues.append("High correlation (feature collapse)")
            elif avg_corr > correlation_threshold:
                issues.append("Moderate correlation")

            if sparsity < 0.05:
                issues.append("Too dense")
            elif sparsity > 0.5:
                issues.append("Too sparse")

            diversity_status = " | ".join(issues) if issues else "DIVERSE"

            self.logger.debug(
                f"   [{network_name}] Feature Diversity (Step {self.total_it}):\n"
                f"   Avg correlation: {avg_corr:.3f} (target: <{correlation_threshold})\n"
                f"   Sparsity: {sparsity*100:.1f}% (target: 10-30%)\n"
                f"   Effective rank: {effective_rank:.1f} / {n_features}\n"
                f"   Status: {diversity_status}"
            )

        except Exception as e:
            self.logger.warning(f"Could not compute feature diversity for {network_name}: {e}")

    def _log_weight_statistics(self, cnn: torch.nn.Module, network_name: str) -> None:
        if cnn is None:
            return

        self.logger.debug(f"   [{network_name}] Weight Statistics (Step {self.total_it}):")

        for name, param in cnn.named_parameters():
            if 'weight' in name:
                weights = param.data

                mean = weights.mean().item()
                std = weights.std().item()
                min_val = weights.min().item()
                max_val = weights.max().item()
                norm = weights.norm().item()

                if std < 1e-6:
                    status = "DEAD (zero variance)"
                elif norm > 100.0:
                    status = "EXCESSIVE (large norm)"
                else:
                    status = "OK"

                self.logger.debug(
                    f"   {name}:\n"
                    f"      Mean: {mean:.6f}, Std: {std:.6f}\n"
                    f"      Range: [{min_val:.6f}, {max_val:.6f}]\n"
                    f"      L2 norm: {norm:.3f} {status}"
                )

    def _log_learning_rate(self, optimizer: torch.optim.Optimizer, optimizer_name: str) -> None:
        if optimizer is None:
            return

        for idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']

            if lr < 1e-6:
                lr_status = "TOO LOW"
            elif lr > 1e-2:
                lr_status = "TOO HIGH"
            else:
                lr_status = "OK"

            self.logger.debug(
                f"   [{optimizer_name}] Learning Rate Group {idx}: {lr:.6e} {lr_status}"
            )

    def get_gradient_stats(self) -> Dict[str, float]:
        stats = {
            'actor_grad_norm': self._get_grad_norm(self.actor.parameters()),
            'critic_grad_norm': self._get_grad_norm(self.critic.parameters()),
        }

        if self.use_dict_buffer:
            stats.update({
                'actor_cnn_grad_norm': self._get_grad_norm(self.actor_cnn.parameters()) if self.actor_cnn else 0.0,
                'critic_cnn_grad_norm': self._get_grad_norm(self.critic_cnn.parameters()) if self.critic_cnn else 0.0,
            })

        return stats

    def _get_grad_norm(self, parameters) -> float:
        params_with_grad = [p for p in parameters if p.grad is not None]

        if len(params_with_grad) == 0:
            return 0.0

        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach()) for p in params_with_grad])
        )

        return float(total_norm.item())

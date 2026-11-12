#!/usr/bin/env python3
"""TD3 Agent Training Script for Autonomous Vehicle Navigation in CARLA"""

import sys
import os
sys.path.insert(0, '/workspace/av_td3_system')

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.agents.td3_agent import TD3Agent
from src.environment.carla_env import CARLANavigationEnv
from src.networks.cnn_extractor import NatureCNN


class TD3TrainingPipeline:

    def __init__(
        self,
        scenario: int = 0,
        seed: int = 42,
        max_timesteps: int = int(1e6),
        eval_freq: int = 5000,
        checkpoint_freq: int = 10000,
        num_eval_episodes: int = 5,
        carla_config_path: str = "config/carla_config.yaml",
        agent_config_path: str = "config/td3_config.yaml",
        training_config_path: str = "config/training_config.yaml",
        log_dir: str = "data/logs",
        checkpoint_dir: str = "data/checkpoints",
        debug: bool = False,
        device: str = 'cpu',
        carla_host: str = "localhost",
        carla_port: int = 2000,
        tm_port: int = 8000,
        env_isolation: bool = False,
        env_step_timeout: float = 60.0,
        env_reset_timeout: float = 300.0,
    ):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        self.scenario = scenario
        self.scenario = scenario
        self.seed = seed
        self.max_timesteps = max_timesteps
        self.eval_freq = eval_freq
        self.checkpoint_freq = checkpoint_freq
        self.num_eval_episodes = num_eval_episodes
        self.debug = debug

        self.logger = logging.getLogger(self.__class__.__name__)

        self.carla_config_path = carla_config_path
        self.agent_config_path = agent_config_path
        self.training_config_path = training_config_path

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*70)
        print("TD3 TRAINING PIPELINE - AUTONOMOUS VEHICLE NAVIGATION")
        print("="*70)
        print(f"\n[CONFIG] Loading configurations...")

        with open(carla_config_path, 'r') as f:
            self.carla_config = yaml.safe_load(f)

        with open(agent_config_path, 'r') as f:
            self.agent_config = yaml.safe_load(f)

        print(f"[CONFIG] CARLA config: {carla_config_path}")
        print(f"[CONFIG] Agent config: {agent_config_path}")
        print(f"[CONFIG] Scenario: {scenario} (0=20, 1=50, 2=100 NPCs)")

        npc_densities = [20, 50, 100]

        os.environ['CARLA_SCENARIO_INDEX'] = str(scenario)
        print(f"[CONFIG] Set CARLA_SCENARIO_INDEX={scenario}")

        if scenario < len(npc_densities):
            print(f"[CONFIG] Expected NPC count: {npc_densities[scenario]}")
        else:
            print(f"[WARNING] Invalid scenario index, will use default")

        self.carla_host, self.carla_port, self.tm_port = carla_host, carla_port, tm_port
        print(f"[CONFIG] CARLA {self.carla_host}:{self.carla_port} | "
              f"Traffic Manager port: {self.tm_port} (unified across phases within this run)")

        print(f"\n[ENVIRONMENT] Initializing CARLA environment...")
        if env_isolation:
            from src.environment.isolated_env import IsolatedEnv
            self.env = IsolatedEnv(
                "src.environment.carla_env:CARLANavigationEnv",
                dict(carla_config_path=carla_config_path, td3_config_path=agent_config_path,
                     training_config_path=training_config_path, host=self.carla_host, port=self.carla_port,
                     tm_port=self.tm_port),
                seed=seed, step_timeout=env_step_timeout, reset_timeout=env_reset_timeout)
            print(f"[ENVIRONMENT] PROCESS ISOLATION ON (M1): step time-out {env_step_timeout:.0f} s, "
                  f"reset time-out {env_reset_timeout:.0f} s")
        else:
            self.env = CARLANavigationEnv(
                carla_config_path,
                agent_config_path,
                training_config_path,
                host=self.carla_host,
                port=self.carla_port,
                tm_port=self.tm_port
            )
        print(f"[ENVIRONMENT] State space: {self.env.observation_space}")
        print(f"[ENVIRONMENT] Action space: {self.env.action_space}")

        print(f"\n[AGENT] Initializing TD3 agent...")
        if device == 'auto':
            agent_device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[AGENT] Device set to 'auto', detected: {agent_device}")
        else:
            agent_device = device
            print(f"[AGENT] Device explicitly set to: {agent_device}")

        if agent_device == 'cpu':
            print(f"[AGENT] Running on CPU to reserve GPU memory for CARLA simulator")

        print(f"[AGENT] Initializing SEPARATE NatureCNN feature extractors for actor and critic...")

        self.actor_cnn = NatureCNN(
            input_channels=4,
            num_frames=4,
            feature_dim=512
        ).to(agent_device)

        self.critic_cnn = NatureCNN(
            input_channels=4,
            num_frames=4,
            feature_dim=512
        ).to(agent_device)

        print(f"[AGENT] Initializing CNN weights (Kaiming for ReLU networks)...")
        self._initialize_cnn_weights()

        self.actor_cnn.train()
        self.critic_cnn.train()

        print(f"[AGENT] Actor CNN initialized on {agent_device} (id: {id(self.actor_cnn)})")
        print(f"[AGENT] Critic CNN initialized on {agent_device} (id: {id(self.critic_cnn)})")
        print(f"[AGENT] CNNs are SEPARATE instances: {id(self.actor_cnn) != id(self.critic_cnn)}")
        print(f"[AGENT] CNN architecture: 4×84×84 → Conv layers → 512 features")
        print(f"[AGENT] CNN training mode: ENABLED (weights will be updated during training)")

        num_waypoints = self.carla_config.get("route", {}).get("num_waypoints_ahead", 25)
        vector_dim = int(self.env.observation_space["vector"].shape[0])
        assert vector_dim >= 3 + num_waypoints * 2, (vector_dim, num_waypoints)
        state_dim = 512 + vector_dim

        start_timesteps = self.agent_config.get('training', {}).get('learning_starts',
                          self.agent_config.get('algorithm', {}).get('learning_starts', 10000))

        self.agent = TD3Agent(
            state_dim=state_dim,
            action_dim=2,
            max_action=1.0,
            actor_cnn=self.actor_cnn,
            critic_cnn=self.critic_cnn,
            use_dict_buffer=True,
            config=self.agent_config,
            device=agent_device,
            vector_dim=vector_dim,
        )


        print(f"[AGENT] CNN extractor initialized on {agent_device}")
        print(f"[AGENT] CNN passed to TD3Agent for end-to-end training")
        print(f"[AGENT] DictReplayBuffer enabled for gradient flow")


        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scenario_name = f"scenario_{scenario}_npcs_{npc_densities[scenario]}"
        algo_tag = self.agent_config.get('experiment', {}).get('algorithm', 'TD3')
        self.log_name = f"{algo_tag}_{scenario_name}_{timestamp}"
        self.log_path = self.log_dir / self.log_name
        self.log_path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.log_path))

        print(f"[LOGGING] TensorBoard logs: {self.log_path}")

        if self.debug:
            print(f"\n{'='*70}")
            print(f"[DEBUG MODE ENABLED]")
            print(f"{'='*70}")
            print(f"[DEBUG] Visual feedback enabled (OpenCV display)")
            print(f"[DEBUG] Press 'q' to quit, 'p' to pause/unpause")
            print(f"\n[DEBUG] CNN diagnostics enabled for training monitoring")
            print(f"[DEBUG] Tracking: gradient flow, weight updates, feature statistics")
            print(f"[DEBUG] TensorBoard metrics: cnn_diagnostics/*")
            print(f"[DEBUG] Console output: Every 1000 steps")
            print(f"{'='*70}\n")

            self.agent.enable_diagnostics()

            self.window_name = "TD3 Training - Debug View"
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1200, 600)
            self.paused = False

        self.episode_num = 0
        self._explore_action = None
        self.episode_reward = 0
        self.episode_timesteps = 0
        self.episode_collision_count = 0
        self.episode_lane_invasion_count = 0
        self.episode_steps_since_collision = 0

        self.in_eval_phase = False

        self.episode_reward_components = {
            'efficiency': 0.0,
            'lane_keeping': 0.0,
            'comfort': 0.0,
            'safety': 0.0,
            'progress': 0.0
        }

        self.training_rewards = []
        self.eval_rewards = []
        self.eval_success_rates = []
        self.eval_collisions = []
        self.eval_lane_invasions = []

        print(f"\n[INIT] Training pipeline ready!")
        print(f"[INIT] Max timesteps: {max_timesteps:,}")
        print(f"[INIT] Seed: {seed}")
        print(f"[INIT] Debug mode: {self.debug}")
        print(f"[INIT] Evaluation frequency: {eval_freq}")
        print(f"[INIT] Checkpoint frequency: {checkpoint_freq}")
        print(f"[INIT] Number of evaluation episodes: {num_eval_episodes}")
        print("="*70 + "\n")

    def _initialize_cnn_weights(self):
        print(f"[INIT] Initializing Actor CNN weights...")
        for module in self.actor_cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        print(f"[INIT] Initializing Critic CNN weights...")
        for module in self.critic_cnn.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        print("[AGENT] CNN weights initialized with Kaiming normal (optimized for ReLU)")

    def flatten_dict_obs(self, obs_dict, enable_grad=False):
        image = obs_dict['image']

        image_tensor = torch.from_numpy(image).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.agent.device)

        with torch.no_grad():
            image_features = self.actor_cnn(image_tensor)

        image_features = image_features.cpu().numpy().squeeze()

        vector = obs_dict['vector']

        flat_state = np.concatenate([image_features, vector]).astype(np.float32)

        return flat_state

    def _visualize_debug(self, obs_dict, action, reward, info, t):
        try:
            if 'image' not in obs_dict:
                return

            latest_frame = obs_dict['image'][-1]

            latest_frame_denorm = (latest_frame + 1.0) / 2.0

            frame_uint8 = (latest_frame_denorm * 255).astype(np.uint8)

            frame_resized = cv2.resize(frame_uint8, (800, 600), interpolation=cv2.INTER_LINEAR)

            display_frame = cv2.cvtColor(frame_resized, cv2.COLOR_GRAY2BGR)

            vehicle_state = info.get('vehicle_state', {})
            reward_breakdown = info.get('reward_breakdown', {})

            info_panel = np.zeros((600, 400, 3), dtype=np.uint8)

            cv2.putText(info_panel, "TD3 TRAINING - DEBUG", (10, 30),
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 2)

            y_offset = 70
            cv2.putText(info_panel, f"Timestep: {t}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"Episode: {self.episode_num}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"Episode Step: {self.episode_timesteps}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 40
            cv2.putText(info_panel, "ACTION:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Steering: {action[0]:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Throttle/Brake: {action[1]:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 40
            cv2.putText(info_panel, "VEHICLE STATE:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            speed_kmh = vehicle_state.get('velocity', 0) * 3.6
            cv2.putText(info_panel, f"  Speed: {speed_kmh:.1f} km/h", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Lat Dev: {vehicle_state.get('lateral_deviation', 0):.2f} m",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            cv2.putText(info_panel, f"  Head Err: {vehicle_state.get('heading_error', 0):.2f} rad",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 40
            cv2.putText(info_panel, "REWARD:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            reward_color = (0, 255, 0) if reward > 0 else (0, 0, 255)
            cv2.putText(info_panel, f"  Step: {reward:+.3f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, reward_color, 1)
            y_offset += 25
            episode_reward_color = (0, 255, 0) if self.episode_reward > 0 else (0, 0, 255) if self.episode_reward < 0 else (255, 255, 255)
            cv2.putText(info_panel, f"  Episode: {self.episode_reward:+.2f}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, episode_reward_color, 1)
            y_offset += 20
            print(f"[DEBUG CV2] Episode Reward: {self.episode_reward:+.2f} at Timestep: {t} and Episode: {self.episode_num} and Episode Step: {self.episode_timesteps}")
            cv2.putText(info_panel, f"    (Sum of {self.episode_timesteps} steps)", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

            y_offset += 30
            cv2.putText(info_panel, "Breakdown:", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            for component, (weight, value, weighted) in reward_breakdown.items():
                y_offset += 20
                text = f"  {component[:8]}: {weighted:+.2f}"
                cv2.putText(info_panel, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            y_offset += 30
            cv2.putText(info_panel, "PROGRESS:", (10, y_offset),
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            distance_to_goal = info.get('distance_to_goal', 0)
            cv2.putText(info_panel, f"  To Goal: {distance_to_goal:.1f}m", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            progress_pct = info.get('progress_percentage', 0)
            cv2.putText(info_panel, f"  Progress: {progress_pct:.1f}%", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            wp_idx = info.get('current_waypoint_idx', 0)
            cv2.putText(info_panel, f"  Waypoint: {wp_idx}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            y_offset += 30
            collision_color = (0, 0, 255) if self.episode_collision_count > 0 else (0, 255, 0)
            cv2.putText(info_panel, f"Collisions: {self.episode_collision_count}",
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, collision_color, 1)

            y_offset = 580
            cv2.putText(info_panel, "q:quit  p:pause", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            combined_frame = np.hstack([display_frame, info_panel])

            cv2.imshow(self.window_name, combined_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n[DEBUG] User requested quit (q pressed)")
                cv2.destroyAllWindows()
                self.env.close()
                import sys
                sys.exit(0)
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"[DEBUG] Paused: {self.paused}")

            while self.paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p'):
                    self.paused = False
                    print("[DEBUG] Resumed")
                elif key == ord('q'):
                    print("\n[DEBUG] User requested quit (q pressed)")
                    cv2.destroyAllWindows()
                    self.env.close()
                    import sys
                    sys.exit(0)

        except Exception as e:
            print(f"[DEBUG] Visualization error: {e}")

    def train(self):
        print(f"[TRAINING] Starting training loop...")
        print(f"[TRAINING] Initializing CARLA environment (spawning actors)...")
        print(f"[TRAINING] This may take 1-5 minutes on first reset. Please be patient...")
        print(f"[TRAINING] Connecting to CARLA server...")

        import time
        start_reset = time.time()

        obs_dict, reset_info = self.env.reset()

        reset_duration = time.time() - start_reset
        print(f"[TRAINING] Environment initialized successfully in {reset_duration:.1f} seconds!")
        print(f"[TRAINING] Episode {reset_info.get('episode', 1)}: Route {reset_info.get('route_length_m', 0):.0f}m, NPCs {reset_info.get('npc_count', 0)}")
        print(f"[TRAINING] Actors spawned, sensors ready")
        print(f"[TRAINING] Beginning training from timestep 1 to {self.max_timesteps:,}")

        done = False

        print(f"\n{'='*70}")
        print(f"[DEBUG] CNN -> TD3 DATA FLOW VERIFICATION (Initialization)")
        print(f"{'='*70}")
        print(f"[DEBUG] Camera Input:")
        print(f"   Shape: {obs_dict['image'].shape}")
        print(f"   Range: [{obs_dict['image'].min():.3f}, {obs_dict['image'].max():.3f}]")
        print(f"   Mean: {obs_dict['image'].mean():.3f}, Std: {obs_dict['image'].std():.3f}")
        print(f"\n[DEBUG] Vector State (Kinematic + Waypoints):")
        print(f"   Shape: {obs_dict['vector'].shape}")
        print(f"   Velocity: {obs_dict['vector'][0]:.3f} m/s")
        print(f"   Lateral Deviation: {obs_dict['vector'][1]:.3f} m")
        print(f"   Heading Error: {obs_dict['vector'][2]:.3f} rad")
        print(f"   Waypoints: {obs_dict['vector'][3:23].shape} (was 10 waypoints × 2)")
        print(f"\n[DEBUG] Dict Observation Structure:")
        print(f"   Type: {type(obs_dict)}")
        print(f"   Keys: {list(obs_dict.keys())}")
        print(f"   Image shape: {obs_dict['image'].shape}")
        print(f"   Vector shape: {obs_dict['vector'].shape}")
        print(f"{'='*70}\n")

        start_timesteps = self.agent_config.get('algorithm', {}).get('learning_starts', 25000)
        batch_size = self.agent_config.get('algorithm', {}).get('batch_size', 256)

        print(f"\n[TRAINING PHASES]")
        print(f"  Phase 1 (Steps 1-{start_timesteps:,}): EXPLORATION (random actions, filling replay buffer)")
        print(f"  Phase 2 (Steps {start_timesteps+1:,}-{self.max_timesteps:,}): LEARNING (policy updates)")
        print(f"  Evaluation every {self.eval_freq:,} steps")
        print(f"  Checkpoints every {self.checkpoint_freq:,} steps")
        print(f"\n[PROGRESS] Training starting now - logging every 100 steps...\n")

        first_training_logged = False

        for t in range(1, int(self.max_timesteps) + 1):
            self.episode_timesteps += 1

            if t % 100 == 0:
                phase = "EXPLORATION" if t <= start_timesteps else "LEARNING"
                print(f"[{phase}] Processing step {t:6d}/{self.max_timesteps:,}. The t is {t:6d}...", flush=True)

            current_noise = 0.0

            if t < start_timesteps:
                repeat = self.agent_config.get('algorithm', {}).get(
                    'exploration_action_repeat', 5
                )
                if (self._explore_action is None) or (t % max(1, int(repeat)) == 0):
                    self._explore_action = np.array([
                        np.random.uniform(-1, 1),
                        np.random.uniform(-1, 1)
                    ])
                action = self._explore_action.copy()
            else:
                current_noise = self.agent_config.get('algorithm', {}).get(
                    'exploration_noise', 0.1)

                _aligned_log = (
                    (self.agent.total_it % 10 == 0) if t >= start_timesteps
                    else (t % 10 == 0)
                )
                if _aligned_log:
                    self.writer.add_scalar('train/exploration_noise', current_noise, t)

                    action_stats = self.agent.get_action_stats()
                    for key, value in action_stats.items():
                        self.writer.add_scalar(f'debug/{key}', value, t)

                    if t % 100 == 0 and self.debug:
                        self.logger.debug(
                            f"\n   ACTION STATISTICS (last 1000 actions):\n"
                            f"   ══════════════════════════════════════\n"
                            f"   STEERING:\n"
                            f"      Mean: {action_stats['action_steering_mean']:+.3f}  (expect ~0.0, no bias)\n"
                            f"      Std:  {action_stats['action_steering_std']:.3f}   (expect 0.1-0.3, exploration)\n"
                            f"      Range: [{action_stats['action_steering_min']:+.3f}, {action_stats['action_steering_max']:+.3f}]\n"
                            f"   ──────────────────────────────────────\n"
                            f"   THROTTLE/BRAKE:\n"
                            f"      Mean: {action_stats['action_throttle_mean']:+.3f}  (expect 0.0-0.5, forward bias)\n"
                            f"      Std:  {action_stats['action_throttle_std']:.3f}   (expect 0.1-0.3, exploration)\n"
                            f"      Range: [{action_stats['action_throttle_min']:+.3f}, {action_stats['action_throttle_max']:+.3f}]\n"
                            f"   ══════════════════════════════════════"
                        )


                if t % 100 == 0 and self.debug:
                    image_obs = obs_dict['image']
                    vector_obs = obs_dict['vector']
                    print(f"\n[DIAGNOSTIC][Step {t}] PRE-ACTION OBSERVATION:")
                    print(f"  Image: shape={image_obs.shape}, range=[{image_obs.min():.3f}, {image_obs.max():.3f}]")
                    print(f"  Vector: shape={vector_obs.shape}, range=[{vector_obs.min():.3f}, {vector_obs.max():.3f}]")
                    print(f"  Exploration noise: {current_noise:.4f} (decaying exponentially)")
                    print(f"  Phase: {'EXPLORATION' if t < start_timesteps else 'LEARNING'}")

                action = self.agent.select_action(
                    obs_dict,
                    noise=current_noise,
                    deterministic=False
                )

                if t % 100 == 0 and self.debug:
                    action_stats = self.agent.get_action_stats() if hasattr(self.agent, 'get_action_stats') else None
                    print(f"\n[DIAGNOSTIC][Step {t}] POST-ACTION OUTPUT:")
                    print(f"  Current action: steer={action[0]:+.3f}, throttle/brake={action[1]:+.3f}")
                    if action_stats:
                        print(f"  Rolling stats (last 100): steer_mean={action_stats.get('steer_mean', 0):+.3f}, "
                              f"steer_std={action_stats.get('steer_std', 0):.3f}")
                        print(f"BIAS DETECTED!" if abs(action_stats.get('steer_mean', 0)) > 0.3 else "  Balanced steering")
                    else:
                        print(f"Action statistics not available (action_buffer may be empty)")

            if self.logger.isEnabledFor(logging.INFO):
                self.logger.info(
                    f"[ACTION_EXECUTION] Step {t}: "
                    f"steer={action[0]:+.4f}, throttle={action[1]:+.4f}, "
                    f"noise={current_noise:.4f}, "
                    f"phase={'EXPLORE' if t < start_timesteps else 'TRAIN'}"
                )

            next_obs_dict, reward, done, truncated, info = self.env.step(action)

            if self.logger.isEnabledFor(logging.INFO):
                vehicle_state = info.get('vehicle_state', {})
                self.logger.info(
                    f"[ACTION_RESULT] Step {t}: "
                    f"Action=[{action[0]:+.4f}, {action[1]:+.4f}] → "
                    f"Reward={reward:+.3f} | "
                    f"Velocity={vehicle_state.get('velocity', 0):.1f}km/h, "
                    f"Lateral_dev={vehicle_state.get('lateral_deviation', 0):+.2f}m, "
                    f"Heading_err={vehicle_state.get('heading_error', 0):+.2f}°"
                )

            if t % 100 == 0 and self.debug:
                reward_breakdown = info.get('reward_breakdown', {})
                vehicle_state = info.get('vehicle_state', {})
                print(f"\n[DIAGNOSTIC][Step {t}] POST-STEP REWARD:")
                print(f"  Total reward: {reward:+.3f}")
                if reward_breakdown:
                    efficiency_weighted = reward_breakdown.get('efficiency', (0,0,0))[2]
                    lane_weighted = reward_breakdown.get('lane_keeping', (0,0,0))[2]
                    progress_weighted = reward_breakdown.get('progress', (0,0,0))[2]

                    print(f"  Components: efficiency={efficiency_weighted:+.2f}, "
                          f"lane_keeping={lane_weighted:+.2f}, "
                          f"progress={progress_weighted:+.2f}")

                    if progress_weighted > 0 and lane_weighted < 0:
                        print(f"  🚨 BUG SIGNATURE: progress↑ but lane_keeping↓ (conflicting incentives!)")
                    elif progress_weighted < 0 and lane_weighted > 0:
                        print(f"  🚨 BUG SIGNATURE: progress↓ but lane_keeping↑ (conflicting incentives!)")
                    else:
                        print(f"  ✓ Aligned incentives: progress and lane_keeping have same sign")
                print(f"  Vehicle: vel={vehicle_state.get('velocity', 0):.1f} km/h, "
                      f"lateral_dev={vehicle_state.get('lateral_deviation', 0):+.2f}m")
                print(f"  Episode: step={self.episode_timesteps}, done={done}, truncated={truncated}")


            if t % 100 == 0 and self.debug:
                with torch.no_grad():
                    image_tensor = torch.FloatTensor(next_obs_dict['image']).unsqueeze(0).to(self.agent.device)
                    cnn_features = self.actor_cnn(image_tensor).cpu().numpy().squeeze()

                print(f"\n[DEBUG][Step {t}] CNN Feature Stats:")
                print(f"  L2 Norm: {np.linalg.norm(cnn_features):.3f}")
                print(f"  Mean: {cnn_features.mean():.3f}, Std: {cnn_features.std():.3f}")
                print(f"  Range: [{cnn_features.min():.3f}, {cnn_features.max():.3f}]")
                print(f"  Action: [{action[0]:.3f}, {action[1]:.3f}] (steering, throttle/brake)")

            if self.debug:
                self._visualize_debug(next_obs_dict, action, reward, info, t)

            if t % 100 == 0 and self.debug:
                vehicle_state = info.get('vehicle_state', {})
                reward_breakdown = info.get('reward_breakdown', {})

                vector_obs = next_obs_dict.get('vector', np.array([]))
                image_obs = next_obs_dict.get('image', np.array([]))

                velocity = vector_obs[0] if len(vector_obs) > 0 else 0.0
                lat_dev = vector_obs[1] if len(vector_obs) > 1 else 0.0
                heading_err = vector_obs[2] if len(vector_obs) > 2 else 0.0

                waypoints_flat = vector_obs[3:23] if len(vector_obs) >= 23 else []

                if len(image_obs.shape) == 3:
                    img_mean = np.mean(image_obs)
                    img_std = np.std(image_obs)
                    img_min = np.min(image_obs)
                    img_max = np.max(image_obs)
                else:
                    img_mean = img_std = img_min = img_max = 0.0

                eff_tuple = reward_breakdown.get('efficiency', (0, 0, 0))
                lane_tuple = reward_breakdown.get('lane_keeping', (0, 0, 0))
                comfort_tuple = reward_breakdown.get('comfort', (0, 0, 0))
                safety_tuple = reward_breakdown.get('safety', (0, 0, 0))
                progress_tuple = reward_breakdown.get('progress', (0, 0, 0))

                eff_reward = eff_tuple[2] if isinstance(eff_tuple, tuple) else 0.0
                lane_reward = lane_tuple[2] if isinstance(lane_tuple, tuple) else 0.0
                comfort_reward = comfort_tuple[2] if isinstance(comfort_tuple, tuple) else 0.0
                safety_reward = safety_tuple[2] if isinstance(safety_tuple, tuple) else 0.0
                progress_reward = progress_tuple[2] if isinstance(progress_tuple, tuple) else 0.0

                print(
                    f"\n[DEBUG Step {t:4d}] "
                    f"Act=[steer:{action[0]:+.3f}, thr/brk:{action[1]:+.3f}] | "
                    f"Rew={reward:+7.2f} | "
                    f"Speed={vehicle_state.get('velocity', 0)*3.6:5.1f} km/h | "
                    f"LatDev={vehicle_state.get('lateral_deviation', 0):+.2f}m | "
                    f"Collisions={self.episode_collision_count}"
                )

                print(
                    f"   [Reward] Efficiency={eff_reward:+.2f} | "
                    f"Lane={lane_reward:+.2f} | "
                    f"Comfort={comfort_reward:+.2f} | "
                    f"Safety={safety_reward:+.2f} | "
                    f"Progress={progress_reward:+.2f}"
                )

                if len(waypoints_flat) >= 6:
                    wp1_x, wp1_y = waypoints_flat[0], waypoints_flat[1]
                    wp2_x, wp2_y = waypoints_flat[2], waypoints_flat[3]
                    wp3_x, wp3_y = waypoints_flat[4], waypoints_flat[5]

                    dist1 = np.sqrt(wp1_x**2 + wp1_y**2)
                    dist2 = np.sqrt(wp2_x**2 + wp2_y**2)
                    dist3 = np.sqrt(wp3_x**2 + wp3_y**2)

                    print(
                        f"   [Waypoints] (vehicle frame): "
                        f"WP1=[{wp1_x:+6.1f}, {wp1_y:+6.1f}]m (d={dist1:5.1f}m) | "
                        f"WP2=[{wp2_x:+6.1f}, {wp2_y:+6.1f}]m (d={dist2:5.1f}m) | "
                        f"WP3=[{wp3_x:+6.1f}, {wp3_y:+6.1f}]m (d={dist3:5.1f}m)"
                    )

                print(
                    f"   [Image] shape={image_obs.shape} | "
                    f"mean={img_mean:.3f} | std={img_std:.3f} | "
                    f"range=[{img_min:.3f}, {img_max:.3f}]"
                )

                print(
                    f"   [State] velocity={velocity:.2f} m/s | "
                    f"lat_dev={lat_dev:+.3f}m | "
                    f"heading_err={heading_err:+.3f} rad ({np.degrees(heading_err):+.1f}°) | "
                    f"vector_dim={len(vector_obs)}"
                )

            episode_reward_before = self.episode_reward
            self.episode_reward += reward
            episode_reward_after = self.episode_reward

            reward_breakdown = info.get('reward_breakdown', {})

            eff_tuple = reward_breakdown.get('efficiency', (0, 0, 0))
            lane_tuple = reward_breakdown.get('lane_keeping', (0, 0, 0))
            comfort_tuple = reward_breakdown.get('comfort', (0, 0, 0))
            safety_tuple = reward_breakdown.get('safety', (0, 0, 0))
            progress_tuple = reward_breakdown.get('progress', (0, 0, 0))

            eff_reward = eff_tuple[2] if isinstance(eff_tuple, tuple) else 0.0
            lane_reward = lane_tuple[2] if isinstance(lane_tuple, tuple) else 0.0
            comfort_reward = comfort_tuple[2] if isinstance(comfort_tuple, tuple) else 0.0
            safety_reward = safety_tuple[2] if isinstance(safety_tuple, tuple) else 0.0
            progress_reward = progress_tuple[2] if isinstance(progress_tuple, tuple) else 0.0

            self.episode_reward_components['efficiency'] += eff_reward
            self.episode_reward_components['lane_keeping'] += lane_reward
            self.episode_reward_components['comfort'] += comfort_reward
            self.episode_reward_components['safety'] += safety_reward
            self.episode_reward_components['progress'] += progress_reward

            if abs(reward) > 0.01:
                self.logger.debug(
                    f"[REWARD ACCUMULATION] Episode {self.episode_num}, Step {self.episode_timesteps}: "
                    f"step_reward={reward:+.3f} | "
                    f"episode_reward_before={episode_reward_before:+.3f} | "
                    f"episode_reward_after={episode_reward_after:+.3f} | "
                    f"expected={episode_reward_before + reward:+.3f} | "
                    f"match={abs(episode_reward_after - (episode_reward_before + reward)) < 0.001}"
                )

            if (episode_reward_before < 0 and episode_reward_after > 0 and reward < 0) or \
               (episode_reward_before > 0 and episode_reward_after < 0 and reward > 0):
                self.logger.warning(
                    f"[REWARD SIGN ANOMALY] Episode {self.episode_num}, Step {self.episode_timesteps}: "
                    f"Episode reward changed sign unexpectedly! "
                    f"before={episode_reward_before:+.3f}, step={reward:+.3f}, after={episode_reward_after:+.3f}"
                )

            if reward < -1.0:
                collision_this_step = info.get('collision_count', 0) > 0
                lane_invasion_this_step = info.get('lane_invasion_count', 0) > 0
                termination_reason = info.get('termination_reason', 'running')
                offroad_termination = (termination_reason == 'off_road')

                vehicle_state = info.get('vehicle_state', {})
                lateral_dev = vehicle_state.get('lateral_deviation', 0.0)

                self.logger.info(
                    f"[NEGATIVE REWARD] Episode {self.episode_num}, Step {self.episode_timesteps}: "
                    f"Large negative step reward={reward:+.3f} | "
                    f"Episode total: {episode_reward_before:+.3f} → {episode_reward_after:+.3f} | "
                    f"Lateral_dev={lateral_dev:.2f}m | "
                    f"Collision={collision_this_step} | "
                    f"Lane_invasion={lane_invasion_this_step} | "
                    f"Offroad_termination={offroad_termination} | "
                    f"Status={termination_reason}"
                )

            self.episode_collision_count += info.get('collision_count', 0)
            self.episode_lane_invasion_count += info.get('lane_invasion_count', 0)
            self.episode_steps_since_collision = info.get('steps_since_collision', 0)

            done_bool = float(done)

            if info.get('termination_reason') != 'env_process_timeout':
                self.agent.replay_buffer.add(
                    obs_dict=obs_dict,
                    action=action,
                    next_obs_dict=next_obs_dict,
                    reward=reward,
                    done=done_bool,
                    info=info
                )
            else:
                self.logger.error(f"[M1] step {t}: environment process restarted "
                                  f"({info.get('env_process_restarts')} so far); synthetic transition NOT stored")

            obs_dict = next_obs_dict

            if t >= start_timesteps:
                if not first_training_logged:
                    print(f"\n{'='*70}")
                    print(f"[PHASE TRANSITION] Starting LEARNING phase at step {t:,}")
                    print(f"[PHASE TRANSITION] Replay buffer size: {len(self.agent.replay_buffer):,}")
                    print(f"[PHASE TRANSITION] Current episode step: {self.episode_timesteps}")
                    print(f"[PHASE TRANSITION] Episode reward: {self.episode_reward:+.2f}")

                    if done or truncated or self.episode_timesteps > 1:
                        print(f"[PHASE TRANSITION] ⚠️  Episode in progress detected!")
                        print(f"[PHASE TRANSITION] Forcing episode reset to start clean...")

                        self.logger.warning(
                            f"[PHASE TRANSITION] Abandoning Episode {self.episode_num} at step {self.episode_timesteps}: "
                            f"episode_reward={self.episode_reward:+.3f} (exploration phase contaminated)"
                        )

                        obs_dict, _ = self.env.reset()

                        self.episode_reward = 0
                        self.episode_timesteps = 0
                        self.episode_collision_count = 0
                        self.episode_lane_invasion_count = 0
                        self.episode_steps_since_collision = 0
                        self.episode_num += 1

                        for key in self.episode_reward_components:
                            self.episode_reward_components[key] = 0.0

                        print(f"[PHASE TRANSITION] ✓ Episode reset complete. Starting Episode {self.episode_num}")
                        print(f"[PHASE TRANSITION] Learning will begin with CLEAN episode from spawn")
                    else:
                        print(f"[PHASE TRANSITION] ✓ Episode just started, no reset needed")

                    print(f"[PHASE TRANSITION] Policy updates will now begin...")
                    print(f"{'='*70}\n")
                    first_training_logged = True

                metrics = self.agent.train(batch_size=batch_size)

                if self.agent.total_it % 10 == 0:
                    self.writer.add_scalar('train/critic_loss', metrics['critic_loss'], t)
                    self.writer.add_scalar('train/q1_value', metrics['q1_value'], t)
                    self.writer.add_scalar('train/q2_value', metrics['q2_value'], t)

                    if 'actor_loss' in metrics:
                        self.writer.add_scalar('train/actor_loss', metrics['actor_loss'], t)


                    for _k in ('debug/twin_q_abs_diff', 'debug/clipped_double_q_gap',
                               'debug/twin_q_disagree_frac', 'debug/twin_q_abs_diff_online'):
                        if _k in metrics:
                            self.writer.add_scalar(_k, metrics[_k], t)

                    for _k, _v in metrics.items():
                        if _k.startswith('actor_reg/'):
                            self.writer.add_scalar(_k, _v, t)

                    if 'debug/q1_std' in metrics:
                        self.writer.add_scalar('debug/q1_std', metrics['debug/q1_std'], t)
                        self.writer.add_scalar('debug/q1_min', metrics['debug/q1_min'], t)
                        self.writer.add_scalar('debug/q1_max', metrics['debug/q1_max'], t)
                    if 'debug/q2_std' in metrics:
                        self.writer.add_scalar('debug/q2_std', metrics['debug/q2_std'], t)
                        self.writer.add_scalar('debug/q2_min', metrics['debug/q2_min'], t)
                        self.writer.add_scalar('debug/q2_max', metrics['debug/q2_max'], t)

                    if 'debug/target_q_mean' in metrics:
                        self.writer.add_scalar('debug/target_q_mean', metrics['debug/target_q_mean'], t)
                        self.writer.add_scalar('debug/target_q_std', metrics['debug/target_q_std'], t)
                        self.writer.add_scalar('debug/target_q_min', metrics['debug/target_q_min'], t)
                        self.writer.add_scalar('debug/target_q_max', metrics['debug/target_q_max'], t)

                    if 'debug/td_error_q1' in metrics:
                        self.writer.add_scalar('debug/td_error_q1', metrics['debug/td_error_q1'], t)
                        self.writer.add_scalar('debug/td_error_q2', metrics['debug/td_error_q2'], t)

                    if 'debug/reward_mean' in metrics:
                        self.writer.add_scalar('debug/reward_mean', metrics['debug/reward_mean'], t)
                        self.writer.add_scalar('debug/reward_std', metrics['debug/reward_std'], t)
                        self.writer.add_scalar('debug/reward_min', metrics['debug/reward_min'], t)
                        self.writer.add_scalar('debug/reward_max', metrics['debug/reward_max'], t)

                    if 'debug/done_ratio' in metrics:
                        self.writer.add_scalar('debug/done_ratio', metrics['debug/done_ratio'], t)
                        self.writer.add_scalar('debug/effective_discount', metrics['debug/effective_discount'], t)

                    if 'debug/actor_q_mean' in metrics:
                        self.writer.add_scalar('debug/actor_q_mean', metrics['debug/actor_q_mean'], t)
                        self.writer.add_scalar('debug/actor_q_std', metrics['debug/actor_q_std'], t)
                        self.writer.add_scalar('debug/actor_q_min', metrics['debug/actor_q_min'], t)
                        self.writer.add_scalar('debug/actor_q_max', metrics['debug/actor_q_max'], t)


                    if 'debug/actor_grad_norm_BEFORE_clip' in metrics:
                        self.writer.add_scalar('debug/actor_grad_norm_BEFORE_clip', metrics['debug/actor_grad_norm_BEFORE_clip'], t)
                    if 'debug/actor_grad_norm_AFTER_clip' in metrics:
                        self.writer.add_scalar('debug/actor_grad_norm_AFTER_clip', metrics['debug/actor_grad_norm_AFTER_clip'], t)
                    if 'debug/actor_grad_clip_ratio' in metrics:
                        self.writer.add_scalar('debug/actor_grad_clip_ratio', metrics['debug/actor_grad_clip_ratio'], t)

                    if 'debug/critic_grad_norm_BEFORE_clip' in metrics:
                        self.writer.add_scalar('debug/critic_grad_norm_BEFORE_clip', metrics['debug/critic_grad_norm_BEFORE_clip'], t)
                    if 'debug/critic_grad_norm_AFTER_clip' in metrics:
                        self.writer.add_scalar('debug/critic_grad_norm_AFTER_clip', metrics['debug/critic_grad_norm_AFTER_clip'], t)
                    if 'debug/critic_grad_clip_ratio' in metrics:
                        self.writer.add_scalar('debug/critic_grad_clip_ratio', metrics['debug/critic_grad_clip_ratio'], t)

                    if 'debug/actor_cnn_grad_norm_AFTER_clip' in metrics:
                        self.writer.add_scalar('debug/actor_cnn_grad_norm_AFTER_clip', metrics['debug/actor_cnn_grad_norm_AFTER_clip'], t)
                    if 'debug/critic_cnn_grad_norm_AFTER_clip' in metrics:
                        self.writer.add_scalar('debug/critic_cnn_grad_norm_AFTER_clip', metrics['debug/critic_cnn_grad_norm_AFTER_clip'], t)

                    if 'debug/actor_mlp_grad_norm_AFTER_clip' in metrics:
                        self.writer.add_scalar('debug/actor_mlp_grad_norm_AFTER_clip', metrics['debug/actor_mlp_grad_norm_AFTER_clip'], t)
                    if 'debug/critic_mlp_grad_norm_AFTER_clip' in metrics:
                        self.writer.add_scalar('debug/critic_mlp_grad_norm_AFTER_clip', metrics['debug/critic_mlp_grad_norm_AFTER_clip'], t)

                    if 'actor_cnn_grad_norm' in metrics:
                        actor_cnn_grad = metrics['actor_cnn_grad_norm']
                        self.writer.add_scalar('gradients/actor_cnn_norm', actor_cnn_grad, t)

                        _a_bound = getattr(self.agent, 'actor_max_norm', float('inf'))
                        if actor_cnn_grad > 2.0 * _a_bound:
                            self.writer.add_scalar('alerts/gradient_explosion_critical', 1, t)
                            print(f"\n{'!'*70}")
                            print(f"   CRITICAL ALERT: Actor CNN gradient violation detected!")
                            print(f"   Step: {t:,}")
                            print(f"   Actor CNN grad norm: {actor_cnn_grad:.4f}")
                            print(f"   Bound in force: {_a_bound}, alert at 2x")
                            print(f"   This means the clipper did not apply the bound it was given.")
                            print(f"{'!'*70}\n")
                        elif actor_cnn_grad > 1.5 * _a_bound:
                            self.writer.add_scalar('alerts/gradient_explosion_warning', 1, t)
                            print(f"\n WARNING: Actor CNN gradient elevated at step {t:,}: "
                                  f"{actor_cnn_grad:.4f} (bound in force: {_a_bound})")
                        else:
                            self.writer.add_scalar('alerts/gradient_explosion_critical', 0, t)
                            self.writer.add_scalar('alerts/gradient_explosion_warning', 0, t)

                    if 'critic_cnn_grad_norm' in metrics:
                        critic_cnn_grad = metrics['critic_cnn_grad_norm']
                        self.writer.add_scalar('gradients/critic_cnn_norm', critic_cnn_grad, t)

                        _c_bound = getattr(self.agent, 'critic_max_norm', float('inf'))
                        if critic_cnn_grad > 20.0 * _c_bound:
                            self.writer.add_scalar('alerts/critic_gradient_explosion_critical', 1, t)
                            print(f"\n{'!'*70}")
                            print(f"   CRITICAL ALERT: Critic CNN gradient violation detected!")
                            print(f"   Step: {t:,}")
                            print(f"   Critic CNN grad norm: {critic_cnn_grad:.4f}")
                            print(f"   Bound in force: {_c_bound}, alert at 2x")
                            print(f"   This means the clipper did not apply the bound it was given.")
                            print(f"{'!'*70}\n")
                        elif critic_cnn_grad > 15.0 * _c_bound:
                            self.writer.add_scalar('alerts/critic_gradient_explosion_warning', 1, t)
                            print(f"\n   WARNING: Critic CNN gradient elevated at step {t:,}: "
                                  f"{critic_cnn_grad:.4f} (bound in force: {_c_bound})")
                        else:
                            self.writer.add_scalar('alerts/critic_gradient_explosion_critical', 0, t)
                            self.writer.add_scalar('alerts/critic_gradient_explosion_warning', 0, t)

                    if 'actor_mlp_grad_norm' in metrics:
                        self.writer.add_scalar('gradients/actor_mlp_norm', metrics['actor_mlp_grad_norm'], t)

                    if 'critic_mlp_grad_norm' in metrics:
                        self.writer.add_scalar('gradients/critic_mlp_norm', metrics['critic_mlp_grad_norm'], t)

                    if self.debug and self.agent.cnn_diagnostics is not None:
                        self.agent.cnn_diagnostics.log_to_tensorboard(self.writer, t)

                        if t % 1000 == 0:
                            print(f"\n{'='*70}")
                            print(f"[CNN DIAGNOSTICS] Step {t:,}")
                            print(f"{'='*70}")
                            self.agent.print_diagnostics(max_history=1000)
                            print(f"{'='*70}\n")

                if t % 10 == 0:
                    agent_stats = self.agent.get_stats()

                    self.writer.add_scalar('agent/total_iterations', agent_stats['total_iterations'], t)
                    self.writer.add_scalar('agent/is_training', int(agent_stats['is_training']), t)

                    self.writer.add_scalar('agent/buffer_utilization', agent_stats['buffer_utilization'], t)

                    self.writer.add_scalar('agent/actor_lr', agent_stats['actor_lr'], t)
                    self.writer.add_scalar('agent/critic_lr', agent_stats['critic_lr'], t)

                    if agent_stats.get('actor_cnn_lr') is not None:
                        self.writer.add_scalar('agent/actor_cnn_lr', agent_stats['actor_cnn_lr'], t)
                        self.writer.add_scalar('agent/critic_cnn_lr', agent_stats['critic_cnn_lr'], t)

                    self.writer.add_scalar('agent/actor_param_mean', agent_stats['actor_param_mean'], t)
                    self.writer.add_scalar('agent/actor_param_std', agent_stats['actor_param_std'], t)
                    self.writer.add_scalar('agent/critic_param_mean', agent_stats['critic_param_mean'], t)
                    self.writer.add_scalar('agent/critic_param_std', agent_stats['critic_param_std'], t)

                    if agent_stats.get('actor_cnn_param_mean') is not None:
                        self.writer.add_scalar('agent/actor_cnn_param_mean', agent_stats['actor_cnn_param_mean'], t)
                        self.writer.add_scalar('agent/actor_cnn_param_std', agent_stats['actor_cnn_param_std'], t)
                        self.writer.add_scalar('agent/critic_cnn_param_mean', agent_stats['critic_cnn_param_mean'], t)
                        self.writer.add_scalar('agent/critic_cnn_param_std', agent_stats['critic_cnn_param_std'], t)

                    if t % 10000 == 0:
                        print(f"\n{'='*70}")
                        print(f"[AGENT STATISTICS] Step {t:,}")
                        print(f"{'='*70}")
                        print(f"Training Phase: {'LEARNING' if agent_stats['is_training'] else 'EXPLORATION'}")
                        print(f"Buffer Utilization: {agent_stats['buffer_utilization']:.1%}")
                        print(f"Learning Rates:")
                        print(f"  Actor:  {agent_stats['actor_lr']:.6f}")
                        print(f"  Critic: {agent_stats['critic_lr']:.6f}")
                        if agent_stats.get('actor_cnn_lr') is not None:
                            print(f"  Actor CNN:  {agent_stats['actor_cnn_lr']:.6f}")
                            print(f"  Critic CNN: {agent_stats['critic_cnn_lr']:.6f}")
                        print(f"Network Stats:")
                        print(f"  Actor  - mean: {agent_stats['actor_param_mean']:+.6f}, std: {agent_stats['actor_param_std']:.6f}")
                        print(f"  Critic - mean: {agent_stats['critic_param_mean']:+.6f}, std: {agent_stats['critic_param_std']:.6f}")
                        print(f"{'='*70}\n")

            if t % 100 == 0:
                phase = "EXPLORATION" if t <= start_timesteps else "LEARNING"

            vehicle_state = info.get('vehicle_state', {})
            speed_kmh = vehicle_state.get('velocity', 0) * 3.6
            '''
            print(
                    f"[{phase}] Step {t:6d}/{self.max_timesteps:,} | "
                    f"Episode {self.episode_num:4d} | "
                    f"Ep Step {self.episode_timesteps:4d} | "
                    f"Reward={reward:+7.2f} | "
                    f"Speed={speed_kmh:5.1f} km/h | "
                    f"Buffer={len(self.agent.replay_buffer):7d}/{self.agent.replay_buffer.max_size}"
                )
            '''

            self.writer.add_scalar('progress/buffer_size', len(self.agent.replay_buffer), t)
            self.writer.add_scalar('progress/episode_steps', self.episode_timesteps, t)
            self.writer.add_scalar('progress/current_reward', reward, t)
            self.writer.add_scalar('progress/speed_kmh', speed_kmh, t)

            for _k, _v in (info.get('safety_terms') or {}).items():
                self.writer.add_scalar(f'safety/{_k}', _v, t)

            if 'progress_percentage' in info:
                self.writer.add_scalar('progress/route_progress_pct',
                                       info['progress_percentage'], t)
            if 'distance_to_goal' in info:
                self.writer.add_scalar('progress/distance_to_goal_m',
                                       info['distance_to_goal'], t)
            for _k, _tag in (('diag_vehicle_yaw_deg', 'heading/vehicle_yaw_deg'),
                             ('diag_lane_heading_deg', 'heading/lane_heading_deg'),
                             ('diag_route_heading_deg', 'heading/route_heading_deg')):
                if _k in info:
                    self.writer.add_scalar(_tag, info[_k], t)
            if all(k in info for k in ('diag_vehicle_yaw_deg', 'diag_route_heading_deg')):
                _e = (info['diag_vehicle_yaw_deg'] - info['diag_route_heading_deg'] + 180.0) % 360.0 - 180.0
                self.writer.add_scalar('heading/vehicle_vs_route_deg', _e, t)
                self.writer.add_scalar('heading/wrong_way_vs_route', 1.0 if abs(_e) > 90.0 else 0.0, t)
            if all(k in info for k in ('diag_lane_heading_deg', 'diag_route_heading_deg')):
                _l = (info['diag_lane_heading_deg'] - info['diag_route_heading_deg'] + 180.0) % 360.0 - 180.0
                self.writer.add_scalar('heading/lane_vs_route_deg', _l, t)

            self.writer.flush()
            if done or truncated:
                termination_type = "DONE (collision/goal)" if done else "TRUNCATED (max steps)"
                self.logger.info(
                    f"[EPISODE END] Episode {self.episode_num} {termination_type}: "
                    f"Final episode_reward={self.episode_reward:+.3f} | "
                    f"Episode lasted {self.episode_timesteps} steps | "
                    f"Collisions={self.episode_collision_count} | "
                    f"Lane invasions={self.episode_lane_invasion_count}"
                )

                component_sum = sum(self.episode_reward_components.values())
                if abs(component_sum - self.episode_reward) > 0.01:
                    self.logger.error(
                        f"[REWARD MISMATCH] Episode {self.episode_num}: "
                        f"episode_reward={self.episode_reward:+.3f} != "
                        f"component_sum={component_sum:+.3f} | "
                        f"diff={abs(component_sum - self.episode_reward):.3f}"
                    )

                self.training_rewards.append(self.episode_reward)
                self.writer.add_scalar('train/episode_reward', self.episode_reward, self.episode_num)
                self.writer.add_scalar('train/episode_length', self.episode_timesteps, self.episode_num)
                self.writer.add_scalar(
                    'train/collisions_per_episode',
                    self.episode_collision_count,
                    self.episode_num
                )
                self.writer.add_scalar(
                    'train/lane_invasions_per_episode',
                    self.episode_lane_invasion_count,
                    self.episode_num
                )
                _reason = info.get('termination_reason', 'running')
                for _r in ('collision', 'off_road', 'route_completed', 'carla_tick_timeout'):
                    self.writer.add_scalar(f'train/terminated_{_r}', float(_reason == _r), self.episode_num)
                self.writer.add_scalar('train/truncated', float(not done), self.episode_num)

                total_magnitude = sum(abs(v) for v in self.episode_reward_components.values())

                if total_magnitude > 0:
                    self.writer.add_scalar(
                        'rewards/efficiency_component',
                        self.episode_reward_components['efficiency'],
                        self.episode_num
                    )
                    self.writer.add_scalar(
                        'rewards/lane_keeping_component',
                        self.episode_reward_components['lane_keeping'],
                        self.episode_num
                    )
                    self.writer.add_scalar(
                        'rewards/comfort_component',
                        self.episode_reward_components['comfort'],
                        self.episode_num
                    )
                    self.writer.add_scalar(
                        'rewards/safety_component',
                        self.episode_reward_components['safety'],
                        self.episode_num
                    )
                    self.writer.add_scalar(
                        'rewards/progress_component',
                        self.episode_reward_components['progress'],
                        self.episode_num
                    )

                    for component, value in self.episode_reward_components.items():
                        percentage = (abs(value) / total_magnitude) * 100.0
                        self.writer.add_scalar(
                            f'rewards/{component}_percentage',
                            percentage,
                            self.episode_num
                        )

                    max_component = max(self.episode_reward_components.items(), key=lambda x: abs(x[1]))
                    max_percentage = (abs(max_component[1]) / total_magnitude) * 100.0

                    if max_percentage > 70.0:
                        print(
                            f"[REWARD BALANCE] '{max_component[0]}' dominates at {max_percentage:.1f}% "
                            f"(target <70%, literature <60%)"
                        )

                if self.episode_num % 10 == 0:
                    avg_reward = np.mean(self.training_rewards[-10:]) if len(self.training_rewards) >= 10 else np.mean(self.training_rewards)
                    print(
                        f"[TRAIN] Episode {self.episode_num:4d} | "
                        f"Timestep {t:7d} | "
                        f"Reward {self.episode_reward:8.2f} | "
                        f"Avg Reward (10ep) {avg_reward:8.2f} | "
                        f"Collisions {self.episode_collision_count:2d} | "
                        f"Lane Invasions {self.episode_lane_invasion_count:2d}"
                    )

                obs_dict, _ = self.env.reset()
                self.episode_num += 1
                self.episode_reward = 0
                self.episode_timesteps = 0
                self.episode_collision_count = 0
                self.episode_lane_invasion_count = 0

                self.episode_reward_components = {
                    'efficiency': 0.0,
                    'lane_keeping': 0.0,
                    'comfort': 0.0,
                    'safety': 0.0,
                    'progress': 0.0
                }

            if t % self.eval_freq == 0:
                print(f"\n[EVAL] Evaluation at timestep {t:,}...")

                self.in_eval_phase = True

                eval_metrics, obs_dict = self.evaluate()

                self.in_eval_phase = False

                self.writer.add_scalar('eval/mean_reward', eval_metrics['mean_reward'], t)
                self.writer.add_scalar('eval/success_rate', eval_metrics['success_rate'], t)
                self.writer.add_scalar('eval/avg_collisions', eval_metrics['avg_collisions'], t)
                self.writer.add_scalar('eval/avg_lane_invasions', eval_metrics['avg_lane_invasions'], t)
                self.writer.add_scalar('eval/avg_episode_length', eval_metrics['avg_episode_length'], t)

                self.eval_rewards.append(eval_metrics['mean_reward'])
                self.eval_success_rates.append(eval_metrics['success_rate'])
                self.eval_collisions.append(eval_metrics['avg_collisions'])
                self.eval_lane_invasions.append(eval_metrics['avg_lane_invasions'])

                print(
                    f"[EVAL] Mean Reward: {eval_metrics['mean_reward']:.2f} | "
                    f"Success Rate: {eval_metrics['success_rate']*100:.1f}% | "
                    f"Avg Collisions: {eval_metrics['avg_collisions']:.2f} | "
                    f"Avg Lane Invasions: {eval_metrics['avg_lane_invasions']:.2f} | "
                    f"Avg Length: {eval_metrics['avg_episode_length']:.0f}"
                )
                print()

                done = False
                self.episode_reward = 0
                self.episode_timesteps = 0
                self.episode_collision_count = 0
                self.episode_lane_invasion_count = 0
                self.episode_steps_since_collision = 0
                self.episode_reward_components = {
                    'efficiency': 0.0,
                    'lane_keeping': 0.0,
                    'comfort': 0.0,
                    'safety': 0.0,
                    'progress': 0.0
                }

            if t % self.checkpoint_freq == 0:
                _run_ckpt_dir = self.checkpoint_dir / self.log_name
                _run_ckpt_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = _run_ckpt_dir / f"step_{t}.pth"
                self.agent.save_checkpoint(str(checkpoint_path))
                print(f"[CHECKPOINT] Saved to {checkpoint_path}")

        print(f"\n[TRAINING] Training complete!")
        self.close()

    def evaluate(self):
        print(f"\n[EVAL] Starting evaluation phase (deterministic policy, no training)...")

        if hasattr(self.env, 'vehicle') and self.env.vehicle is not None:
            vehicle_id_before_eval = self.env.vehicle.id
            vehicle_alive_before = self.env.vehicle.is_alive
            print(f"[EVAL] Vehicle state before evaluation:")
            print(f"[EVAL]   ID: {vehicle_id_before_eval}")
            print(f"[EVAL]   is_alive: {vehicle_alive_before}")
        else:
            print(f"[EVAL] Warning: Vehicle reference not available before evaluation")
            vehicle_id_before_eval = None

        episode_num_before_eval = self.episode_num

        eval_rewards = []
        eval_successes = []
        eval_collisions = []
        eval_lane_invasions = []
        eval_lengths = []

        max_eval_steps = self.agent_config.get("training", {}).get("max_episode_steps", 1000)

        for episode in range(self.num_eval_episodes):
            obs_dict, reset_info = self.env.reset(options={"random_start": False})

            episode_reward = 0
            episode_length = 0
            done = False

            while not done and episode_length < max_eval_steps:

                action = self.agent.select_action(
                    obs_dict,
                    deterministic=True
                )

                next_obs_dict, reward, done, truncated, info = self.env.step(action)

                if self.debug:
                    self._visualize_debug(next_obs_dict, action, reward, info, episode_length)

                episode_reward += reward
                episode_length += 1
                obs_dict = next_obs_dict

                if truncated:
                    done = True

            if episode_length >= max_eval_steps:
                print(f"[EVAL] Warning: Episode {episode+1} reached max eval steps ({max_eval_steps})")

            eval_rewards.append(episode_reward)
            eval_successes.append(info.get('success', 0))
            eval_collisions.append(info.get('collision_count', 0))
            eval_lane_invasions.append(info.get('lane_invasion_count', 0))
            eval_lengths.append(episode_length)

        self.episode_num = episode_num_before_eval

        print(f"[EVAL] Resetting environment for fresh training episode...")
        fresh_obs_dict, _ = self.env.reset()

        if hasattr(self.env, 'vehicle') and self.env.vehicle is not None:
            vehicle_id_after_eval = self.env.vehicle.id
            vehicle_alive_after = self.env.vehicle.is_alive
            print(f"[EVAL] Vehicle state after evaluation:")
            print(f"[EVAL]   ID: {vehicle_id_after_eval}")
            print(f"[EVAL]   is_alive: {vehicle_alive_after}")

            if not vehicle_alive_after:
                print(f"[EVAL] ERROR: Vehicle destroyed during evaluation!")

            if vehicle_id_before_eval is not None and vehicle_id_before_eval != vehicle_id_after_eval:
                print(f"[EVAL] Note: Vehicle ID changed ({vehicle_id_before_eval} -> {vehicle_id_after_eval})")
                print(f"[EVAL]       This is EXPECTED after env.reset() (respawns actors)")

        eval_metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'success_rate': np.mean(eval_successes),
            'avg_collisions': np.mean(eval_collisions),
            'avg_lane_invasions': np.mean(eval_lane_invasions),
            'avg_episode_length': np.mean(eval_lengths)
        }

        print(f"[EVAL] Evaluation complete:")
        print(f"[EVAL]   Mean Reward: {eval_metrics['mean_reward']:.2f}")
        print(f"[EVAL]   Success Rate: {eval_metrics['success_rate']*100:.1f}%")
        print(f"[EVAL]   Avg Collisions: {eval_metrics['avg_collisions']:.2f}")
        print(f"[EVAL]   Avg Lane Invasions: {eval_metrics['avg_lane_invasions']:.2f}")
        print(f"[EVAL]   Avg Episode Length: {eval_metrics['avg_episode_length']:.0f}")
        print(f"[EVAL] Returning to training...\n")

        return eval_metrics, fresh_obs_dict

    def save_final_results(self):
        results = {
            'scenario': self.scenario,
            'seed': self.seed,
            'total_timesteps': self.max_timesteps,
            'total_episodes': self.episode_num,
            'training_rewards': self.training_rewards,
            'eval_rewards': self.eval_rewards,
            'eval_success_rates': [float(x) for x in self.eval_success_rates],
            'eval_collisions': [float(x) for x in self.eval_collisions],
            'eval_lane_invasions': [float(x) for x in self.eval_lane_invasions],
            'final_eval_mean_reward': float(np.mean(self.eval_rewards[-5:])) if len(self.eval_rewards) > 0 else 0,
            'final_eval_success_rate': float(np.mean(self.eval_success_rates[-5:])) if len(self.eval_success_rates) > 0 else 0
        }

        results_path = self.log_path / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[RESULTS] Saved to {results_path}")

    def close(self):
        try:
            if self.debug:
                cv2.destroyAllWindows()
                print(f"[DEBUG] OpenCV windows closed")
        except Exception as e:
            print(f"[WARNING] OpenCV cleanup failed: {e}")

        try:
            self.save_final_results()
        except Exception as e:
            print(f"[WARNING] Result saving failed: {e}")

        try:
            self.env.close()
        except Exception as e:
            print(f"[WARNING] Environment cleanup failed: {e}")

        try:
            self.writer.close()
        except Exception as e:
            print(f"[WARNING] TensorBoard cleanup failed: {e}")

        print(f"[CLEANUP] Environment closed, logging finalized")


def main():
    parser = argparse.ArgumentParser(
        description="Train TD3 agent for autonomous vehicle navigation in CARLA"
    )
    parser.add_argument(
        '--scenario',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Traffic scenario (0=20, 1=50, 2=100 NPCs)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--max-timesteps',
        type=int,
        default=int(1e6),
        help='Maximum training timesteps'
    )
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5000,
        help='Evaluation frequency (timesteps)'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=10000,
        help='Checkpoint saving frequency (timesteps)'
    )
    parser.add_argument(
        '--num-eval-episodes',
        type=int,
        default=10,
        help='Number of episodes per evaluation'
    )
    parser.add_argument(
        '--carla-config',
        type=str,
        default='config/carla_config.yaml',
        help='Path to CARLA config file'
    )
    parser.add_argument(
        '--agent-config',
        type=str,
        default='config/td3_config.yaml',
        help='Path to TD3 agent config file'
    )
    parser.add_argument(
        '--training-config',
        type=str,
        default='config/training_config.yaml',
        help='Path to the training config (reward, scenarios, random start). Added 2026-10-14 so a new '
             'configuration can be trained WITHOUT editing the frozen config/training_config.yaml; the default '
             'is the path the pipeline always used.'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='data/logs',
        help='Directory for TensorBoard logs'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='data/checkpoints',
        help='Directory for checkpoints'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: OpenCV visualization + CNN diagnostics (gradient flow, weight updates, feature stats). Recommended for short runs to verify learning. TensorBoard: cnn_diagnostics/* metrics. Console: detailed output every 1000 steps.'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Console log level when --debug is NOT set (default: WARNING). '
             'The per-step [ACTION_EXECUTION]/[ACTION_RESULT]/"DEBUG Step N" records '
             'are emitted at INFO and cost ~87%% of wall-clock time (measured: 575 ms '
             'per step total against 77 ms of actual work). Use INFO to diagnose, '
             'WARNING to train. TensorBoard scalars are unaffected by this setting.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda', 'auto'],
        help='Device for TD3 agent (cpu/cuda/auto). Use "cpu" for systems with limited GPU memory (CARLA uses GPU), "cuda" for dedicated training GPUs, "auto" to automatically detect.'
    )
    parser.add_argument(
        '--carla-host', type=str, default='localhost',
        help='CARLA server host (default: localhost).'
    )
    parser.add_argument(
        '--carla-port', type=int, default=2000,
        help='CARLA server RPC port (default: 2000). GIVE EACH CONCURRENT RUN ITS OWN PORT '
             'and start a CARLA server per run with --carla-rpc-port. Two runs on the same '
             'port connect to the SAME simulator and drive the same vehicle, silently - '
             'connecting to a live server is what success looks like, so nothing errors.'
    )
    parser.add_argument(
        '--tm-port', type=int, default=8000,
        help='Traffic Manager port (default: 8000). Must also be unique per concurrent run. '
             'Kept unified across EXPLORATION/LEARNING/EVAL *within* a run, because CARLA '
             'shares one world per client.'
    )
    parser.add_argument(
        '--env-isolation', action='store_true',
        help='M1 (2026-10-16): run the environment in its own process with per-call time-outs, so that a CARLA '
             'traffic-manager hang (#9172) costs one truncated episode instead of the run. Default off.'
    )
    parser.add_argument('--env-step-timeout', type=float, default=60.0,
                        help='M1: seconds a step may take before the environment process is restarted (default 60; '
                             'a normal step takes ~50 ms).')
    parser.add_argument('--env-reset-timeout', type=float, default=300.0,
                        help='M1: seconds a reset may take (default 300; a reset reloads traffic).')

    args = parser.parse_args()

    if args.carla_port == args.tm_port:
        parser.error(f"--carla-port and --tm-port are both {args.carla_port}; they must differ")

    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        logging.getLogger().setLevel(logging.DEBUG)

        print("\n" + "="*70)
        print("[DEBUG LOGGING ENABLED]")
        print("="*70)
        print("[DEBUG] All pipeline logs will be shown:")
        print("[DEBUG]   Image preprocessing & frame stacking")
        print("[DEBUG]   CNN layer-by-layer forward pass")
        print("[DEBUG]   TD3 feature extraction & training")
        print("[DEBUG]   Reward component breakdown")
        print("="*70 + "\n")
    else:
        level = getattr(logging, args.log_level.upper(), logging.WARNING)
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            force=True
        )
        logging.getLogger().setLevel(level)

    trainer = TD3TrainingPipeline(
        scenario=args.scenario,
        seed=args.seed,
        max_timesteps=args.max_timesteps,
        eval_freq=args.eval_freq,
        checkpoint_freq=args.checkpoint_freq,
        num_eval_episodes=args.num_eval_episodes,
        carla_config_path=args.carla_config,
        agent_config_path=args.agent_config,
        training_config_path=args.training_config,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        debug=args.debug,
        device=args.device,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
        tm_port=args.tm_port,
        env_isolation=args.env_isolation,
        env_step_timeout=args.env_step_timeout,
        env_reset_timeout=args.env_reset_timeout,
    )

    trainer.train()


if __name__ == "__main__":
    main()

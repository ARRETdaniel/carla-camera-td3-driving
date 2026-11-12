# Camera-based TD3 driving in CARLA: a training and evaluation system

Code and configurations of a system that trains deep reinforcement-learning drivers in the
[CARLA](https://carla.org) 0.9.16 simulator and evaluates them, together with a classical controller, through one
evaluator. The repository accompanies a paper under double-blind review.

The agents drive urban routes with traffic. Their observation is four stacked 84×84 grey front-camera frames and a
route vector (speed, signed lateral offset, heading error, the next 25 route waypoints in the ego frame, yaw rate,
lateral velocity). Their action is steering and a combined throttle/brake command. Two agents share one
implementation:

- **TD3**;
- **AHE**, the TD3 authors' ablation baseline: the same code with one critic instead of two, a policy update at every
  step, and no target-action noise.

The classical controller is a PID speed controller with Pure Pursuit steering. It reads the vehicle state and the
route, and it does not perceive traffic.

## Contents

| Path | What it holds |
|---|---|
| `src/environment/` | The Gymnasium environment: the CARLA client, the sensors, the route and its waypoints, the reward, the episode rules and the route pool. `isolated_env.py` runs the environment in a child process with a time-out on every call. |
| `src/agents/`, `src/networks/`, `src/utils/` | The TD3/AHE agent, the CNN encoder, the actor and the critics, the replay buffer. |
| `src/baselines/` | The classical controller. |
| `scripts/train_td3.py` | Training. |
| `scripts/measure_route_progress_v5.py` | Evaluation of a checkpoint or of the classical controller, with per-step telemetry. |
| `config/` | Agent, environment, training and controller configurations (see the tables below). |
| `config/waypoints.txt`, `config/routes/` | The training route, the ten further routes of the multi-route pool, three held-out Town01 routes and three Town02 routes, with checksums. |
| `Dockerfile` | The training and evaluation image: the CARLA 0.9.16 client, Python 3.10, PyTorch 2.4.1, Gymnasium 0.29.1. |

## Configurations

Every arm uses `config/training_config_competence.yaml`.

| Arm | Agent configuration | Environment configuration | Training steps |
|---|---|---|---|
| Single-route TD3 | `config/td3_config.yaml` | `config/carla_config_competence_s2c.yaml` | 300,000 |
| Single-route AHE | `config/td3_config_ahe.yaml` | `config/carla_config_competence_s2c.yaml` | 300,000 |
| Multi-route TD3 | `config/td3_config.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| Multi-route AHE | `config/td3_config_ahe.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| Multi-route TD3 with the actor regulariser | `config/td3_config_caps_preact.yaml` | `config/carla_config_v2.yaml` | 600,000 |
| Multi-route AHE with the actor regulariser | `config/td3_config_ahe_caps_preact.yaml` | `config/carla_config_v2.yaml` | 600,000 |

- **AHE against TD3:** `networks.critic.n_critics` 1 instead of 2, `algorithm.policy_freq` 1 instead of 2,
  `algorithm.noise_clip` 0.0 instead of 0.5. Nothing else differs.
- **The actor regulariser:** five keys added to the agent configuration (`algorithm.caps.enabled`,
  `lambda_temporal` 1.0, `lambda_spatial` 1.0, `sigma_vector` 0.02, and `algorithm.pre_activation_weight` 0.05).
- **Multi-route against single-route:** one route drawn per episode from a pool of eleven (`route.pool`), an episode
  limit of 6 steps per metre of route (`episode.max_time_steps_per_m`), and the target speed ahead added to the
  route vector (`observation.extra_vector`).
- **The classical controller:** `config/baseline_config_c08.yaml`.

Evaluation configurations:

| Files | Used for |
|---|---|
| `config/eval_v2/carla_v2_train.yaml` | multi-route agents and the classical controller on the training route |
| `config/eval_v2/carla_v2_P00.yaml`, `_P01`, `_P04` | three routes of the training pool |
| `config/eval_v2/carla_v2_H1.yaml`, `_H2`, `_H3` | the three held-out Town01 routes |
| `config/eval_v2/carla_v2_T02vA.yaml`, `_T02vB`, `_T02vC` | the three Town02 routes |
| `config/eval_v1_town02/` | single-route agents on the three Town02 routes |

## Running

The commands below are those of our launch scripts, with the site-specific parts removed. Run them from the
repository root. One CARLA server serves one training or evaluation process.

**1. Build the image.**

```bash
docker build -t td3-carla:py310 .
```

**2. Start a CARLA server.**

```bash
docker run -d --name carla-run1 --runtime=nvidia --net=host \
  --env=NVIDIA_VISIBLE_DEVICES=all --env=NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 bash CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000
```

**3. Train** (here multi-route TD3; `--scenario 0` is 20 NPC vehicles).

```bash
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace \
  -v "$(pwd)":/workspace -w /workspace td3-carla:py310 \
  python3 scripts/train_td3.py \
    --scenario 0 --max-timesteps 600000 \
    --eval-freq 999999 --checkpoint-freq 25000 \
    --seed 42 --log-level WARNING --device cuda \
    --carla-port 2000 --tm-port 8000 \
    --agent-config config/td3_config.yaml \
    --carla-config config/carla_config_v2.yaml \
    --training-config config/training_config_competence.yaml \
    --env-isolation
```

**4. Evaluate a checkpoint** (10 deterministic episodes; `CARLA_SCENARIO_INDEX` 0 is 20 NPC vehicles, 1 is 50).

```bash
docker run --rm --network host --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace -e CARLA_SCENARIO_INDEX=0 \
  -v "$(pwd)":/workspace -v /path/to/checkpoints:/ck:ro -v /path/to/output:/out -w /workspace td3-carla:py310 \
  python3 scripts/measure_route_progress_v5.py \
    --checkpoint /ck/step_600000.pth --episodes 10 --max-steps 3000 --device cpu \
    --carla-port 2000 --tm-port 8000 \
    --carla-config config/eval_v2/carla_v2_H1.yaml \
    --training-config config/training_config_competence.yaml \
    --csv /out/h1_telemetry.csv
```

The environment applies the episode limit of its configuration. `--max-steps` is the evaluator's outer cap: our
multi-route cells used 3000, and the single-route cells the default of 1000.

**5. Evaluate the classical controller** through the same evaluator: replace `--checkpoint ...` with

```bash
    --controller pid_pp --baseline-config config/baseline_config_c08.yaml --seed 42
```

Our protocol starts a fresh CARLA server for every evaluation cell of 10 episodes.

## Notes for reproduction

- **Runs are not reproducible from a seed.** In this system two training runs with identical code and seed diverge
  early. Treat every run as an independent draw and train several.
- **The route of the single-route regime.** `config/carla_config_competence_s2c.yaml` sets
  `route.use_dynamic_generation: true`. The image built from this `Dockerfile` does not provide CARLA's `agents`
  package, so the environment logs a warning and reads the route from `config/waypoints.txt`. The single-route runs
  ran that way. The multi-route configurations set the key to `false`.
- **Which revision trained which runs.** These files are the revision that trained the runs with the actor
  regulariser. The multi-route runs without it were trained on the preceding revision, which lacks one agent switch
  that none of these configurations sets. The single-route runs were trained on an earlier revision, before the
  multi-route options and the evaluator in `scripts/` were added; the four configuration files they used are
  unchanged since.
- **What was edited for this release.** Comments were removed, each module keeps the first line of its docstring, and
  identifying fields were removed from the configuration files. Every Python file was checked to have the same syntax
  tree as its source, docstrings aside, and every YAML file to parse to the same content, those fields aside. The
  routes, their checksums and the `Dockerfile` are unchanged.

## Not included

Trained checkpoints, per-step telemetry, analysis scripts and the launch scripts of our machines are not part of
this release.

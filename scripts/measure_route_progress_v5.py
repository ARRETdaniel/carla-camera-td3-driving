#!/usr/bin/env python3
"""V5 (2026-10-17, master, tracker A12): V4 plus three getter-only traffic-light telemetry columns (tl_at,"""
import argparse
import csv
import logging
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.agents.td3_agent import TD3Agent
from src.environment.carla_env import CARLANavigationEnv
from src.networks.cnn_extractor import NatureCNN


def _ablate(obs, mode):
    if mode == "none":
        return obs
    out = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in obs.items()}
    if mode == "image":
        out["image"] = np.zeros_like(out["image"])
    elif mode == "waypoints":
        out["vector"] = out["vector"].copy(); out["vector"][3:53] = 0.0
    elif mode == "extras":
        out["vector"] = out["vector"].copy(); out["vector"][53:] = 0.0
    elif mode == "kinematics":
        out["vector"] = out["vector"].copy(); out["vector"][:3] = 0.0
    return out


def _hud(obs, info, action, reward, ep, step):
    img = np.asarray(obs["image"])
    frame = img[-1] if img.ndim == 3 else img
    lo, hi = float(frame.min()), float(frame.max())
    frame = (frame - lo) / (hi - lo + 1e-8)
    view = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    view = cv2.resize(view, (420, 420), interpolation=cv2.INTER_NEAREST)

    panel = np.zeros((420, 380, 3), dtype=np.uint8)
    vs = info.get("vehicle_state", {}) or {}
    d = info.get("distance_to_goal", float("nan"))
    lines = [
        (f"episode {ep}   step {step}", (255, 255, 255)),
        ("", None),
        (f"speed        {vs.get('velocity', 0.0)*3.6:8.1f} km/h", (0, 255, 255)),
        (f"DIST TO GOAL {d:8.1f} m", (0, 255, 0) if d < 10 else (255, 255, 255)),
        (f"route done   {info.get('progress_percentage', float('nan')):8.1f} %", (0, 255, 255)),
        (f"lateral dev  {info.get('diag_lateral_dev', float('nan')):8.2f} m",
         (0, 165, 255) if info.get("diag_lateral_dev", 0) > 2.0 else (255, 255, 255)),
        ("", None),
        (f"steer        {action[0]:+8.3f}", (255, 255, 255)),
        (f"throttle/brk {action[1]:+8.3f}", (255, 255, 255)),
        (f"reward       {reward:+8.2f}", (0, 255, 0) if reward > 0 else (0, 0, 255)),
        ("", None),
        (f"goal thresh  3.0 m", (150, 150, 150)),
        (f"is_route_finished  {info.get('diag_is_route_finished', '?')}",
         (0, 255, 0) if info.get("diag_is_route_finished") else (150, 150, 150)),
        (f"check_goal_reached {info.get('diag_check_goal_reached', '?')}",
         (0, 255, 0) if info.get("diag_check_goal_reached") else (150, 150, 150)),
    ]
    y = 28
    for text, colour in lines:
        if text:
            cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        colour or (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    cv2.putText(panel, "agent camera input (84x84, upscaled)", (12, 408),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    return np.hstack([view, panel])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="", help="agent checkpoint (required for --controller agent)")
    ap.add_argument("--controller", choices=["agent", "pid_pp"], default="agent")
    ap.add_argument("--baseline-config", default="config/baseline_config.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="np.random.seed before the env is built (evaluate_baseline.py:84); default unseeded")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--carla-host", default="localhost")
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--carla-config", default="config/carla_config.yaml")
    ap.add_argument("--agent-config", default="config/td3_config.yaml")
    ap.add_argument("--training-config", default="config/training_config.yaml")
    ap.add_argument("--render", action="store_true",
                    help="show the agent's camera + a HUD, and follow the car with CARLA's "
                         "spectator so the drive can be watched directly")
    ap.add_argument("--video", default="", help="also write the rendered view to this .mp4")
    ap.add_argument("--csv", default="", help="write per-step telemetry here for auditing")
    ap.add_argument("--ablate", default="none",
                    choices=["none", "image", "waypoints", "kinematics", "extras"],
                    help="zero one input modality to see what the policy actually relies on. "
                         "vector layout is [velocity, lateral_dev, heading_err, 25x(x,y)]")
    args = ap.parse_args()
    print(f"V5: input ablation = {args.ablate}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.controller == "agent" and not args.checkpoint:
        ap.error("--checkpoint is required for --controller agent")
    if args.seed is not None:
        np.random.seed(args.seed)

    env = CARLANavigationEnv(args.carla_config, args.agent_config,
                             args.training_config, host=args.carla_host,
                             port=args.carla_port, tm_port=args.tm_port)

    agent = controller = None
    if args.controller == "agent":
        actor_cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512).to(args.device)
        critic_cnn = NatureCNN(input_channels=4, num_frames=4, feature_dim=512).to(args.device)
        vector_dim = int(env.observation_space["vector"].shape[0])
        agent = TD3Agent(state_dim=512 + vector_dim, action_dim=2, max_action=1.0,
                         actor_cnn=actor_cnn, critic_cnn=critic_cnn,
                         use_dict_buffer=True, config_path=args.agent_config,
                         device=args.device, vector_dim=vector_dim)
        agent.load_checkpoint(args.checkpoint)
        print(f"\nloaded {args.checkpoint}\n")
    else:
        import yaml
        from src.baselines.baseline_controller import BaselineController
        bc = yaml.safe_load(open(args.baseline_config))["baseline_controller"]
        controller = BaselineController(
            pid_kp=bc["pid"]["kp"], pid_ki=bc["pid"]["ki"], pid_kd=bc["pid"]["kd"],
            kp_lookahead=bc["pure_pursuit"]["kp_lookahead"],
            min_lookahead=bc["pure_pursuit"]["min_lookahead"],
            wheelbase=bc["pure_pursuit"]["wheelbase"],
            target_speed=bc["general"]["target_speed_kmh"])
        ccfg = yaml.safe_load(open(args.carla_config))
        wp_file = ccfg.get("route", {}).get("waypoints_file", "config/waypoints.txt")
        waypoints = []
        for line in open(wp_file):
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split(",")
                if len(parts) >= 3:
                    waypoints.append(tuple(float(v.strip()) for v in parts[:3]))
        dt = float(ccfg.get("world", {}).get("fixed_delta_seconds", 0.05))
        print(f"\ncontroller pid_pp: {controller}\n{len(waypoints)} waypoints from {wp_file}, dt={dt}\n")

    writer = None
    csv_f = csv_w = None
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["episode", "step", "speed_kmh", "distance_to_goal_m",
                        "route_progress_pct", "lateral_dev_m", "steer", "throttle_brake",
                        "reward", "is_route_finished", "check_goal_reached", "termination",
                        "dist_obstacle_m",
                        "q1", "q2", "terminated",
                        "x", "y", "yaw_deg", "speed_mps", "ax_long", "ay_lat", "yaw_rate_dps",
                        "tl_at", "tl_state", "tl_id"])

    rows = []
    for ep in range(args.episodes):
        obs, info = env.reset(options={"random_start": False})
        if controller is not None:
            controller.reset()
        prog, dist = [], []
        veh_route, lane_route = [], []
        near = []
        ret, steps = 0.0, 0
        reason = "running"
        for _ in range(args.max_steps):
            if agent is not None:
                obs = _ablate(obs, args.ablate)
                action = agent.select_action(obs, deterministic=True)
                with torch.no_grad():
                    _o = {"image": torch.FloatTensor(obs["image"]).unsqueeze(0).to(args.device),
                          "vector": torch.FloatTensor(obs["vector"]).unsqueeze(0).to(args.device)}
                    _s = agent.extract_features(_o, enable_grad=False, use_actor_cnn=False)
                    _a = torch.FloatTensor(np.asarray(action, dtype=np.float32)).unsqueeze(0).to(args.device)
                    _q1, _q2 = agent.critic(_s, _a)
                    q1_t, q2_t = float(_q1.item()), float(_q2.item())
            else:
                ctl = controller.compute_control(vehicle=env.vehicle, waypoints=waypoints, dt=dt)
                action = np.array([ctl.steer, ctl.throttle - ctl.brake])
                q1_t = q2_t = float("nan")
            obs, reward, terminated, truncated, info = env.step(action)
            _tf = env.vehicle.get_transform(); _v = env.vehicle.get_velocity()
            _acc = env.vehicle.get_acceleration(); _w = env.vehicle.get_angular_velocity()
            _psi = np.radians(_tf.rotation.yaw)
            kin = (_tf.location.x, _tf.location.y, _tf.rotation.yaw,
                   float(np.sqrt(_v.x**2 + _v.y**2 + _v.z**2)),
                   _acc.x * np.cos(_psi) + _acc.y * np.sin(_psi),
                   -_acc.x * np.sin(_psi) + _acc.y * np.cos(_psi),
                   _w.z)
            _tl_at = bool(env.vehicle.is_at_traffic_light())
            _tl_state = str(env.vehicle.get_traffic_light_state())
            _tl = env.vehicle.get_traffic_light()
            _tl_id = int(_tl.id) if _tl is not None else -1
            ret += reward
            steps += 1
            if "progress_percentage" in info:
                prog.append(float(info["progress_percentage"]))
            if "distance_to_goal" in info:
                dist.append(float(info["distance_to_goal"]))
            if all(k in info for k in ("diag_vehicle_yaw_deg", "diag_route_heading_deg")):
                wrap = lambda a: (a + 180.0) % 360.0 - 180.0
                veh_route.append(abs(wrap(info["diag_vehicle_yaw_deg"]
                                          - info["diag_route_heading_deg"])))
                lane_route.append(abs(wrap(info["diag_lane_heading_deg"]
                                           - info["diag_route_heading_deg"])))
            if info.get("distance_to_goal", 1e9) < 15.0 and "diag_goal_euclid_m" in info:
                near.append((float(info["distance_to_goal"]), float(info["diag_goal_euclid_m"]),
                             bool(info["diag_is_route_finished"]),
                             bool(info["diag_check_goal_reached"]),
                             int(info["diag_dense_idx"]), int(info["diag_dense_len"]),
                             float(info["diag_lateral_dev"])))
            if csv_w is not None:
                vs = info.get("vehicle_state", {}) or {}
                csv_w.writerow([ep, steps, f"{vs.get('velocity', 0.0)*3.6:.2f}",
                                f"{info.get('distance_to_goal', float('nan')):.2f}",
                                f"{info.get('progress_percentage', float('nan')):.2f}",
                                f"{info.get('diag_lateral_dev', float('nan')):.2f}",
                                f"{action[0]:+.3f}", f"{action[1]:+.3f}", f"{reward:+.3f}",
                                info.get("diag_is_route_finished", ""),
                                info.get("diag_check_goal_reached", ""),
                                info.get("termination_reason", ""),
                                f"{info.get('distance_to_nearest_obstacle', float('inf')):.2f}",
                                f"{q1_t:.4f}", f"{q2_t:.4f}", int(bool(terminated)),
                                f"{kin[0]:.4f}", f"{kin[1]:.4f}", f"{kin[2]:.3f}", f"{kin[3]:.5f}",
                                f"{kin[4]:.5f}", f"{kin[5]:.5f}", f"{kin[6]:.4f}",
                                int(_tl_at), _tl_state, _tl_id])

            if args.render or args.video:
                frame = _hud(obs, info, action, reward, ep, steps)
                if args.video and writer is None:
                    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
                    writer = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             20.0, (frame.shape[1], frame.shape[0]))
                if writer is not None:
                    writer.write(frame)
            if args.render:
                cv2.imshow("TD3 agent - camera input + telemetry", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                try:
                    tf = env.vehicle.get_transform()
                    import carla as _c
                    env.world.get_spectator().set_transform(_c.Transform(
                        tf.location + _c.Location(x=-6 * np.cos(np.radians(tf.rotation.yaw)),
                                                  y=-6 * np.sin(np.radians(tf.rotation.yaw)),
                                                  z=3.5),
                        _c.Rotation(pitch=-15, yaw=tf.rotation.yaw)))
                except Exception:
                    pass

            if terminated or truncated:
                reason = info.get("termination_reason", "truncated")
                print(f"     >>> EPISODE {ep} ENDED: termination_reason = {reason!r} "
                      f"at step {steps}, distance_to_goal "
                      f"{info.get('distance_to_goal', float('nan')):.2f} m")
                break
        rows.append({
            "ep": ep, "steps": steps, "return": ret, "reason": reason,
            "max_progress": max(prog) if prog else float("nan"),
            "end_progress": prog[-1] if prog else float("nan"),
            "min_dist": min(dist) if dist else float("nan"),
            "end_dist": dist[-1] if dist else float("nan"),
            "veh_vs_route": float(np.median(veh_route)) if veh_route else float("nan"),
            "veh_wrong_pct": float(np.mean(np.array(veh_route) > 90)) if veh_route else float("nan"),
            "lane_vs_route": float(np.median(lane_route)) if lane_route else float("nan"),
            "lane_flip_pct": float(np.mean(np.array(lane_route) > 90)) if lane_route else float("nan"),
        })
        if near:
            near.sort(key=lambda x: x[0])
            arc, euc, fin, chk, di, dl, lat = near[0]
            print(f"     closest step: arc {arc:6.2f} m | euclid {euc:6.2f} m | "
                  f"is_route_finished={fin} check_goal_reached={chk} | "
                  f"dense_idx {di}/{dl} (need >= {dl-300}) | lateral {lat:.2f} m")
        r = rows[-1]
        print(f"  ep {ep}: {r['steps']:>4} steps  return {r['return']:+9.1f}  "
              f"{r['reason']:<16} max route progress {r['max_progress']:6.1f}%  "
              f"closest to goal {r['min_dist']:8.1f} m")

    env.close()
    if writer is not None:
        writer.release()
        print(f"\nvideo written to {args.video}")
    if csv_f is not None:
        csv_f.close()
        print(f"per-step telemetry written to {args.csv}")
    if args.render:
        cv2.destroyAllWindows()

    mp = np.array([r["max_progress"] for r in rows], float)
    md = np.array([r["min_dist"] for r in rows], float)
    print("\n" + "=" * 84)
    print(f"ROUTE PROGRESS over {len(rows)} deterministic episodes")
    print("=" * 84)
    print(f"  max route progress : median {np.nanmedian(mp):.1f}%   best {np.nanmax(mp):.1f}%")
    print(f"  closest to goal    : median {np.nanmedian(md):.1f} m   best {np.nanmin(md):.1f} m")
    print(f"  goal threshold is 3.0 m (carla_env.py:803, waypoint_manager.py:456)")
    print(f"  completions        : {sum(1 for r in rows if r['reason']=='route_completed')}/{len(rows)}")

    vr = np.array([r["veh_vs_route"] for r in rows], float)
    vw = np.array([r["veh_wrong_pct"] for r in rows], float)
    lr = np.array([r["lane_vs_route"] for r in rows], float)
    lf = np.array([r["lane_flip_pct"] for r in rows], float)
    if np.isfinite(vr).any():
        print("\nHEADING DIAGNOSTIC - which reading of the wrong-way events is right?")
        print(f"  |vehicle - ROUTE| : median {np.nanmedian(vr):6.1f} deg   "
              f"steps beyond 90 deg: {100*np.nanmean(vw):5.1f}%")
        print(f"  |lane    - ROUTE| : median {np.nanmedian(lr):6.1f} deg   "
              f"steps beyond 90 deg: {100*np.nanmean(lf):5.1f}%")
        print("  vehicle-vs-route high  -> the car really is turned around")
        print("  lane-vs-route high     -> it is in the oncoming lane and the LANE-derived")
        print("                            heading flipped; NOT a U-turn (reverse=False)")

    best = np.nanmax(mp)
    print("\nVERDICT")
    if best > 90:
        print(f"  Reaches {best:.0f}% of the route -> the goal region IS reachable.")
        print("  Hypothesis (a): the INCENTIVE stops it finishing. Fix the reward.")
    elif best < 50:
        print(f"  Only reaches {best:.0f}% of the route -> it does not navigate there.")
        print("  Hypothesis (b): route-following is the blocker. A reward tweak alone")
        print("  would NOT produce completions - do not apply one on this evidence.")
    else:
        print(f"  Reaches {best:.0f}% - between the two hypotheses. BOTH the incentive and")
        print("  route-following are implicated; report both rather than picking one.")


if __name__ == "__main__":
    main()

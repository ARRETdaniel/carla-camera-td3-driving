"""Reward calculation for autonomous vehicle navigation."""

import numpy as np
from typing import Dict, Optional
import logging


class RewardCalculator:

    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)

        self.weights = config.get("weights", {
            "efficiency": 1.0,
            "lane_keeping": 5.0,
            "comfort": 0.5,
            "safety": 1.0,
            "progress": 1.0,
        })

        self.logger.info("=" * 80)
        self.logger.info("REWARD WEIGHTS VERIFICATION (addressing WARNING-002)")
        self.logger.info("=" * 80)
        for component, weight in self.weights.items():
            self.logger.info(f"  {component:15s}: {weight:6.1f}")
        self.logger.info("=" * 80)

        self.target_speed = config.get("efficiency", {}).get("target_speed", 8.33)
        self.speed_tolerance = config.get("efficiency", {}).get("speed_tolerance", 1.39)
        self.overspeed_penalty_scale = config.get("efficiency", {}).get(
            "overspeed_penalty_scale", 2.0
        )
        self.speed_tolerance_relative = config.get("efficiency", {}).get("speed_tolerance_relative", None)

        self.lateral_tolerance = config.get("lane_keeping", {}).get(
            "lateral_tolerance", 0.5
        )
        self.heading_tolerance = config.get("lane_keeping", {}).get(
            "heading_tolerance", 0.1
        )

        self.jerk_threshold = config.get("comfort", {}).get("jerk_threshold", 3.0)

        _safety_cfg = config.get("safety", {})
        self.collision_penalty_low = _safety_cfg.get("collision_penalty_low", -5.0)
        self.collision_penalty_medium = _safety_cfg.get("collision_penalty_medium", -25.0)
        self.collision_penalty_high = _safety_cfg.get("collision_penalty_high", -100.0)
        self.collision_low_speed_threshold = _safety_cfg.get(
            "collision_low_speed_threshold", 2.0
        )
        self.collision_medium_speed_threshold = _safety_cfg.get(
            "collision_medium_speed_threshold", 5.0
        )
        self.offroad_penalty = _safety_cfg.get("off_road_penalty", -20.0)

        self.wrong_way_penalty = config.get("safety", {}).get(
            "wrong_way_penalty", -200.0
        )
        self.lane_invasion_penalty = config.get("safety", {}).get(
            "lane_invasion_penalty", -10.0
        )

        self.waypoint_bonus = config.get("progress", {}).get("waypoint_bonus", 1.0)
        self.distance_scale = config.get("progress", {}).get("distance_scale", 1.0)
        self.goal_reached_bonus = config.get("progress", {}).get("goal_reached_bonus", 100.0)

        self.logger.info("PROGRESS REWARD PARAMETERS VERIFICATION (addressing WARNING-001)")
        self.logger.info("=" * 80)
        self.logger.info(f"  waypoint_bonus      : {self.waypoint_bonus:6.1f} (was 10.0)")
        self.logger.info(f"  distance_scale      : {self.distance_scale:6.1f} (was 0.1)")
        self.logger.info(f"  goal_reached_bonus  : {self.goal_reached_bonus:6.1f} (was 100.0)")
        self.logger.info("=" * 80)

        self.prev_acceleration = 0.0
        self.prev_acceleration_lateral = 0.0

        self.steering_smoothness_scale = config.get("comfort", {}).get(
            "steering_smoothness_scale", 2.0)
        self.prev_steering = 0.0

        self.steer_magnitude_weight = float(config.get("comfort", {}).get("steer_magnitude_weight", 0.0))

        self.hazard_v_floor = float(((config.get("efficiency", {}) or {}).get("hazard_target", {}) or {}).get("v_floor", 0.5))

        self.overspeed_from_target = bool((config.get("efficiency", {}) or {}).get("overspeed_from_target", False))

        self.prev_distance_to_goal = None
        self.prev_safety_potential = None

        self.discount = config.get("discount", 0.99)

        self.gamma = config.get("gamma", 0.99)

        self.step_counter = 0
        self.log_frequency = 100

    def calculate(
        self,
        velocity: float,
        lateral_deviation: float,
        heading_error: float,
        acceleration: float,
        acceleration_lateral: float,
        collision_detected: bool,
        offroad_detected: bool,
        wrong_way: bool = False,
        lane_invasion_detected: bool = False,
        distance_to_goal: float = 0.0,
        waypoint_reached: bool = False,
        goal_reached: bool = False,
        lane_half_width: float = None,
        dt: float = 0.05,
        distance_to_nearest_obstacle: float = None,
        time_to_collision: float = None,
        collision_impulse: float = None,
        wrong_way_penalty: float = 0.0,
        steering: float = 0.0,
        target_speed: Optional[float] = None,
        hazard_limited: bool = False,
    ) -> Dict:
        reward_dict = {}


        efficiency = self._calculate_efficiency_reward(velocity, heading_error, target_speed,
                                                       hazard_limited=hazard_limited)
        reward_dict["efficiency"] = efficiency

        progress = self._calculate_progress_reward(
            distance_to_goal, waypoint_reached, goal_reached
        )
        reward_dict["progress"] = progress

        lane_keeping = self._calculate_lane_keeping_reward(
            lateral_deviation, heading_error, velocity, lane_half_width, lane_invasion_detected
        )
        reward_dict["lane_keeping"] = lane_keeping

        comfort = self._calculate_comfort_reward(
            acceleration, acceleration_lateral, velocity, dt, steering=steering
        )
        reward_dict["comfort"] = comfort


        safety = self._calculate_safety_reward(
            collision_detected,
            offroad_detected,
            wrong_way,
            lane_invasion_detected,
            velocity,
            distance_to_goal,
            distance_to_nearest_obstacle,
            time_to_collision,
            collision_impulse,
            wrong_way_penalty,
        )
        reward_dict["safety"] = safety
        reward_dict["safety_terms"] = dict(self.last_safety_terms)

        total_reward = (
            self.weights["efficiency"] * efficiency
            + self.weights["lane_keeping"] * lane_keeping
            + self.weights["comfort"] * comfort
            + self.weights["safety"] * safety
            + self.weights["progress"] * progress
        )

        reward_dict["total"] = total_reward
        reward_dict["breakdown"] = {
            "efficiency": (
                self.weights["efficiency"],
                efficiency,
                self.weights["efficiency"] * efficiency,
            ),
            "lane_keeping": (
                self.weights["lane_keeping"],
                lane_keeping,
                self.weights["lane_keeping"] * lane_keeping,
            ),
            "comfort": (
                self.weights["comfort"],
                comfort,
                self.weights["comfort"] * comfort,
            ),
            "safety": (
                self.weights["safety"],
                safety,
                self.weights["safety"] * safety,
            ),
            "progress": (
                self.weights["progress"],
                progress,
                self.weights["progress"] * progress,
            ),
        }

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   REWARD BREAKDOWN (Step {self.step_counter}):\n"
                f"   ══════════════════════════════════════\n"
                f"   EFFICIENCY (target speed tracking):\n"
                f"      Raw: {efficiency:.4f}\n"
                f"      Weight: {self.weights['efficiency']:.2f}\n"
                f"      Contribution: {self.weights['efficiency']*efficiency:.4f}\n"
                f"   ──────────────────────────────────────\n"
                f"   LANE KEEPING (stay in lane):\n"
                f"      Raw: {lane_keeping:.4f}\n"
                f"      Weight: {self.weights['lane_keeping']:.2f}\n"
                f"      Contribution: {self.weights['lane_keeping']*lane_keeping:.4f}\n"
                f"   ──────────────────────────────────────\n"
                f"   COMFORT (minimize jerk):\n"
                f"      Raw: {comfort:.4f}\n"
                f"      Weight: {self.weights['comfort']:.2f}\n"
                f"      Contribution: {self.weights['comfort']*comfort:.4f}\n"
                f"   ──────────────────────────────────────\n"
                f"   SAFETY (collision/offroad penalty):\n"
                f"      Raw: {safety:.4f}\n"
                f"      Weight: {self.weights['safety']:.2f}\n"
                f"      Contribution: {self.weights['safety']*safety:.4f}\n"
                f"      Status: {' COLLISION' if collision_detected else ' OFFROAD' if offroad_detected else 'SAFE'}\n"
                f"   ──────────────────────────────────────\n"
                f"   PROGRESS (goal-directed movement):\n"
                f"      Raw: {progress:.4f}\n"
                f"      Weight: {self.weights['progress']:.2f}\n"
                f"      Contribution: {self.weights['progress']*progress:.4f}\n"
                f"   ══════════════════════════════════════\n"
                f"    TOTAL REWARD: {total_reward:.4f}\n"
                f"   ══════════════════════════════════════"
            )

            component_magnitudes = {
                "efficiency": abs(self.weights['efficiency'] * efficiency),
                "lane_keeping": abs(self.weights['lane_keeping'] * lane_keeping),
                "comfort": abs(self.weights['comfort'] * comfort),
                "safety": abs(self.weights['safety'] * safety),
                "progress": abs(self.weights['progress'] * progress),
            }
            total_magnitude = sum(component_magnitudes.values())

            if total_magnitude > 0:
                max_component = max(component_magnitudes.items(), key=lambda x: x[1])
                component_name, magnitude = max_component
                ratio = magnitude / total_magnitude

                if ratio > 0.8:
                    self.logger.debug(
                        f"     WARNING: '{component_name}' dominates ({ratio*100:.1f}% of total magnitude)"
                    )
        else:
            self.logger.debug(
                f"[REWARD] Components - "
                f"Efficiency: {efficiency:.3f}×{self.weights['efficiency']:.1f}={self.weights['efficiency']*efficiency:.2f}, "
                f"Lane: {lane_keeping:.3f}×{self.weights['lane_keeping']:.1f}={self.weights['lane_keeping']*lane_keeping:.2f}, "
                f"Comfort: {comfort:.3f}×{self.weights['comfort']:.1f}={self.weights['comfort']*comfort:.2f}, "
                f"Safety: {safety:.3f}×{self.weights['safety']:.1f}={self.weights['safety']*safety:.2f}, "
                f"Progress: {progress:.3f}×{self.weights['progress']:.1f}={self.weights['progress']*progress:.2f}"
            )
            self.logger.debug(f"[REWARD] TOTAL: {total_reward:.2f}")

        self.step_counter += 1

        if not self.logger.isEnabledFor(logging.DEBUG) and self.step_counter % self.log_frequency == 0:
            component_magnitudes = {
                "efficiency": abs(self.weights['efficiency'] * efficiency),
                "lane_keeping": abs(self.weights['lane_keeping'] * lane_keeping),
                "comfort": abs(self.weights['comfort'] * comfort),
                "safety": abs(self.weights['safety'] * safety),
                "progress": abs(self.weights['progress'] * progress),
            }
            total_magnitude = sum(component_magnitudes.values())

            if total_magnitude > 0:
                for component, magnitude in component_magnitudes.items():
                    ratio = magnitude / total_magnitude
                    if ratio > 0.8:
                        self.logger.warning(
                            f"[REWARD] Component '{component}' is dominating: "
                            f"{ratio*100:.1f}% of total magnitude (threshold: 80%) "
                            f"[Logged at step {self.step_counter}]"
                        )

        return reward_dict

    def _calculate_efficiency_reward(self, velocity: float, heading_error: float,
                                     target_speed: Optional[float] = None,
                                     hazard_limited: bool = False) -> float:
        forward_velocity = velocity * np.cos(heading_error)

        if hazard_limited and target_speed is not None:
            v_t = float(target_speed)
            efficiency = 1.0 - abs(forward_velocity - v_t) / max(v_t, self.hazard_v_floor)
            if velocity > 0.5 and v_t > self.hazard_v_floor and (not self.overspeed_from_target or forward_velocity <= v_t):
                efficiency += 0.15
            return float(np.clip(efficiency, -1.0, 1.0))

        v_t = self.target_speed if target_speed is None else float(target_speed)
        tol = (self.speed_tolerance if (target_speed is None or self.speed_tolerance_relative is None)
               else float(self.speed_tolerance_relative) * v_t)
        if forward_velocity <= v_t:
            efficiency = forward_velocity / v_t
        else:
            overspeed = max(0.0, forward_velocity - (v_t if self.overspeed_from_target else v_t + tol))
            efficiency = 1.0 - self.overspeed_penalty_scale * (overspeed / v_t)


        if velocity > 0.5 and (not self.overspeed_from_target or forward_velocity <= v_t):
            velocity_bonus = 0.15
            efficiency += velocity_bonus
            self.logger.debug(f"[EFFICIENCY-VELOCITY-BONUS] velocity={velocity:.2f} m/s → bonus={velocity_bonus:+.2f}")

        return float(np.clip(efficiency, -1.0, 1.0))

    def _calculate_lane_keeping_reward(
        self, lateral_deviation: float, heading_error: float, velocity: float,
        lane_half_width: float = None, lane_invasion_detected: bool = False
    ) -> float:
        if lane_invasion_detected:
            self.logger.warning("[LANE_KEEPING] Lane invasion detected - applying maximum penalty (-1.0)")
            return -1.0

        if velocity < 0.1:
            return 0.0

        velocity_scale = min((velocity - 0.1) / 2.9, 1.0)

        effective_tolerance = (
            lane_half_width if lane_half_width is not None
            else self.lateral_tolerance
        )

        lat_error = min(abs(lateral_deviation) / effective_tolerance, 1.0)
        lat_reward = 1.0 - lat_error * 0.7

        head_error = min(abs(heading_error) / self.heading_tolerance, 1.0)
        head_reward = 1.0 - head_error * 0.3

        lane_keeping = (lat_reward + head_reward) / 2.0 - 0.5

        lane_keeping_base = lane_keeping * velocity_scale


        direction_scale = 1.0

        lane_keeping_scaled = lane_keeping_base * direction_scale

        final_reward = float(np.clip(lane_keeping_scaled, -1.0, 1.0))
        if self.step_counter % 500 == 0:
            self.logger.debug(
                f"Lane Keeping Penalty Active: lateral_dev={lateral_deviation:.3f}m, "
                f"lat_error={lat_error:.3f}, lat_reward={lat_reward:.3f}, "
                f"final={final_reward:.3f} (lit: Chen2019 d/w, Perot2017 -d term)"
            )

        return final_reward

    def _calculate_comfort_reward(
        self, acceleration: float, acceleration_lateral: float, velocity: float, dt: float,
        steering: float = 0.0
    ) -> float:
        if velocity < 0.1:
            return 0.0

        jerk_long = (acceleration - self.prev_acceleration) / dt
        jerk_lat = (acceleration_lateral - self.prev_acceleration_lateral) / dt

        jerk_long_sq = jerk_long ** 2
        jerk_lat_sq = jerk_lat ** 2

        total_jerk = np.sqrt(jerk_long_sq + jerk_lat_sq)


        normalized_jerk = min(total_jerk / self.jerk_threshold, 2.0)

        if normalized_jerk <= 1.0:
            comfort = (1.0 - normalized_jerk) * 0.3
        else:
            excess_normalized = normalized_jerk - 1.0
            comfort = -0.3 * (excess_normalized ** 2)

        self.prev_acceleration = acceleration
        self.prev_acceleration_lateral = acceleration_lateral

        d_steer = float(steering) - float(self.prev_steering)
        self.prev_steering = float(steering)
        comfort -= self.steering_smoothness_scale * (d_steer ** 2)

        comfort = float(np.clip(comfort, -3.0, 0.3))
        if self.steer_magnitude_weight:
            comfort -= self.steer_magnitude_weight * float(steering) ** 2
        return comfort

    @staticmethod
    def _obstacle_potential(distance_to_nearest_obstacle: Optional[float]) -> float:
        if distance_to_nearest_obstacle is None:
            return 0.0
        if distance_to_nearest_obstacle >= 10.0:
            return 0.0
        return -1.0 / max(distance_to_nearest_obstacle, 0.5)

    def _calculate_safety_reward(
        self,
        collision_detected: bool,
        offroad_detected: bool,
        wrong_way: bool,
        lane_invasion_detected: bool,
        velocity: float,
        distance_to_goal: float,
        distance_to_nearest_obstacle: float = None,
        time_to_collision: float = None,
        collision_impulse: float = None,
        wrong_way_penalty: float = 0.0,
    ) -> float:
        safety = 0.0

        self.last_safety_terms = {
            "proximity": 0.0, "ttc": 0.0, "collision": 0.0, "offroad": 0.0,
            "wrong_way": 0.0, "lane_invasion": 0.0, "stopping": 0.0,
        }


        if distance_to_nearest_obstacle is not None:

            phi_current = self._obstacle_potential(distance_to_nearest_obstacle)

            if self.prev_safety_potential is None:
                proximity_shaping = 0.0
            else:
                proximity_shaping = (
                    self.discount * phi_current - self.prev_safety_potential
                )

            self.prev_safety_potential = phi_current
            safety += proximity_shaping
            self.last_safety_terms["proximity"] = float(proximity_shaping)

            if distance_to_nearest_obstacle < 10.0:
                self.logger.debug(
                    f"[SAFETY-PBRS] Obstacle @ {distance_to_nearest_obstacle:.2f}m "
                    f"-> Phi={phi_current:.3f}, shaping={proximity_shaping:+.3f}"
                )


            if time_to_collision is not None and time_to_collision < 3.0:

                ttc_penalty = -0.5 / max(time_to_collision, 0.1)
                safety += ttc_penalty
                self.last_safety_terms["ttc"] = float(ttc_penalty)

                self.logger.debug(
                    f"[SAFETY-TTC] TTC={time_to_collision:.2f}s "
                    f"→ ttc_penalty={ttc_penalty:.3f}"
                )


        if collision_detected:
            collision_speed = velocity

            if collision_speed < self.collision_low_speed_threshold:
                collision_penalty = self.collision_penalty_low
            elif collision_speed < self.collision_medium_speed_threshold:
                collision_penalty = self.collision_penalty_medium
            else:
                collision_penalty = self.collision_penalty_high

            safety += collision_penalty
            self.last_safety_terms["collision"] = float(collision_penalty)

            self.logger.warning(
                f"[SAFETY-COLLISION] speed={collision_speed:.2f} m/s ({collision_speed*3.6:.1f} km/h) "
                f"→ penalty={collision_penalty:.1f} (graduated by speed)"
            )


        if offroad_detected:
            offroad_penalty = self.offroad_penalty
            safety += offroad_penalty
            self.last_safety_terms["offroad"] = float(offroad_penalty)
            self.logger.warning(f"[SAFETY-OFFROAD] penalty={offroad_penalty:.1f}")

        if wrong_way_penalty != 0.0:
            safety += wrong_way_penalty
            self.last_safety_terms["wrong_way"] = float(wrong_way_penalty)
            self.logger.warning(f"[SAFETY-WRONG-WAY] penalty={wrong_way_penalty:.2f}")


        if lane_invasion_detected:

            safety += self.lane_invasion_penalty
            self.last_safety_terms["lane_invasion"] = float(self.lane_invasion_penalty)
            self.logger.warning(
                f"[SAFETY-LANE_INVASION] penalty={self.lane_invasion_penalty:.1f} "
                f"(crossed lane markings)"
            )


        if not collision_detected and not offroad_detected:
            if velocity < 0.5:
                stopping_penalty = -0.01

                if distance_to_goal > 20.0:
                    stopping_penalty += -0.04
                elif distance_to_goal > 10.0:
                    stopping_penalty += -0.02

                safety += stopping_penalty
                self.last_safety_terms["stopping"] = float(stopping_penalty)

                if stopping_penalty < -0.02:
                    self.logger.debug(
                        f"[SAFETY-STOPPING] velocity={velocity:.2f} m/s, "
                        f"distance_to_goal={distance_to_goal:.1f}m "
                        f"→ penalty={stopping_penalty:.2f} (REDUCED 10x from previous -0.50)"
                    )

        self.logger.debug(f"[SAFETY] Total safety reward: {safety:.3f}")

        return float(safety)

    def _calculate_progress_reward(
        self,
        distance_to_goal: float,
        waypoint_reached: bool,
        goal_reached: bool,
    ) -> float:

        if not hasattr(self, 'none_count'):
            self.none_count = 0

        if distance_to_goal is None:
            if self.prev_distance_to_goal is not None and self.prev_distance_to_goal > 0.0:
                distance_to_goal = self.prev_distance_to_goal
                self.none_count += 1

                self.logger.debug(
                    f"[PROGRESS-SMOOTH] distance_to_goal was None, "
                    f"using prev={distance_to_goal:.2f}m (none_count={self.none_count})"
                )

                if self.none_count > 50:
                    self.logger.error(
                        f"[PROGRESS-ERROR] Waypoint manager returning None persistently! "
                        f"none_count={self.none_count}, vehicle likely stuck off-route >20m. "
                        f"Investigate WaypointManager._find_nearest_segment() search window."
                    )
            else:
                self.logger.warning(
                    f"[PROGRESS] No previous distance available for smoothing, "
                    f"skipping progress reward (expected at episode start)"
                )
                return 0.0
        else:
            if self.none_count > 0:
                self.logger.info(
                    f"[PROGRESS-RECOVER] Waypoint manager recovered after {self.none_count} None values. "
                    f"Resuming normal progress tracking."
                )
                self.none_count = 0

        if distance_to_goal <= 0.0:
            self.logger.warning(
                f"[PROGRESS] Invalid distance_to_goal={distance_to_goal:.2f}m (≤0.0), "
                f"skipping progress calculation"
            )
            self.prev_distance_to_goal = None
            return 0.0

        prev_dist_str = f"{self.prev_distance_to_goal:.2f}" if self.prev_distance_to_goal is not None else "None"
        self.logger.debug(
            f"[PROGRESS] Input: route_distance={distance_to_goal:.2f}m, "
            f"waypoint_reached={waypoint_reached}, goal_reached={goal_reached}, "
            f"prev_route_distance={prev_dist_str}m"
        )

        progress = 0.0

        if self.prev_distance_to_goal is not None and self.prev_distance_to_goal > 0.0:
            distance_delta = self.prev_distance_to_goal - distance_to_goal
            distance_reward = distance_delta * self.distance_scale
            progress += distance_reward

            self.last_route_distance_delta = distance_delta

            self.logger.debug(
                f"[PROGRESS] Route Distance Delta: {distance_delta:.3f}m "
                f"({'forward' if distance_delta > 0 else 'backward'}), "
                f"Reward: {distance_reward:.2f} (scale={self.distance_scale})"
            )


        else:
            self.logger.debug(
                f"[PROGRESS] First step: initializing prev_route_distance={distance_to_goal:.2f}m"
            )
            self.last_route_distance_delta = 0.0

        self.prev_distance_to_goal = distance_to_goal

        if waypoint_reached and self.step_counter > 0:
            progress += self.waypoint_bonus
            self.logger.info(
                f"[PROGRESS] Waypoint reached! Bonus: +{self.waypoint_bonus:.1f}, "
                f"total_progress={progress:.2f}"
            )
        elif waypoint_reached and self.step_counter == 0:
            self.logger.debug(
                f"[PROGRESS] Skipping waypoint bonus at spawn (step_counter=0) "
                f"to prevent free reward before action"
            )

        clipped_progress = float(np.clip(progress, -10.0, 110.0))

        if goal_reached:
            clipped_progress += self.goal_reached_bonus
            self.logger.info(
                f"[PROGRESS]  Goal reached! Bonus: +{self.goal_reached_bonus:.1f} "
                f"(post-clip), total_progress={clipped_progress:.2f}"
            )

        if progress != clipped_progress:
            self.logger.warning(
                f"[PROGRESS]  CLIPPED: raw={progress:.2f} → clipped={clipped_progress:.2f}"
            )

        distance_rew_str = f"{distance_reward:.2f}" if self.prev_distance_to_goal is not None and 'distance_reward' in locals() else "0.00"
        waypoint_str = f"{self.waypoint_bonus:.1f}" if waypoint_reached else "0.0"
        goal_str = f"{self.goal_reached_bonus:.1f}" if goal_reached else "0.0"

        self.logger.debug(
            f"[PROGRESS] Final: progress={clipped_progress:.2f} "
            f"(route_distance_reward: {distance_rew_str}, "
            f"waypoint: {waypoint_str}, "
            f"goal: {goal_str})"
        )

        return clipped_progress

    def reset(self):
        self.prev_acceleration = 0.0
        self.prev_steering = 0.0
        self.prev_acceleration_lateral = 0.0
        self.prev_distance_to_goal = None
        self.prev_safety_potential = None
        self.step_counter = 0
        self.none_count = 0
        self.last_route_distance_delta = 0.0

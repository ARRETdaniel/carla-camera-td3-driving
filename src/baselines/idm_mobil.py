"""Intelligent Driver Model (IDM) + MOBIL lane change decision."""

import numpy as np
import carla
from typing import Dict, Tuple, Optional, List
import math


class IDMMOBILBaseline:

    def __init__(
        self,
        ego_vehicle: carla.Vehicle,
        world: carla.World,
        map_obj: carla.Map,
        scenario: int = 0,
        config_file: str = None
    ):
        self.ego_vehicle = ego_vehicle
        self.world = world
        self.map_obj = map_obj
        self.scenario = scenario

        self.v_desired = 25.0
        self.a_max = 1.5
        self.b = 2.0
        self.delta = 4.0
        self.s_0 = 2.0
        self.T = 1.5

        self.p_bias_change_left = 0.05
        self.p_bias_change_right = 0.05
        self.a_threshold = 0.1
        self.safe_distance_margin = 2.0

        self.current_speed = 0.0
        self.current_accel = 0.0
        self.lane_change_timer = 0
        self.lane_change_cooldown = 50

    def select_action(self, observation: Dict) -> Tuple[float, float]:
        self.current_speed = observation.get('ego_speed', 0.0)
        ego_transform = self.ego_vehicle.get_transform()

        desired_accel = self._compute_idm_acceleration(observation)

        lane_change_direction = self._compute_mobil_lane_change(observation)

        steering = self._compute_steering(observation, lane_change_direction)

        if desired_accel >= 0:
            throttle_brake = min(desired_accel / self.a_max, 1.0)
        else:
            throttle_brake = desired_accel / self.b

        self.lane_change_timer = max(0, self.lane_change_timer - 1)
        self.current_accel = desired_accel

        return steering, throttle_brake

    def _compute_idm_acceleration(self, observation: Dict) -> float:
        v = self.current_speed

        gap_to_leader = self._get_gap_to_leader(observation)

        if gap_to_leader < 0 or gap_to_leader > 100:
            accel = self.a_max * (1.0 - (v / self.v_desired) ** self.delta)
        else:
            delta_v = 0.0

            s_desired = self.s_0 + self.T * v + (v * delta_v) / (2.0 * math.sqrt(self.a_max * self.b))
            s_desired = max(s_desired, self.s_0)

            free_accel = self.a_max * (1.0 - (v / self.v_desired) ** self.delta)
            braking_term = (s_desired / (gap_to_leader + 1e-6)) ** 2
            accel = free_accel * (1.0 - braking_term)

        accel = np.clip(accel, -self.b, self.a_max)
        return accel

    def _get_gap_to_leader(self, observation: Dict) -> float:
        vehicles = observation.get('vehicles', [])

        if not vehicles:
            return 100.0

        ego_rot = self.ego_vehicle.get_transform().rotation
        ego_heading = math.radians(ego_rot.yaw)
        ego_pos = self.ego_vehicle.get_transform().location

        min_gap = 100.0

        for vehicle_data in vehicles:
            v_relative_pos = vehicle_data.get('relative_position', [0, 0, 0])
            v_distance_forward = v_relative_pos[0]

            if v_distance_forward > 0:
                v_distance = math.sqrt(
                    v_relative_pos[0]**2 + v_relative_pos[1]**2
                )
                if v_distance < min_gap and v_distance_forward < min_gap:
                    min_gap = v_distance

        return min_gap

    def _compute_mobil_lane_change(self, observation: Dict) -> int:
        if self.lane_change_timer > 0:
            return 0

        ego_waypoint = self._get_current_waypoint()
        if ego_waypoint is None:
            return 0

        left_lane = ego_waypoint.get_left_lane()
        right_lane = ego_waypoint.get_right_lane()

        current_accel_estimate = self.current_accel
        best_accel = current_accel_estimate
        best_direction = 0

        if left_lane is not None:
            if self._is_lane_change_safe(observation, left_lane, direction=-1):
                left_accel = current_accel_estimate + self.p_bias_change_left
                if left_accel > best_accel + self.a_threshold:
                    best_accel = left_accel
                    best_direction = -1

        if right_lane is not None:
            if self._is_lane_change_safe(observation, right_lane, direction=1):
                right_accel = current_accel_estimate - self.p_bias_change_right
                if right_accel > best_accel + self.a_threshold:
                    best_accel = right_accel
                    best_direction = 1

        if best_direction != 0:
            self.lane_change_timer = self.lane_change_cooldown

        return best_direction

    def _is_lane_change_safe(self, observation: Dict, target_waypoint: carla.Waypoint, direction: int) -> bool:
        if target_waypoint.is_junction:
            return False

        if direction < 0 and target_waypoint.lane_change != carla.LaneChange.Left and \
           target_waypoint.lane_change != carla.LaneChange.Both:
            return False
        elif direction > 0 and target_waypoint.lane_change != carla.LaneChange.Right and \
             target_waypoint.lane_change != carla.LaneChange.Both:
            return False

        vehicles = observation.get('vehicles', [])

        safe_distance_leader = self.T * self.current_speed + self.safe_distance_margin
        safe_distance_follower = 10.0

        for vehicle_data in vehicles:
            v_relative_pos = vehicle_data.get('relative_position', [0, 0, 0])
            v_lateral = abs(v_relative_pos[1])
            v_longitudinal = v_relative_pos[0]

            if v_lateral < 2.0:
                if v_longitudinal > 0 and v_longitudinal < safe_distance_leader:
                    return False
                if v_longitudinal < 0 and abs(v_longitudinal) < safe_distance_follower:
                    return False

        return True

    def _compute_steering(self, observation: Dict, lane_change_direction: int) -> float:
        ego_waypoint = self._get_current_waypoint()
        if ego_waypoint is None:
            return 0.0

        if lane_change_direction < 0:
            target_waypoint = ego_waypoint.get_left_lane()
        elif lane_change_direction > 0:
            target_waypoint = ego_waypoint.get_right_lane()
        else:
            target_waypoint = ego_waypoint

        if target_waypoint is None:
            target_waypoint = ego_waypoint

        ego_transform = self.ego_vehicle.get_transform()
        target_pos = target_waypoint.transform.location
        ego_pos = ego_transform.location

        direction = target_pos - ego_pos
        direction_norm = math.sqrt(direction.x**2 + direction.y**2)

        if direction_norm < 0.1:
            return 0.0

        ego_heading = ego_transform.rotation.yaw
        target_heading = math.atan2(direction.y, direction.x)

        heading_error = target_heading - math.radians(ego_heading)

        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi

        steering_scale = 1.0
        steering = np.clip(heading_error * steering_scale, -1.0, 1.0)

        return steering

    def _get_current_waypoint(self) -> Optional[carla.Waypoint]:
        try:
            ego_pos = self.ego_vehicle.get_transform().location
            waypoint = self.map_obj.get_waypoint(
                ego_pos,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            return waypoint
        except Exception:
            return None

    def get_name(self) -> str:
        return "IDM+MOBIL"

    def reset(self):
        self.current_speed = 0.0
        self.current_accel = 0.0
        self.lane_change_timer = 0

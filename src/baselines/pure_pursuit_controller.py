"""Pure Pursuit Controller for Lateral Vehicle Control."""

from typing import List, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PurePursuitController:

    def __init__(
        self,
        kp_lookahead: float = 0.8,
        min_lookahead: float = 10.0,
        wheelbase: float = 3.0,
        debug_log: bool = False
    ):
        self.kp_lookahead = kp_lookahead
        self.min_lookahead = min_lookahead

        self.wheelbase = wheelbase

        self.debug_log = False
        self.step_count = 0

        self.conv_rad_to_steer = 180.0 / 70.0 / np.pi

        self.pi = np.pi
        self.two_pi = 2.0 * np.pi


    def _compute_rear_axle_position(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float
    ) -> Tuple[float, float]:
        half_wheelbase = self.wheelbase / 2.0

        x_rear = current_x - half_wheelbase * np.cos(current_yaw)
        y_rear = current_y - half_wheelbase * np.sin(current_yaw)

        return x_rear, y_rear

    def _compute_lookahead_distance(self, current_speed: float) -> float:
        lookahead = max(self.min_lookahead, self.kp_lookahead * current_speed)
        return lookahead

    def _find_carrot_waypoint(
        self,
        x_rear: float,
        y_rear: float,
        waypoints: List[Tuple[float, float, float]],
        lookahead_distance: float
    ) -> Tuple[float, float]:
        carrot_x, carrot_y = waypoints[-1][0], waypoints[-1][1]

        for waypoint in waypoints:
            dist = np.sqrt(
                (waypoint[0] - x_rear)**2 +
                (waypoint[1] - y_rear)**2
            )

            if dist >= lookahead_distance:
                carrot_x = waypoint[0]
                carrot_y = waypoint[1]
                break

        return carrot_x, carrot_y

    def _normalize_angle(self, angle: float) -> float:
        return (angle + self.pi) % self.two_pi - self.pi


    def update(
        self,
        current_x: float,
        current_y: float,
        current_yaw: float,
        current_speed: float,
        waypoints: List[Tuple[float, float, float]]
    ) -> float:
        x_rear, y_rear = self._compute_rear_axle_position(
            current_x, current_y, current_yaw
        )

        lookahead_distance = self._compute_lookahead_distance(current_speed)

        carrot_x, carrot_y = self._find_carrot_waypoint(
            x_rear, y_rear, waypoints, lookahead_distance
        )

        alpha = self._normalize_angle(
            np.arctan2(carrot_y - y_rear, carrot_x - x_rear) - current_yaw
        )

        steer_rad = np.arctan2(
            2.0 * self.wheelbase * np.sin(alpha),
            lookahead_distance
        )

        steer_normalized = self.conv_rad_to_steer * steer_rad

        steer_normalized = np.clip(steer_normalized, -1.0, 1.0)


        if self.debug_log:
            self.step_count += 1
            if self.step_count % 10 == 0 or (120 <= self.step_count <= 150):
                target_y = 129.49
                crosstrack_error = current_y - target_y

                print(
                    f"[PP-DEBUG Step {self.step_count:3d}] "
                    f"Pos=({current_x:.2f}, {current_y:.4f}) "
                    f"RearAxle=({x_rear:.2f}, {y_rear:.4f}) "
                    f"Carrot=({carrot_x:.2f}, {carrot_y:.4f}) | "
                    f"Lookahead={lookahead_distance:.2f}m "
                    f"Alpha={np.degrees(alpha):+.2f}° "
                    f"Steer={steer_normalized:+.4f} "
                    f"CT_err={crosstrack_error:+.4f}m",
                    flush=False
                )

        return steer_normalized

    def __repr__(self) -> str:
        return (
            f"PurePursuitController("
            f"kp_lookahead={self.kp_lookahead}, "
            f"min_lookahead={self.min_lookahead}m, "
            f"wheelbase={self.wheelbase}m"
            f")"
        )

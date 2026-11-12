"""Combined Baseline Controller for Autonomous Vehicle Navigation."""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import carla

from src.baselines.pid_controller import PIDController
from src.baselines.pure_pursuit_controller import PurePursuitController


class BaselineController:

    def __init__(
        self,
        pid_kp: float = 0.50,
        pid_ki: float = 0.30,
        pid_kd: float = 0.13,
        integrator_min: float = 0.0,
        integrator_max: float = 10.0,
        kp_lookahead: float = 0.8,
        min_lookahead: float = 10.0,
        wheelbase: float = 3.0,
        target_speed: float = 30.0
    ):
        self.pid_controller = PIDController(
            kp=pid_kp,
            ki=pid_ki,
            kd=pid_kd,
            integrator_min=integrator_min,
            integrator_max=integrator_max
        )

        self.pure_pursuit_controller = PurePursuitController(
            kp_lookahead=kp_lookahead,
            min_lookahead=min_lookahead,
            wheelbase=wheelbase,
            debug_log=True
        )

        self.target_speed = target_speed / 3.6

        self.step_count = 0
        self.debug_log = False

    def reset(self) -> None:
        self.pid_controller.reset()
        self.step_count = 0

    def _get_target_speed_from_waypoints(
        self,
        current_x: float,
        current_y: float,
        waypoints: List[Tuple[float, float, float]],
        lookahead_distance: float = 20.0,
        max_decel: float = 1.5
    ) -> float:
        if len(waypoints) == 0:
            return self.target_speed

        waypoints_np = np.array(waypoints)

        distances = np.sqrt(
            (waypoints_np[:, 0] - current_x)**2 +
            (waypoints_np[:, 1] - current_y)**2
        )
        closest_index = np.argmin(distances)

        min_speed = waypoints_np[closest_index, 2]

        for i in range(closest_index, len(waypoints_np)):
            wp_dist = np.sqrt(
                (waypoints_np[i, 0] - current_x)**2 +
                (waypoints_np[i, 1] - current_y)**2
            )

            if wp_dist > lookahead_distance:
                break

            wp_speed = waypoints_np[i, 2]
            min_speed = min(min_speed, wp_speed)

        if self.debug_log and self.step_count % 10 == 0:
            closest_dist = distances[closest_index]
            print(f"[SPEED-LOOKAHEAD] Step {self.step_count}: "
                  f"Pos=({current_x:.2f}, {current_y:.2f}) | "
                  f"Closest WP: idx={closest_index}, dist={closest_dist:.2f}m, "
                  f"speed={waypoints_np[closest_index, 2]:.3f} m/s | "
                  f"Min speed in {lookahead_distance}m lookahead: {min_speed:.3f} m/s")

        return min_speed

    def _filter_waypoints_ahead(
        self,
        current_x: float,
        current_y: float,
        waypoints: List[Tuple[float, float, float]],
        lookahead_distance: float = 20.0
    ) -> List[Tuple[float, float, float]]:
        if len(waypoints) == 0:
            return waypoints

        waypoints_np = np.array(waypoints)

        distances = np.sqrt(
            (waypoints_np[:, 0] - current_x)**2 +
            (waypoints_np[:, 1] - current_y)**2
        )
        closest_index = np.argmin(distances)

        start_index = max(0, closest_index - 1)

        end_index = closest_index
        total_distance = 0.0

        for i in range(closest_index, len(waypoints) - 1):
            wp_dist = np.sqrt(
                (waypoints_np[i+1, 0] - waypoints_np[i, 0])**2 +
                (waypoints_np[i+1, 1] - waypoints_np[i, 1])**2
            )
            total_distance += wp_dist
            end_index = i + 1

            if total_distance >= lookahead_distance:
                break

        return waypoints[start_index:end_index+1]


    def compute_control(
        self,
        vehicle: carla.Vehicle,
        waypoints: List[Tuple[float, float, float]],
        dt: float,
        target_speed: Optional[float] = None
    ) -> carla.VehicleControl:

        transform = vehicle.get_transform()

        velocity = vehicle.get_velocity()

        current_x = transform.location.x
        current_y = transform.location.y

        current_yaw = np.radians(transform.rotation.yaw)

        current_speed = np.sqrt(
            velocity.x**2 + velocity.y**2 + velocity.z**2
        )


        if target_speed is None:
            target_speed = self._get_target_speed_from_waypoints(
                current_x=current_x,
                current_y=current_y,
                waypoints=waypoints
            )


        throttle, brake = self.pid_controller.update(
            current_speed=current_speed,
            target_speed=target_speed,
            dt=dt
        )


        filtered_waypoints = self._filter_waypoints_ahead(
            current_x=current_x,
            current_y=current_y,
            waypoints=waypoints,
            lookahead_distance=20.0
        )


        steer = self.pure_pursuit_controller.update(
            current_x=current_x,
            current_y=current_y,
            current_yaw=current_yaw,
            current_speed=current_speed,
            waypoints=filtered_waypoints
        )


        self.step_count += 1
        if self.debug_log and (
            self.step_count % 10 == 0 or
            (1340 <= self.step_count <= 1380) or
            (120 <= self.step_count <= 150)
        ):
            speed_error = target_speed - current_speed
            print(
                f"[CTRL-DEBUG Step {self.step_count:4d}] "
                f"Pos=({current_x:.2f}, {current_y:.2f}) | "
                f"Speed={current_speed:.2f} m/s | "
                f"Target={target_speed:.2f} m/s ({target_speed*3.6:.1f} km/h) | "
                f"Error={speed_error:+.2f} m/s | "
                f"Throttle={throttle:.3f} Brake={brake:.3f} Steer={steer:+.4f}",
                flush=True
            )

        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
            gear=0
        )

        return control

    def get_debug_info(self, vehicle: carla.Vehicle) -> Dict[str, Any]:
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()

        return {
            'position': {
                'x': transform.location.x,
                'y': transform.location.y,
                'z': transform.location.z
            },
            'rotation': {
                'pitch': transform.rotation.pitch,
                'yaw': transform.rotation.yaw,
                'roll': transform.rotation.roll
            },
            'speed_m_s': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            'pid_state': {
                'integral': self.pid_controller.v_error_integral,
                'prev_error': self.pid_controller.v_error_prev
            },
            'target_speed_m_s': self.target_speed,
            'target_speed_km_h': self.target_speed * 3.6
        }

    def __repr__(self) -> str:
        return (
            f"BaselineController(\n"
            f"  PID: {self.pid_controller},\n"
            f"  PurePursuit: {self.pure_pursuit_controller},\n"
            f"  target_speed={self.target_speed:.2f} m/s ({self.target_speed * 3.6:.2f} km/h)\n"
            f")"
        )

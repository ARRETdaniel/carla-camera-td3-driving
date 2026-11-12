"""PID Controller for Longitudinal Vehicle Control."""

from typing import Tuple
import numpy as np


class PIDController:

    def __init__(
        self,
        kp: float = 0.50,
        ki: float = 0.30,
        kd: float = 0.13,
        integrator_min: float = 0.0,
        integrator_max: float = 10.0
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integrator_min = integrator_min
        self.integrator_max = integrator_max

        self.v_error_integral = 0.0
        self.v_error_prev = 0.0

    def reset(self) -> None:
        self.v_error_integral = 0.0
        self.v_error_prev = 0.0

    def update(
        self,
        current_speed: float,
        target_speed: float,
        dt: float
    ) -> Tuple[float, float]:
        v_error = target_speed - current_speed

        self.v_error_integral += v_error * dt
        self.v_error_integral = np.clip(
            self.v_error_integral,
            self.integrator_min,
            self.integrator_max
        )

        if dt > 0:
            v_error_derivative = (v_error - self.v_error_prev) / dt
        else:
            v_error_derivative = 0.0

        control_output = (
            self.kp * v_error +
            self.ki * self.v_error_integral +
            self.kd * v_error_derivative
        )

        if control_output >= 0:
            throttle = np.clip(control_output, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-control_output, 0.0, 1.0)

        self.v_error_prev = v_error

        return throttle, brake

    def __repr__(self) -> str:
        return (
            f"PIDController(kp={self.kp}, ki={self.ki}, kd={self.kd}, "
            f"integrator_range=[{self.integrator_min}, {self.integrator_max}])"
        )

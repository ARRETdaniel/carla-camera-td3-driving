"""Curvature-aware target speed along a static route (added 2026-10-14, competence programme, Phase 2.1)."""
from typing import Sequence, Tuple

import numpy as np


class CurvatureSpeedProfile:

    def __init__(self, waypoints_xy: Sequence, a_lat: float = 3.0, a_dec: float = 2.0, v_cap: float = 8.33,
                 ds: float = 0.5, half_window: float = 4.0):
        wp = np.asarray(waypoints_xy, dtype=float)[:, :2]
        if len(wp) < 2:
            raise ValueError("speed profile needs at least two waypoints")
        seg = np.hypot(*np.diff(wp, axis=0).T)
        keep = np.concatenate([[True], seg > 1e-9])
        wp = wp[keep]
        self._seg = np.hypot(*np.diff(wp, axis=0).T)
        self._s_wp = np.concatenate([[0.0], np.cumsum(self._seg)])
        self.length = float(self._s_wp[-1])
        self._head = np.unwrap(np.arctan2(np.diff(wp[:, 1]), np.diff(wp[:, 0])))
        self.a_lat, self.a_dec, self.v_cap, self.half_window = float(a_lat), float(a_dec), float(v_cap), float(half_window)
        self.grid = np.arange(0.0, self.length + ds, ds)
        vlim = np.array([min(self.v_cap, np.sqrt(self.a_lat / max(self.curvature(s), 1e-9))) for s in self.grid])
        for i in range(len(self.grid) - 2, -1, -1):
            vlim[i] = min(vlim[i], np.sqrt(vlim[i + 1] ** 2 + 2.0 * self.a_dec * (self.grid[i + 1] - self.grid[i])))
        self.v = vlim

    def _heading_at(self, s: float) -> float:
        i = int(np.searchsorted(self._s_wp, s, side="right") - 1)
        return float(self._head[min(max(i, 0), len(self._head) - 1)])

    def curvature(self, s: float) -> float:
        a, b = max(s - self.half_window, 0.0), min(s + self.half_window, self.length)
        return abs(self._heading_at(b) - self._heading_at(a)) / max(b - a, 1e-6)

    def at(self, s: float) -> float:
        return float(np.interp(min(max(s, 0.0), self.length), self.grid, self.v))

    def minimum(self) -> Tuple[float, float]:
        i = int(np.argmin(self.v))
        return float(self.v[i]), float(self.grid[i])

    def ahead(self, s: float, offsets: Sequence = (0.0, 10.0, 20.0)) -> np.ndarray:
        return np.array([self.at(s + d) / self.v_cap for d in offsets], dtype=np.float32)

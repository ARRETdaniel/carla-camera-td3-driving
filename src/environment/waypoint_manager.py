"""Waypoint Manager for CARLA Autonomous Navigation"""

import numpy as np
import math
from typing import List, Tuple, Optional
import logging

try:
    import carla
except ImportError:
    carla = None


class WaypointManager:

    def __init__(
        self,
        waypoints_file: str,
        lookahead_distance: float = 50.0,
        num_waypoints_ahead: int = 10,
        waypoint_spacing: float = 5.0,
        carla_map=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.waypoints_file = waypoints_file
        self.lookahead_distance = lookahead_distance
        self.num_waypoints_ahead = num_waypoints_ahead
        self.waypoint_spacing = waypoint_spacing
        self.carla_map = carla_map

        self.waypoints = self._load_waypoints()

        self.current_waypoint_idx = 0
        self.prev_waypoint_idx = 0
        self.dense_idx = 0

        self.dense_waypoints = self._create_dense_waypoints()
        self.total_route_length = self._calculate_total_route_length()

        original_count = len(self.waypoints)
        dense_count = len(self.dense_waypoints)
        self.logger.info(
            f"Loaded {original_count} waypoints from {waypoints_file}, "
            f"interpolated to {dense_count} dense waypoints (1cm resolution), "
            f"total route length: {self.total_route_length:.2f}m"
        )

    def _load_waypoints(self) -> List[Tuple[float, float, float]]:
        waypoints = []
        try:
            with open(self.waypoints_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue

                    parts = [float(x.strip()) for x in line.split(",")]
                    if len(parts) >= 3:
                        waypoints.append((parts[0], parts[1], parts[2]))

            if not waypoints:
                raise ValueError(f"No waypoints found in {self.waypoints_file}")

            return waypoints

        except Exception as e:
            self.logger.error(f"Failed to load waypoints from {self.waypoints_file}: {e}")
            raise

    def _create_dense_waypoints(self) -> List[Tuple[float, float, float]]:
        if not self.waypoints or len(self.waypoints) < 2:
            return self.waypoints.copy() if self.waypoints else []

        INTERP_DISTANCE_RES = 0.01

        waypoints_np = np.array(self.waypoints)

        wp_distance = []
        for i in range(1, waypoints_np.shape[0]):
            dist = np.sqrt(
                (waypoints_np[i, 0] - waypoints_np[i-1, 0])**2 +
                (waypoints_np[i, 1] - waypoints_np[i-1, 1])**2
            )
            wp_distance.append(dist)

        wp_interp = []

        for i in range(waypoints_np.shape[0] - 1):
            wp_interp.append(tuple(waypoints_np[i]))

            num_pts_to_interp = int(np.floor(wp_distance[i] / INTERP_DISTANCE_RES) - 1)

            if num_pts_to_interp > 0:
                wp_vector = waypoints_np[i+1] - waypoints_np[i]
                wp_uvector = wp_vector / np.linalg.norm(wp_vector[0:2])

                for j in range(num_pts_to_interp):
                    next_wp_vector = INTERP_DISTANCE_RES * float(j+1) * wp_uvector
                    interp_point = waypoints_np[i] + next_wp_vector
                    wp_interp.append(tuple(interp_point))

        wp_interp.append(tuple(waypoints_np[-1]))

        self.logger.info(
            f"Dense waypoint interpolation: {len(self.waypoints)} original → "
            f"{len(wp_interp)} interpolated (resolution={INTERP_DISTANCE_RES}m)"
        )

        return wp_interp

    def _calculate_total_route_length(self) -> float:
        if not self.dense_waypoints or len(self.dense_waypoints) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.dense_waypoints)):
            wp_prev = self.dense_waypoints[i-1]
            wp_curr = self.dense_waypoints[i]

            dist = math.sqrt(
                (wp_curr[0] - wp_prev[0])**2 +
                (wp_curr[1] - wp_prev[1])**2
            )
            total_length += dist

        return total_length

    def dense_index_for_sparse(self, sparse_idx: int) -> int:
        if not self.dense_waypoints or sparse_idx <= 0:
            return 0
        sparse_idx = min(sparse_idx, len(self.waypoints) - 1)
        tx, ty = self.waypoints[sparse_idx][0], self.waypoints[sparse_idx][1]
        best_i, best_d = 0, float("inf")
        for i, w in enumerate(self.dense_waypoints):
            d = (w[0] - tx) ** 2 + (w[1] - ty) ** 2
            if d < best_d:
                best_d, best_i = d, i
        return best_i

    def reset(self, start_sparse_idx: int = 0):
        start_sparse_idx = max(0, min(int(start_sparse_idx), len(self.waypoints) - 1))
        self.current_waypoint_idx = start_sparse_idx
        self.prev_waypoint_idx = start_sparse_idx
        self.dense_idx = self.dense_index_for_sparse(start_sparse_idx)

    def get_next_waypoints(
        self,
        vehicle_location: Tuple[float, float, float],
        vehicle_heading: float,
    ) -> np.ndarray:
        waypoints_local = []

        self._update_current_waypoint(vehicle_location)

        for i in range(self.num_waypoints_ahead):
            idx = min(self.current_waypoint_idx + i, len(self.waypoints) - 1)
            wp_global = self.waypoints[idx]
            wp_local = self._global_to_local(
                wp_global, vehicle_location, vehicle_heading
            )

            dist = np.linalg.norm(wp_local)
            if dist <= self.lookahead_distance:
                waypoints_local.append(wp_local)
            else:
                waypoints_local.append([0.0, 0.0])

        return np.array(waypoints_local, dtype=np.float32)

    def _update_current_waypoint(self, vehicle_location):
        if hasattr(vehicle_location, 'x'):
            vx, vy, vz = vehicle_location.x, vehicle_location.y, vehicle_location.z
        else:
            vx, vy, vz = vehicle_location

        WAYPOINT_PASSED_THRESHOLD = 5.0

        if self.current_waypoint_idx < len(self.waypoints):
            wpx, wpy, wpz = self.waypoints[self.current_waypoint_idx]
            dist_to_current = math.sqrt((vx - wpx) ** 2 + (vy - wpy) ** 2)

            if dist_to_current < WAYPOINT_PASSED_THRESHOLD:
                if self.current_waypoint_idx < len(self.waypoints) - 1:
                    self.prev_waypoint_idx = self.current_waypoint_idx
                    self.current_waypoint_idx += 1

    def _global_to_local(
        self,
        global_point: Tuple[float, float, float],
        vehicle_location: Tuple[float, float, float],
        vehicle_heading: float,
    ) -> np.ndarray:
        if hasattr(global_point, 'x'):
            gx, gy = global_point.x, global_point.y
        else:
            gx, gy = global_point[0], global_point[1]

        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        dx = gx - vx
        dy = gy - vy


        cos_h = math.cos(vehicle_heading)
        sin_h = math.sin(vehicle_heading)

        local_x = cos_h * dx + sin_h * dy
        local_y = -sin_h * dx + cos_h * dy

        return np.array([local_x, local_y], dtype=np.float32)

    def get_progress(self) -> float:
        if len(self.waypoints) == 0:
            return 0.0
        return self.current_waypoint_idx / len(self.waypoints)

    def is_route_finished(self) -> bool:
        goal_threshold_segments = 300
        return self.dense_idx >= len(self.dense_waypoints) - goal_threshold_segments

    def get_target_heading(self, vehicle_location) -> float:

        if self.current_waypoint_idx >= len(self.waypoints) - 1:
            idx = len(self.waypoints) - 2
        else:
            idx = self.current_waypoint_idx

        wp_current = self.waypoints[idx]
        wp_next = self.waypoints[idx + 1]

        dx = wp_next[0] - wp_current[0]
        dy = wp_next[1] - wp_current[1]

        heading_rad = math.atan2(dy, dx)

        return heading_rad

    def get_lane_heading(self, vehicle_location) -> float:
        if getattr(self, 'carla_map', None) is None or carla is None:
            return float('nan')
        if hasattr(vehicle_location, 'x'):
            loc = vehicle_location
        else:
            loc = carla.Location(
                x=vehicle_location[0], y=vehicle_location[1],
                z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)
        wp = self.carla_map.get_waypoint(loc, project_to_road=True,
                                         lane_type=carla.LaneType.Driving)
        return float(np.radians(wp.transform.rotation.yaw)) if wp is not None else float('nan')

    def get_lateral_deviation(self, vehicle_location) -> float:
        if not hasattr(self, 'carla_map') or self.carla_map is None:
            return self._get_lateral_deviation_legacy(vehicle_location)

        if carla is None:
            return self._get_lateral_deviation_legacy(vehicle_location)

        if hasattr(vehicle_location, 'x'):
            loc = vehicle_location
        else:
            loc = carla.Location(x=vehicle_location[0],
                                y=vehicle_location[1],
                                z=vehicle_location[2] if len(vehicle_location) > 2 else 0.0)

        waypoint = self.carla_map.get_waypoint(
            loc,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            return float('inf')

        lane_center = waypoint.transform.location

        lateral_deviation = math.sqrt(
            (loc.x - lane_center.x)**2 +
            (loc.y - lane_center.y)**2
        )

        return lateral_deviation

    def get_signed_route_offset(self, vehicle_location, back_segments: int = 200,
                                ahead_segments: int = 800) -> float:
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]
        n = len(self.dense_waypoints)
        if n < 2:
            return 0.0
        lo = max(0, self.dense_idx - back_segments)
        hi = min(n - 1, self.dense_idx + ahead_segments)
        pts = np.asarray(self.dense_waypoints[lo:hi + 1], dtype=float)[:, :2]
        a, b = pts[:-1], pts[1:]
        d = b - a
        L2 = (d ** 2).sum(1)
        L2[L2 < 1e-12] = 1e-12
        p = np.array([vx, vy])
        t = np.clip(((p - a) * d).sum(1) / L2, 0.0, 1.0)
        q = a + d * t[:, None]
        i = int(np.argmin(((p - q) ** 2).sum(1)))
        tangent = d[i] / math.sqrt(L2[i])
        right = np.array([-tangent[1], tangent[0]])
        return float(np.dot(p - q[i], right))

    def lead_vehicle_ahead(self, others, horizon_m: float = 30.0, corridor_m: float = 2.0):
        n = len(self.dense_waypoints)
        others = list(others)
        if n < 2 or not others:
            return None
        lo = max(0, self.dense_idx)
        hi = min(n - 1, lo + int(horizon_m / 0.01) + 1)
        if hi - lo < 2:
            return None
        pts = np.asarray(self.dense_waypoints[lo:hi + 1], dtype=float)[:, :2]
        a, b = pts[:-1], pts[1:]
        d = b - a
        L2 = (d ** 2).sum(1)
        L2[L2 < 1e-12] = 1e-12
        seg_len = np.sqrt(L2)
        s_start = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
        best = None
        for (x, y, vx, vy) in others:
            p = np.array([x, y], dtype=float)
            t = np.clip(((p - a) * d).sum(1) / L2, 0.0, 1.0)
            q = a + d * t[:, None]
            i = int(np.argmin(((p - q) ** 2).sum(1)))
            tangent = d[i] / seg_len[i]
            lateral = float(np.dot(p - q[i], np.array([-tangent[1], tangent[0]])))
            along = float(s_start[i] + t[i] * seg_len[i])
            if along <= 0.0 or along > horizon_m or abs(lateral) > corridor_m:
                continue
            v_along = max(0.0, float(vx * tangent[0] + vy * tangent[1]))
            if best is None or along < best[0]:
                best = (along, v_along)
        return best

    def _get_lateral_deviation_legacy(self, vehicle_location) -> float:
        print("WARNING: Using legacy lateral deviation calculation (may be inaccurate at intersections)")
        if self.current_waypoint_idx >= len(self.waypoints):
            return 0.0

        if hasattr(vehicle_location, 'x'):
            vx_pos, vy_pos = vehicle_location.x, vehicle_location.y
        else:
            vx_pos, vy_pos = vehicle_location[0], vehicle_location[1]

        if self.current_waypoint_idx == 0:
            wp1 = self.waypoints[0]
        else:
            wp1 = self.waypoints[self.current_waypoint_idx - 1]

        wp2 = self.waypoints[self.current_waypoint_idx]

        route_dx = wp2[0] - wp1[0]
        route_dy = wp2[1] - wp1[1]
        route_length = math.sqrt(route_dx**2 + route_dy**2)

        if route_length < 0.001:
            return 0.0

        vx = vx_pos - wp1[0]
        vy = vy_pos - wp1[1]

        cross = route_dx * vy - route_dy * vx
        lateral_dev = cross / route_length

        return lateral_dev

    def get_distance_to_goal_euclidean(self, vehicle_location: Tuple[float, float, float]) -> float:
        if not self.waypoints or len(self.waypoints) == 0:
            return None

        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        goal_x, goal_y, _ = self.waypoints[-1]

        distance = math.sqrt((goal_x - vx) ** 2 + (goal_y - vy) ** 2)
        return distance

    def get_route_distance_to_goal(self, vehicle_location: Tuple[float, float, float]) -> float:
        if not self.dense_waypoints or len(self.dense_waypoints) == 0:
            return None

        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]


        nearest_segment_idx = max(0, self.dense_idx)
        t_final = 0.0
        min_dist = float('inf')
        found_valid_segment = False

        max_search = min(len(self.dense_waypoints) - 1, self.dense_idx + 200)

        for i in range(self.dense_idx, max_search):
            wp_a = self.dense_waypoints[i]
            wp_b = self.dense_waypoints[i + 1]

            seg_x = wp_b[0] - wp_a[0]
            seg_y = wp_b[1] - wp_a[1]
            seg_length_sq = seg_x ** 2 + seg_y ** 2

            if seg_length_sq < 1e-6:
                continue

            t_unclamped = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq

            if t_unclamped < 0.0:
                nearest_segment_idx = i
                t_final = 0.0
                found_valid_segment = True
                break
            elif 0.0 <= t_unclamped <= 1.0:
                nearest_segment_idx = i
                t_final = t_unclamped
                found_valid_segment = True

                closest_x = wp_a[0] + t_final * seg_x
                closest_y = wp_a[1] + t_final * seg_y
                min_dist = math.sqrt((vx - closest_x) ** 2 + (vy - closest_y) ** 2)
                break

        if not found_valid_segment:
            if max_search == len(self.dense_waypoints) - 1:
                self.logger.debug(
                    f"[DENSE_WP_PROJ] Vehicle past goal waypoint! "
                    f"Vehicle=({vx:.2f}, {vy:.2f}), "
                    f"LastWaypoint={self.dense_waypoints[-1]}, "
                    f"Distance=0.00m (GOAL REACHED!)"
                )
                return 0.0
            else:
                nearest_segment_idx = max_search - 1
                wp_a = self.dense_waypoints[nearest_segment_idx]
                wp_b = self.dense_waypoints[nearest_segment_idx + 1]

                seg_x = wp_b[0] - wp_a[0]
                seg_y = wp_b[1] - wp_a[1]
                seg_length_sq = seg_x ** 2 + seg_y ** 2

                if seg_length_sq > 1e-6:
                    t_final = ((vx - wp_a[0]) * seg_x + (vy - wp_a[1]) * seg_y) / seg_length_sq
                    t_final = max(0.0, min(1.0, t_final))
                else:
                    t_final = 0.0

        if nearest_segment_idx > self.dense_idx:
            self.dense_idx = nearest_segment_idx


        wp_a = self.dense_waypoints[nearest_segment_idx]
        wp_b = self.dense_waypoints[nearest_segment_idx + 1]

        seg_x = wp_b[0] - wp_a[0]
        seg_y = wp_b[1] - wp_a[1]
        seg_length_sq = seg_x ** 2 + seg_y ** 2

        t = t_final
        segment_length = math.sqrt(seg_length_sq)

        if min_dist == float('inf'):
            closest_x = wp_a[0] + t * seg_x
            closest_y = wp_a[1] + t * seg_y
            min_dist = math.sqrt((vx - closest_x) ** 2 + (vy - closest_y) ** 2)

        arc_on_current_segment = (1.0 - t) * segment_length

        arc_on_remaining_segments = 0.0
        for i in range(nearest_segment_idx + 1, len(self.dense_waypoints) - 1):
            wp_curr = self.dense_waypoints[i]
            wp_next = self.dense_waypoints[i + 1]

            segment_dist = math.sqrt(
                (wp_next[0] - wp_curr[0])**2 +
                (wp_next[1] - wp_curr[1])**2
            )
            arc_on_remaining_segments += segment_dist

        distance_to_goal = arc_on_current_segment + arc_on_remaining_segments

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"[DENSE_WP_PROJ] Vehicle=({vx:.2f}, {vy:.2f}), "
                f"SegmentIdx={nearest_segment_idx}/{len(self.dense_waypoints)-1}, "
                f"t={t:.4f}, "
                f"PerpendicularDist={min_dist:.3f}m, "
                f"ArcLength={distance_to_goal:.2f}m"
            )

        return distance_to_goal

    def _find_nearest_waypoint_index(self, vehicle_location: Tuple[float, float, float]) -> Optional[int]:
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        min_distance = float('inf')
        nearest_idx = None

        search_start = max(0, self.current_waypoint_idx - 2)
        search_end = min(len(self.waypoints), self.current_waypoint_idx + 10)

        for i in range(search_start, search_end):
            wp = self.waypoints[i]
            dist = math.sqrt((wp[0] - vx) ** 2 + (wp[1] - vy) ** 2)

            if dist < min_distance:
                min_distance = dist
                nearest_idx = i

        if min_distance > 20.0:
            return None

        return nearest_idx


    def _find_nearest_segment(self, vehicle_location: Tuple[float, float, float]) -> Tuple[Optional[int], float]:
        if hasattr(vehicle_location, 'x'):
            vx, vy = vehicle_location.x, vehicle_location.y
        else:
            vx, vy = vehicle_location[0], vehicle_location[1]

        min_distance = float('inf')
        nearest_segment_idx = None

        search_start = max(0, self.current_waypoint_idx - 2)
        search_end = min(len(self.waypoints) - 1, self.current_waypoint_idx + 10)

        for i in range(search_start, search_end):
            wp_start = self.waypoints[i]
            wp_end = self.waypoints[i + 1]


            seg_x = wp_end[0] - wp_start[0]
            seg_y = wp_end[1] - wp_start[1]
            seg_length_sq = seg_x ** 2 + seg_y ** 2

            if seg_length_sq < 1e-6:
                dist = math.sqrt((wp_start[0] - vx) ** 2 + (wp_start[1] - vy) ** 2)
            else:
                t = ((vx - wp_start[0]) * seg_x + (vy - wp_start[1]) * seg_y) / seg_length_sq

                t = max(0.0, min(1.0, t))

                closest_x = wp_start[0] + t * seg_x
                closest_y = wp_start[1] + t * seg_y

                dist = math.sqrt((vx - closest_x) ** 2 + (vy - closest_y) ** 2)

            if dist < min_distance:
                min_distance = dist
                nearest_segment_idx = i

        if min_distance > 20.0:
            self.logger.warning(
                f"[FIND_NEAREST_SEGMENT] Vehicle off-route: "
                f"min_distance={min_distance:.2f}m > 20m threshold"
            )
            return None, min_distance

        return nearest_segment_idx, min_distance

    def _project_onto_segment(
        self,
        point: Tuple[float, float],
        segment_start: Tuple[float, float],
        segment_end: Tuple[float, float]
    ) -> Tuple[float, float]:
        px, py = point
        ax, ay = segment_start
        bx, by = segment_end

        vx = bx - ax
        vy = by - ay

        wx = px - ax
        wy = py - ay

        v_dot_v = vx * vx + vy * vy

        if v_dot_v < 1e-6:
            return (ax, ay)

        t = (wx * vx + wy * vy) / v_dot_v

        t = max(0.0, min(1.0, t))

        proj_x = ax + t * vx
        proj_y = ay + t * vy

        return (proj_x, proj_y)

    def check_waypoint_reached(self) -> bool:
        waypoint_reached = self.current_waypoint_idx > self.prev_waypoint_idx
        self.prev_waypoint_idx = self.current_waypoint_idx
        return waypoint_reached

    def check_goal_reached(self, vehicle_location: Tuple[float, float, float], threshold: float = 5.0) -> bool:
        distance_to_goal = self.get_distance_to_goal_euclidean(vehicle_location)
        return distance_to_goal < threshold

    def get_progress_percentage(self) -> float:
        if len(self.waypoints) == 0:
            return 0.0

        return (self.current_waypoint_idx / (len(self.waypoints) - 1)) * 100.0

    def get_current_waypoint_index(self) -> int:
        return self.current_waypoint_idx

"""Dynamic Route Manager for CARLA Navigation"""

import numpy as np
import logging
from typing import List, Tuple, Optional
import carla

try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
except ImportError:
    logging.warning(
        "Could not import GlobalRoutePlanner from agents.navigation. "
        "Ensure CARLA Python API is installed and PythonAPI/carla/agents is in PYTHONPATH."
    )
    GlobalRoutePlanner = None


class DynamicRouteManager:

    def __init__(
        self,
        carla_world: carla.World,
        start_location: Tuple[float, float, float],
        end_location: Tuple[float, float, float],
        sampling_resolution: float = 2.0,
        logger: Optional[logging.Logger] = None
    ):
        self.world = carla_world
        self.map = carla_world.get_map()
        self.sampling_resolution = sampling_resolution
        self.logger = logger or logging.getLogger(__name__)

        self._start_location_raw = start_location
        self._end_location_raw = end_location

        self.start_waypoint = self._get_road_waypoint(start_location)
        self.end_waypoint = self._get_road_waypoint(end_location)

        if self.start_waypoint is None or self.end_waypoint is None:
            raise RuntimeError(
                f"Could not find road waypoints for start {start_location} or end {end_location}"
            )

        if GlobalRoutePlanner is None:
            raise ImportError(
                "GlobalRoutePlanner not available. Ensure CARLA agents package is installed."
            )

        self.route_planner = GlobalRoutePlanner(self.map, self.sampling_resolution)

        self.route = []
        self.waypoints = []
        self._generate_route()

        self.logger.info(
            f"DynamicRouteManager initialized:\n"
            f"  Start: ({self.start_waypoint.transform.location.x:.2f}, "
            f"{self.start_waypoint.transform.location.y:.2f}, "
            f"{self.start_waypoint.transform.location.z:.2f})\n"
            f"  End: ({self.end_waypoint.transform.location.x:.2f}, "
            f"{self.end_waypoint.transform.location.y:.2f}, "
            f"{self.end_waypoint.transform.location.z:.2f})\n"
            f"  Total waypoints: {len(self.waypoints)}\n"
            f"  Sampling resolution: {self.sampling_resolution}m"
        )

    def _get_road_waypoint(self, location: Tuple[float, float, float]) -> Optional[carla.Waypoint]:
        carla_location = carla.Location(x=location[0], y=location[1], z=0.0)

        waypoint = self.map.get_waypoint(
            carla_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is None:
            self.logger.warning(
                f"No road waypoint found at ({location[0]:.2f}, {location[1]:.2f})"
            )

        return waypoint

    def _generate_route(self):
        self.route = self.route_planner.trace_route(
            self.start_waypoint.transform.location,
            self.end_waypoint.transform.location
        )

        if not self.route:
            raise RuntimeError(
                f"GlobalRoutePlanner could not find route from "
                f"({self.start_waypoint.transform.location.x:.2f}, "
                f"{self.start_waypoint.transform.location.y:.2f}) to "
                f"({self.end_waypoint.transform.location.x:.2f}, "
                f"{self.end_waypoint.transform.location.y:.2f})"
            )

        self.waypoints = np.array([
            [
                wp.transform.location.x,
                wp.transform.location.y,
                wp.transform.location.z
            ]
            for wp, _ in self.route
        ])

        self.logger.info(
            f"Route generated with {len(self.waypoints)} waypoints "
            f"(~{len(self.waypoints) * self.sampling_resolution:.0f}m total distance)"
        )

    def get_waypoints(self) -> np.ndarray:
        return self.waypoints

    def get_start_transform(self) -> carla.Transform:
        first_wp = self.route[0][0]

        spawn_location = carla.Location(
            x=first_wp.transform.location.x,
            y=first_wp.transform.location.y,
            z=first_wp.transform.location.z + 0.5
        )

        spawn_rotation = first_wp.transform.rotation

        return carla.Transform(spawn_location, spawn_rotation)

    def get_next_waypoint_index(
        self,
        vehicle_location: carla.Location,
        current_index: int = 0
    ) -> int:
        vehicle_waypoint = self.map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if vehicle_waypoint is None:
            self.logger.warning(
                f"Vehicle at ({vehicle_location.x:.2f}, {vehicle_location.y:.2f}) "
                f"not on drivable road. Using Euclidean fallback."
            )
            return self._get_nearest_waypoint_euclidean(vehicle_location, current_index)

        vehicle_s = vehicle_waypoint.s
        vehicle_road_id = vehicle_waypoint.road_id
        vehicle_lane_id = vehicle_waypoint.lane_id

        min_s_difference = float('inf')
        best_index = current_index

        search_start = max(0, current_index - 5)
        search_end = min(len(self.waypoints), current_index + 20)

        for i in range(search_start, search_end):
            route_waypoint = self.map.get_waypoint(
                carla.Location(
                    x=self.waypoints[i][0],
                    y=self.waypoints[i][1],
                    z=self.waypoints[i][2]
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            if route_waypoint is None:
                continue

            if route_waypoint.road_id != vehicle_road_id:
                continue

            s_difference = route_waypoint.s - vehicle_s

            if s_difference >= -2.0 and s_difference < min_s_difference and i >= current_index:
                min_s_difference = s_difference
                best_index = i

        return best_index

    def _get_nearest_waypoint_euclidean(
        self,
        vehicle_location: carla.Location,
        current_index: int = 0
    ) -> int:
        vehicle_pos = np.array([
            vehicle_location.x,
            vehicle_location.y,
            vehicle_location.z
        ])

        min_distance = float('inf')
        best_index = current_index

        search_start = max(0, current_index - 5)
        search_end = min(len(self.waypoints), current_index + 20)

        for i in range(search_start, search_end):
            distance = np.linalg.norm(self.waypoints[i] - vehicle_pos)

            if distance < min_distance and i >= current_index:
                min_distance = distance
                best_index = i

        return best_index

    def regenerate_route(self, new_end_location: Optional[Tuple[float, float, float]] = None):
        if new_end_location is not None:
            self.end_waypoint = self._get_road_waypoint(new_end_location)
            if self.end_waypoint is None:
                self.logger.warning(
                    f"Could not find road waypoint for new end {new_end_location}, "
                    f"keeping original destination"
                )
                return

        self._generate_route()
        self.logger.info("Route regenerated")

    def get_route_length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(self.waypoints) - 1):
            total_length += np.linalg.norm(
                self.waypoints[i + 1] - self.waypoints[i]
            )

        return total_length

    def is_route_complete(
        self,
        vehicle_location: carla.Location,
        threshold_distance: float = 5.0
    ) -> bool:
        vehicle_pos = np.array([
            vehicle_location.x,
            vehicle_location.y,
            vehicle_location.z
        ])

        distance_to_end = np.linalg.norm(self.waypoints[-1] - vehicle_pos)

        return distance_to_end < threshold_distance

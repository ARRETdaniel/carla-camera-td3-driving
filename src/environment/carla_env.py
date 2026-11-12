"""CARLA Autonomous Vehicle Environment (Gym Interface)"""

import numpy as np
import logging
import math
import time
import os
from typing import Dict, Tuple, Optional, Any
from collections import deque
import yaml

import carla
from gymnasium import Env, spaces

from .sensors import SensorSuite
from .waypoint_manager import WaypointManager
from .dynamic_route_manager import DynamicRouteManager
from .reward_functions import RewardCalculator


OFFROAD_LATERAL_THRESHOLD_M = 2.5


class CARLANavigationEnv(Env):

    def __init__(
        self,
        carla_config_path: str,
        td3_config_path: str,
        training_config_path: str,
        host: str = "localhost",
        port: int = 2000,
        headless: bool = True,
        tm_port: Optional[int] = None,
        use_ros_bridge: bool = False,
    ):
        self.logger = logging.getLogger(__name__)

        self.use_ros_bridge = use_ros_bridge
        self.ros_interface = None

        if self.use_ros_bridge:
            try:
                print("[ROS BRIDGE] Importing ROSBridgeInterface...")
                self.logger.info("[ROS BRIDGE] Importing ROSBridgeInterface...")
                from src.utils.ros_bridge_interface import ROSBridgeInterface, ROS2_AVAILABLE

                print(f"[ROS BRIDGE] ROS2_AVAILABLE = {ROS2_AVAILABLE}")

                if not ROS2_AVAILABLE:
                    print("[ROS BRIDGE] WARNING: rclpy not available, skipping ROS initialization")
                    raise RuntimeError("rclpy not available")

                print("[ROS BRIDGE] Creating ROSBridgeInterface instance...")
                self.logger.info("[ROS BRIDGE] Creating ROSBridgeInterface instance...")
                self.ros_interface = ROSBridgeInterface(
                    node_name='carla_env_controller',
                    use_docker_exec=False
                )
                print("[ROS BRIDGE] ROSBridgeInterface created successfully!")
                self.logger.info("[ROS BRIDGE] Initialized ROS Bridge interface for vehicle control")

                print("[ROS BRIDGE] Waiting for ROS topics to be available (10s timeout)...")
                self.logger.info("[ROS BRIDGE] Waiting for ROS topics to be available (10s timeout)...")
                if not self.ros_interface.wait_for_topics(timeout=10.0):
                    print("[ROS BRIDGE] WARNING: ROS topics not available, but continuing with direct API + ROS publishing")
                    self.logger.warning("[ROS BRIDGE] ROS topics not available, but continuing with direct API + ROS publishing")
                else:
                    print("[ROS BRIDGE] Successfully connected to ROS topics")
                    self.logger.info("[ROS BRIDGE] Successfully connected to ROS topics")
            except Exception as e:
                print(f"[ROS BRIDGE] EXCEPTION: {type(e).__name__}: {e}")
                self.logger.error(f"[ROS BRIDGE] Failed to initialize ROS Bridge: {type(e).__name__}: {e}")
                import traceback
                self.logger.error(f"[ROS BRIDGE] Traceback:\n{traceback.format_exc()}")
                self.logger.warning("[ROS BRIDGE] Falling back to direct CARLA API control")
                self.use_ros_bridge = False
                self.ros_interface = None

        with open(carla_config_path, "r") as f:
            self.carla_config = yaml.safe_load(f)
        with open(td3_config_path, "r") as f:
            self.td3_config = yaml.safe_load(f)
        with open(training_config_path, "r") as f:
            self.training_config = yaml.safe_load(f)

        _fixes = self.carla_config.get("fixes", {}) or {}
        self._fix_spawn_heading = bool(_fixes.get("spawn_heading_atan2", False))
        self._fix_npc_exclusion = bool(_fixes.get("npc_exclusion_from_spawn_transform", False))
        self._ego_spawn_location = None
        _obs = self.carla_config.get("observation", {}) or {}
        self._lateral_reference = str(_obs.get("lateral_reference", "nearest_lane"))
        if self._lateral_reference not in ("nearest_lane", "route"):
            raise ValueError(f"observation.lateral_reference must be nearest_lane or route, not {self._lateral_reference!r}")
        self._extra_vector = list(_obs.get("extra_vector", []) or [])
        _unknown = set(self._extra_vector) - {"prev_action", "yaw_rate", "lateral_velocity", "target_speed_ahead"}
        if _unknown:
            raise ValueError(f"unknown observation.extra_vector entries: {sorted(_unknown)}")
        self._prev_action = np.zeros(2, dtype=np.float32)

        self.host = host
        self.port = port
        self.tm_port = tm_port
        self.client = None
        self.world = None
        self.traffic_manager = None

        self._connect_to_carla()

        map_name = self.carla_config.get("simulation", {}).get("map", "Town01")
        self.logger.info(f"Loading map: {map_name}")
        self.world = self.client.load_world(map_name)

        self._original_settings = self.world.get_settings()
        self.logger.debug(
            f"Stored original world settings: "
            f"sync={self._original_settings.synchronous_mode}, "
            f"delta={self._original_settings.fixed_delta_seconds}"
        )

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = (
            1.0 / self.carla_config.get("simulation", {}).get("fps", 20)
        )
        self.world.apply_settings(settings)

        self.fixed_delta_seconds = settings.fixed_delta_seconds

        self.logger.info(f"Synchronous mode enabled: delta={settings.fixed_delta_seconds}s")

        weather_cfg = self.carla_config.get("world", {}).get("weather", {}) or {}
        if weather_cfg:
            weather = self.world.get_weather()
            applied, skipped = [], []
            for key, value in weather_cfg.items():
                if hasattr(weather, key):
                    setattr(weather, key, float(value))
                    applied.append(key)
                else:
                    skipped.append(key)
            self.world.set_weather(weather)
            self.logger.info(
                f"Weather applied from config ({len(applied)} fields): "
                f"cloudiness={weather.cloudiness:.0f} precipitation={weather.precipitation:.0f} "
                f"wetness={weather.wetness:.0f} sun_altitude={weather.sun_altitude_angle:.0f}"
            )
            if skipped:
                self.logger.warning(
                    f"Weather keys not supported by this CARLA build, ignored: {skipped}"
                )
        else:
            self.logger.warning(
                "No world.weather in config - the server's default weather is in force and "
                "is NOT recorded with the run. See weather_and_quality.md."
            )

        if headless:
            self.world.unload_map_layer(carla.MapLayer.All)

        self.spawn_points = self.world.get_map().get_spawn_points()
        if not self.spawn_points:
            raise RuntimeError("No spawn points available in map")

        self.vehicle = None
        self.sensors = None
        self.npcs = []

        waypoints_file = self.carla_config.get("route", {}).get(
            "waypoints_file", "waypoints.txt"
        )

        carla_map = self.world.get_map()

        legacy_waypoint_manager = WaypointManager(
            waypoints_file=waypoints_file,
            lookahead_distance=self.carla_config.get("route", {}).get(
                "lookahead_distance", 50.0
            ),
            num_waypoints_ahead=self.carla_config.get("route", {}).get(
                "num_waypoints_ahead", 25
            ),
            carla_map=carla_map,
        )

        route_start_legacy = legacy_waypoint_manager.waypoints[0]
        route_end_legacy = legacy_waypoint_manager.waypoints[-1]

        self.logger.info(
            f"Route definition (from {waypoints_file}):\n"
            f"  Start: ({route_start_legacy[0]:.2f}, {route_start_legacy[1]:.2f}, {route_start_legacy[2]:.2f})\n"
            f"  End: ({route_end_legacy[0]:.2f}, {route_end_legacy[1]:.2f}, {route_end_legacy[2]:.2f})"
        )

        self.use_dynamic_routes = self.carla_config.get("route", {}).get(
            "use_dynamic_generation", True
        )

        if self.use_dynamic_routes:
            self._route_start_location = route_start_legacy
            self._route_end_location = route_end_legacy

            sampling_resolution = self.carla_config.get("route", {}).get(
                "sampling_resolution", 2.0
            )

            try:
                self.route_manager = DynamicRouteManager(
                    carla_world=self.world,
                    start_location=self._route_start_location,
                    end_location=self._route_end_location,
                    sampling_resolution=sampling_resolution,
                    logger=self.logger
                )
                self.waypoint_manager = self._create_waypoint_manager_adapter()
                self.logger.info(
                    f"DynamicRouteManager initialized: {len(self.route_manager.waypoints)} waypoints, "
                    f"~{self.route_manager.get_route_length():.0f}m route length"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize DynamicRouteManager: {e}\n"
                    f"Falling back to legacy waypoint manager"
                )
                self.use_dynamic_routes = False
                self.waypoint_manager = legacy_waypoint_manager
                self.route_manager = None
        else:
            self.waypoint_manager = legacy_waypoint_manager
            self.route_manager = None
            self.logger.info("Using legacy static waypoints from file")

        reward_config = self.training_config.get("reward", {})
        self.reward_calculator = RewardCalculator(reward_config)

        _ct = ((reward_config.get("efficiency", {}) or {}).get("curvature_target", {}) or {})
        self.speed_profile = None
        if _ct.get("enabled", False):
            from .speed_profile import CurvatureSpeedProfile
            self.speed_profile = CurvatureSpeedProfile(
                self.waypoint_manager.waypoints,
                a_lat=float(_ct.get("a_lat", 3.0)), a_dec=float(_ct.get("a_dec", 2.0)),
                v_cap=float(_ct.get("v_cap", reward_config.get("efficiency", {}).get("target_speed", 8.33))),
                half_window=float(_ct.get("half_window_m", 4.0)))
            _vmin, _smin = self.speed_profile.minimum()
            self.logger.info(f"Curvature-aware target speed ON: min {_vmin:.2f} m/s at {_smin:.1f} m "
                             f"of {self.speed_profile.length:.1f} m (a_lat {_ct.get('a_lat', 3.0)}, "
                             f"a_dec {_ct.get('a_dec', 2.0)})")

        _ht = ((reward_config.get("efficiency", {}) or {}).get("hazard_target", {}) or {})
        self._hazard_target = None
        if _ht.get("enabled", False):
            self._hazard_target = {"d_stop": float(_ht.get("d_stop", 8.0)), "a_dec": float(_ht.get("a_dec", 2.0)),
                                   "horizon_m": float(_ht.get("horizon_m", 30.0)),
                                   "corridor_m": float(_ht.get("corridor_m", 2.0))}
            self.logger.info(f"Hazard-aware target speed ON: {self._hazard_target}")

        self._route_pool = None
        self._route_id = os.path.splitext(os.path.basename(str(waypoints_file)))[0]
        _pool = (self.carla_config.get("route", {}) or {}).get("pool", {}) or {}
        if _pool.get("enabled", False):
            if self.use_dynamic_routes:
                raise ValueError("route.pool needs route.use_dynamic_generation: false (one file route per entry)")
            from .route_pool import RoutePool

            def _pool_manager(f):
                return WaypointManager(
                    waypoints_file=f,
                    lookahead_distance=self.carla_config.get("route", {}).get("lookahead_distance", 50.0),
                    num_waypoints_ahead=self.carla_config.get("route", {}).get("num_waypoints_ahead", 25),
                    carla_map=carla_map,
                )

            _pool_profile = None
            if self.speed_profile is not None:
                from .speed_profile import CurvatureSpeedProfile

                def _pool_profile(wps):
                    return CurvatureSpeedProfile(
                        wps, a_lat=float(_ct.get("a_lat", 3.0)), a_dec=float(_ct.get("a_dec", 2.0)),
                        v_cap=float(_ct.get("v_cap", reward_config.get("efficiency", {}).get("target_speed", 8.33))),
                        half_window=float(_ct.get("half_window_m", 4.0)))

            _pool_seed = _pool.get("seed", None)
            if _pool_seed is None:
                _pool_seed = int(np.random.randint(0, 2 ** 31 - 1))
            self._route_pool = RoutePool(_pool.get("files", []) or [], _pool_manager, _pool_profile, seed=int(_pool_seed))
            self.logger.info(f"Route pool ON: {len(self._route_pool)} routes, seed {_pool_seed}: "
                             f"{[e['name'] for e in self._route_pool.entries]}")

        self._target_speed_obs = "target_speed_ahead" in self._extra_vector
        self._route_s = 0.0
        if self._target_speed_obs and self.speed_profile is None:
            raise ValueError("observation.extra_vector target_speed_ahead needs reward.efficiency.curvature_target.enabled")

        self.current_step = 0
        self.episode_start_time = None
        self.episode_count = 0

        self.goal_bonus_awarded = False

        self.max_episode_steps = self.carla_config.get("episode", {}).get("max_time_steps", 1000)
        self._steps_per_m = self.carla_config.get("episode", {}).get("max_time_steps_per_m", None)

        self._setup_spaces()

        self._closed = False

        self.logger.info(
            f"CARLA environment initialized: {map_name}, {self.max_episode_steps} steps/episode"
        )

    def _create_waypoint_manager_adapter(self):
        class WaypointManagerAdapter:
            def __init__(self, route_manager, lookahead_distance, sampling_resolution, carla_map):
                self.route_manager = route_manager
                self.lookahead_distance = lookahead_distance
                self.sampling_resolution = sampling_resolution
                self.carla_map = carla_map

                self.num_waypoints_ahead = int(np.ceil(lookahead_distance / sampling_resolution))

                self._current_waypoint_index = 0

                logger = logging.getLogger(__name__)
                logger.info(
                    f"WaypointManagerAdapter initialized:\n"
                    f"  Lookahead distance: {lookahead_distance}m\n"
                    f"  Sampling resolution: {sampling_resolution}m\n"
                    f"  Calculated waypoints ahead: {self.num_waypoints_ahead}\n"
                    f"  Expected coverage: {self.num_waypoints_ahead * sampling_resolution}m"
                )

            @property
            def waypoints(self):
                return self.route_manager.get_waypoints()

            def reset(self):
                self._current_waypoint_index = 0

            def get_next_waypoints(self, vehicle_location, current_index=None):
                if current_index is None:
                    current_index = self._current_waypoint_index

                self._current_waypoint_index = self.route_manager.get_next_waypoint_index(
                    vehicle_location,
                    current_index
                )

                start_idx = self._current_waypoint_index
                end_idx = min(
                    start_idx + self.num_waypoints_ahead,
                    len(self.waypoints)
                )

                return self.waypoints[start_idx:end_idx]

            def get_lateral_deviation(self, vehicle_location) -> float:
                if not self.carla_map:
                    return 0.0

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

        sampling_resolution = self.carla_config.get("route", {}).get("sampling_resolution", 2.0)

        carla_map = self.world.get_map()

        return WaypointManagerAdapter(
            route_manager=self.route_manager,
            lookahead_distance=self.carla_config.get("route", {}).get("lookahead_distance", 50.0),
            sampling_resolution=sampling_resolution,
            carla_map=carla_map
        )

    def _connect_to_carla(self, max_retries: int = 5):
        for attempt in range(max_retries):
            try:
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(
                    float(self.carla_config.get("server", {}).get("timeout", 10.0)))
                self.client.get_server_version()
                self.logger.info(f"Connected to CARLA server at {self.host}:{self.port}")
                return
            except Exception as e:
                self.logger.warning(
                    f"Connection attempt {attempt + 1}/{max_retries} failed: {e}"
                )
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(
                        f"Failed to connect to CARLA server after {max_retries} attempts"
                    )

    def _extra_vector_dim(self) -> int:
        return sum({"prev_action": 2, "target_speed_ahead": 3}.get(name, 1) for name in self._extra_vector)

    def _setup_spaces(self):
        image_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4, 84, 84),
            dtype=np.float32,
        )

        lookahead_distance = self.carla_config.get("route", {}).get("lookahead_distance", 50.0)
        sampling_resolution = self.carla_config.get("route", {}).get("sampling_resolution", 2.0)
        num_waypoints_ahead = int(np.ceil(lookahead_distance / sampling_resolution))

        vector_size = 3 + (num_waypoints_ahead * 2) + self._extra_vector_dim()

        self.logger.info(
            f"Observation space configuration:\n"
            f"  Image: (4, 84, 84)\n"
            f"  Vector: ({vector_size},) = 3 kinematic + {num_waypoints_ahead} waypoints × 2\n"
            f"  Lookahead: {lookahead_distance}m / {sampling_resolution}m = {num_waypoints_ahead} waypoints"
        )

        vector_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(vector_size,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "image": image_space,
                "vector": vector_space,
            }
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.logger.info(
            f"Observation space: {self.observation_space}\n"
            f"Action space: {self.action_space}"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.episode_count += 1

        self.logger.info(f"Resetting environment (episode {self.episode_count})...")

        self._cleanup_episode()

        if self.use_dynamic_routes and self.route_manager is not None:
            spawn_point = self.route_manager.get_start_transform()

            self.logger.info(
                f"Using DYNAMIC route (GlobalRoutePlanner):\n"
                f"   Start: ({spawn_point.location.x:.2f}, {spawn_point.location.y:.2f}, {spawn_point.location.z:.2f})\n"
                f"   Heading: {spawn_point.rotation.yaw:.2f}°\n"
                f"   Route length: ~{self.route_manager.get_route_length():.0f}m\n"
                f"   Waypoints: {len(self.route_manager.waypoints)}"
            )
        else:
            if self._route_pool is not None:
                _entry = self._route_pool.draw()
                self.waypoint_manager = _entry["manager"]
                self.speed_profile = _entry["profile"]
                self._route_id = _entry["name"]
                self.logger.info(f"[ROUTE-POOL] episode {self.episode_count}: route {_entry['index']} "
                                 f"({_entry['name']}), {len(self.waypoint_manager.waypoints)} waypoints")
            self._episode_start_idx = self._choose_start_index(options)
            _si = self._episode_start_idx

            route_start = self.waypoint_manager.waypoints[_si]

            if len(self.waypoint_manager.waypoints) >= _si + 2:
                wp0 = self.waypoint_manager.waypoints[_si]
                wp1 = self.waypoint_manager.waypoints[_si + 1]
                dx = wp1[0] - wp0[0]
                dy = wp1[1] - wp0[1]


                if self._fix_spawn_heading:
                    heading_rad = math.atan2(dy, dx)
                else:
                    heading_rad = math.atan2(-dy, dx)
                initial_yaw = math.degrees(heading_rad)
            else:
                initial_yaw = 0.0
                self.logger.warning("Only one waypoint available, using default yaw=0")

            carla_map = self.world.get_map()
            road_location = carla.Location(x=route_start[0], y=route_start[1], z=0.0)
            road_waypoint = carla_map.get_waypoint(
                road_location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )

            if road_waypoint is not None:
                spawn_z = road_waypoint.transform.location.z + 0.5
                self.logger.info(f"Using CARLA map Z: {spawn_z:.2f}m (original: {route_start[2]:.2f}m)")
            else:
                spawn_z = route_start[2] + 0.5
                self.logger.warning(f"Could not get road waypoint, using Z + 0.5m: {spawn_z:.2f}m")

            spawn_point = carla.Transform(
                carla.Location(x=route_start[0], y=route_start[1], z=spawn_z),
                carla.Rotation(pitch=0.0, yaw=initial_yaw, roll=0.0)
            )

            self.logger.info(
                f"Using LEGACY static waypoints:\n"
                f"   Total waypoints in route: {len(self.waypoint_manager.waypoints)}\n"
                f"   Spawn location: ({route_start[0]:.2f}, {route_start[1]:.2f}, {spawn_z:.2f})\n"
                f"   Spawn heading: {initial_yaw:.2f}°\n"
                f"   First 5 waypoints (X, Y, Z):\n" +
                "\n".join([f"      WP{i}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f})"
                          for i, wp in enumerate(self.waypoint_manager.waypoints[:5])]) +
                f"\n   Route direction: dx={wp1[0]-wp0[0]:.2f}, dy={wp1[1]-wp0[1]:.2f} → yaw={initial_yaw:.2f}°"
            )

        self._ego_spawn_location = spawn_point.location

        vehicle_bp = self.world.get_blueprint_library().find("vehicle.tesla.model3")

        try:
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            self.logger.info(f"Ego vehicle spawned successfully at ({route_start[0]:.2f}, {route_start[1]:.2f}, {spawn_z:.2f})")
        except RuntimeError as e:
            raise RuntimeError(f"Failed to spawn ego vehicle: {e}")

        self.sensors = SensorSuite(self.vehicle, self.carla_config, self.world)

        self._spawn_npc_traffic()

        self.current_step = 0
        self._prev_action[:] = 0.0
        self.episode_start_time = time.time()

        self.goal_bonus_awarded = False

        self.waypoint_manager.reset(getattr(self, "_episode_start_idx", 0))
        if self._steps_per_m:
            self.max_episode_steps = int(math.ceil(float(self._steps_per_m) * self.waypoint_manager.total_route_length))
            self.logger.info(f"[EPISODE-LIMIT] {self.max_episode_steps} steps = ceil({self._steps_per_m} x "
                             f"{self.waypoint_manager.total_route_length:.1f} m)")
        self.reward_calculator.reset()

        self.world.tick()
        self.sensors.tick()

        actual_transform = self.vehicle.get_transform()
        forward_vec = actual_transform.get_forward_vector()

        if len(self.waypoint_manager.waypoints) >= 2:
            wp0 = self.waypoint_manager.waypoints[0]
            wp1 = self.waypoint_manager.waypoints[1]
            expected_dx = wp1[0] - wp0[0]
            expected_dy = wp1[1] - wp0[1]
            expected_mag = math.sqrt(expected_dx**2 + expected_dy**2)
            expected_fwd = [expected_dx/expected_mag, expected_dy/expected_mag, 0.0] if expected_mag > 0 else [1.0, 0.0, 0.0]

            self.logger.info(
                f"SPAWN VERIFICATION (post-tick):\n"
                f"   Requested spawn yaw: {spawn_point.rotation.yaw:.2f}°\n"
                f"   Actual vehicle yaw: {actual_transform.rotation.yaw:.2f}°\n"
                f"   Actual forward vector: [{forward_vec.x:.3f}, {forward_vec.y:.3f}, {forward_vec.z:.3f}]\n"
                f"   Expected forward (route): [{expected_fwd[0]:.3f}, {expected_fwd[1]:.3f}, {expected_fwd[2]:.3f}]\n"
                f"   Yaw difference: {abs(actual_transform.rotation.yaw - spawn_point.rotation.yaw):.2f}°\n"
                f"   Alignment: {' ALIGNED' if abs(forward_vec.x - expected_fwd[0]) < 0.1 and abs(forward_vec.y - expected_fwd[1]) < 0.1 else ' MISALIGNED'}"
            )

        if self._target_speed_obs:
            _wps = np.asarray(self.waypoint_manager.waypoints, dtype=float)[: getattr(self, "_episode_start_idx", 0) + 1, :2]
            self._route_s = float(np.hypot(*np.diff(_wps, axis=0).T).sum()) if len(_wps) > 1 else 0.0

        observation = self._get_observation()

        route_length_m = getattr(self.waypoint_manager, "total_route_length", None)
        if not route_length_m:
            route_length_m = len(self.waypoint_manager.waypoints) * 2.0

        info = {
            "episode": self.episode_count,
            "route_length_m": route_length_m,
            "route_id": self._route_id,
            "npc_count": len(self.npcs),
            "spawn_location": {
                "x": spawn_point.location.x,
                "y": spawn_point.location.y,
                "z": spawn_point.location.z,
                "yaw": spawn_point.rotation.yaw,
            },
            "observation_shapes": {
                "image": list(observation['image'].shape),
                "vector": list(observation['vector'].shape),
            },
        }

        self.logger.info(
            f"Episode {info['episode']} reset. "
            f"Route: {info['route_length_m']:.0f}m, "
            f"NPCs: {info['npc_count']}, "
            f"Obs shapes: image {tuple(info['observation_shapes']['image'])}, "
            f"vector {tuple(info['observation_shapes']['vector'])}"
        )

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        self._apply_action(action)

        self.world.tick()
        self.sensors.tick()

        if self.current_step < 10:
            snapshot = self.world.get_snapshot()
            self.logger.info(
                f" DEBUG Step {self.current_step} - World State After Tick:\n"
                f"   Frame: {snapshot.frame}\n"
                f"   Timestamp: {snapshot.timestamp.elapsed_seconds:.3f}s\n"
                f"   Delta: {snapshot.timestamp.delta_seconds:.3f}s"
            )

        observation = self._get_observation()
        vehicle_state = self._get_vehicle_state()

        vehicle_location = self.vehicle.get_location()

        distance_to_goal = self.waypoint_manager.get_route_distance_to_goal(vehicle_location)
        if self._target_speed_obs and distance_to_goal is not None:
            self._route_s = (getattr(self.waypoint_manager, "total_route_length", None)
                             or self.speed_profile.length) - distance_to_goal

        waypoint_reached = self.waypoint_manager.check_waypoint_reached()

        goal_detected = self.waypoint_manager.check_goal_reached(vehicle_location, threshold=3.0)

        goal_reached = goal_detected and not self.goal_bonus_awarded
        if goal_reached:
            self.goal_bonus_awarded = True
            self.logger.info("[GOAL] First goal detection - bonus will be awarded (flag set)")
        distance_to_nearest_obstacle = self.sensors.get_distance_to_nearest_obstacle()

        time_to_collision = None
        if (distance_to_nearest_obstacle < float('inf') and
            vehicle_state["velocity"] > 0.1):
            time_to_collision = distance_to_nearest_obstacle / vehicle_state["velocity"]

        collision_info = self.sensors.get_collision_info()
        collision_impulse = None
        if collision_info is not None and "impulse" in collision_info:
            collision_impulse = collision_info["impulse"]

        collision_count = self.sensors.get_step_collision_count()
        lane_invasion_count = self.sensors.get_step_lane_invasion_count()

        target_speed = None
        if self.speed_profile is not None and distance_to_goal is not None:
            _L = getattr(self.waypoint_manager, "total_route_length", None) or self.speed_profile.length
            target_speed = self.speed_profile.at(_L - distance_to_goal)

        hazard_limited, lead = False, None
        if self._hazard_target is not None:
            lead = self._lead_vehicle()
            if lead is not None:
                _h = self._hazard_target
                v_h = math.sqrt(max(0.0, lead[1] ** 2 + 2.0 * _h["a_dec"] * max(0.0, lead[0] - _h["d_stop"])))
                base = target_speed if target_speed is not None else float(self.reward_calculator.target_speed)
                if v_h < base:
                    target_speed, hazard_limited = v_h, True

        reward_dict = self.reward_calculator.calculate(
            velocity=vehicle_state["velocity"],
            lateral_deviation=vehicle_state["lateral_deviation"],
            heading_error=vehicle_state["heading_error"],
            acceleration=vehicle_state["acceleration"],
            acceleration_lateral=vehicle_state["acceleration_lateral"],
            collision_detected=self.sensors.is_collision_detected(),
            offroad_detected=self._is_offroad(vehicle_state),
            wrong_way=vehicle_state["wrong_way"],

            lane_invasion_detected=bool(self.sensors.get_step_lane_invasion_count()),
            distance_to_goal=distance_to_goal,
            waypoint_reached=waypoint_reached,
            goal_reached=goal_reached,
            lane_half_width=vehicle_state["lane_half_width"],
            dt=vehicle_state["dt"],
            distance_to_nearest_obstacle=distance_to_nearest_obstacle,
            time_to_collision=time_to_collision,
            collision_impulse=collision_impulse,
            wrong_way_penalty=vehicle_state.get("wrong_way_penalty", 0.0),
            steering=float(action[0]),
            target_speed=target_speed,
            hazard_limited=hazard_limited,
        )

        reward = reward_dict["total"]

        self.current_step += 1

        done, termination_reason = self._check_termination(vehicle_state)

        truncated = (self.current_step >= self.max_episode_steps) and not done
        terminated = done and not truncated

        if self.current_step >= self.max_episode_steps - 10 and not done:
            self.logger.warning(
                f"[DIAGNOSTIC] Approaching max_episode_steps! "
                f"Step {self.current_step}/{self.max_episode_steps}, "
                f"will truncate in {self.max_episode_steps - self.current_step} steps"
            )

        if truncated:
            self.logger.warning(
                f"[TERMINATION] Episode TRUNCATED at step {self.current_step} "
                f"(reached max_episode_steps={self.max_episode_steps})"
            )

        info = {
            "step": self.current_step,
            "reward_breakdown": reward_dict["breakdown"],
            "reward_components": {
                "total": reward,
                "efficiency": reward_dict["breakdown"]["efficiency"][2],
                "lane_keeping": reward_dict["breakdown"]["lane_keeping"][2],
                "comfort": reward_dict["breakdown"]["comfort"][2],
                "safety": reward_dict["breakdown"]["safety"][2],
                "progress": reward_dict["breakdown"]["progress"][2],
            },
            "state": {
                "velocity": vehicle_state["velocity"],
                "lateral_deviation": vehicle_state["lateral_deviation"],
                "heading_error": vehicle_state["heading_error"],
                "distance_to_goal": distance_to_goal,
            },
            "offroad_lane_type": self.sensors.is_offroad(),
            "termination_reason": termination_reason,
            "vehicle_state": vehicle_state,
            "collision_info": collision_info,
            "collision_count": collision_count,
            "lane_invasion_count": lane_invasion_count,
            "distance_to_goal": distance_to_goal,
            "progress_percentage": self.waypoint_manager.get_progress_percentage(),
            "safety_terms": reward_dict.get("safety_terms", {}),
            "diag_vehicle_yaw_deg": float(self.vehicle.get_transform().rotation.yaw),
            "diag_lane_heading_deg": float(np.degrees(
                self.waypoint_manager.get_lane_heading(vehicle_location))),
            "diag_route_heading_deg": float(np.degrees(
                self._route_heading_from_waypoints(vehicle_location))),
            "diag_goal_euclid_m": float(
                self.waypoint_manager.get_distance_to_goal_euclidean(vehicle_location)),
            "diag_is_route_finished": bool(self.waypoint_manager.is_route_finished()),
            "diag_check_goal_reached": bool(
                self.waypoint_manager.check_goal_reached(vehicle_location, threshold=3.0)),
            "diag_dense_idx": int(getattr(self.waypoint_manager, "dense_idx", -1)),
            "diag_dense_len": int(len(getattr(self.waypoint_manager, "dense_waypoints", []) or [])),
            "diag_lateral_dev": float(abs(vehicle_state.get("lateral_deviation", 0.0))),
            "current_waypoint_idx": self.waypoint_manager.get_current_waypoint_index(),
            "waypoint_reached": waypoint_reached,
            "goal_reached": goal_reached,
            "success": 1 if (termination_reason == "route_completed" and goal_reached) else 0,
            "distance_to_nearest_obstacle": distance_to_nearest_obstacle,
            "time_to_collision": time_to_collision,
            "collision_impulse": collision_impulse,
            "target_speed": target_speed,
            "lead_distance": (lead[0] if lead is not None else None),
            "lead_speed": (lead[1] if lead is not None else None),
            "hazard_limited": hazard_limited,
        }

        self.sensors.reset_step_counters()

        if terminated or truncated:
            self.logger.info(
                f"Episode ended: {termination_reason} after {self.current_step} steps "
                f"(terminated={terminated}, truncated={truncated})"
            )

        return observation, reward, terminated, truncated, info

    def _apply_action(self, action: np.ndarray):
        steering = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        if throttle_brake > 0:
            throttle = throttle_brake
            brake = 0.0
        else:
            throttle = 0.0
            brake = -throttle_brake

        control = carla.VehicleControl(
            throttle=throttle,
            brake=brake,
            steer=steering,
            hand_brake=False,
            reverse=False,
        )

        self.vehicle.apply_control(control)
        self._prev_action[0], self._prev_action[1] = steering, throttle_brake

        if self.use_ros_bridge and self.ros_interface is not None:
            self.ros_interface.publish_control(
                throttle=throttle,
                steer=steering,
                brake=brake,
                hand_brake=False,
                reverse=False
            )

        if self.current_step < 10000:
            velocity = self.vehicle.get_velocity()
            speed_mps = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
            speed_kmh = speed_mps * 3.6

            applied_control = self.vehicle.get_control()

            self.logger.info(
                f"DEBUG Step {self.current_step}:\n"
                f"   Input Action: steering={action[0]:+.4f}, throttle/brake={action[1]:+.4f}\n"
                f"   Sent Control: throttle={throttle:.4f}, brake={brake:.4f}, steer={steering:.4f}\n"
                f"   Applied Control: throttle={applied_control.throttle:.4f}, brake={applied_control.brake:.4f}, steer={applied_control.steer:.4f}\n"
                f"   Speed: {speed_kmh:.2f} km/h ({speed_mps:.2f} m/s)\n"
                f"   Hand Brake: {applied_control.hand_brake}, Reverse: {applied_control.reverse}, Gear: {applied_control.gear}"
            )

    def _handle_tick_timeout(self):
        self.logger.warning(
            f"   Forcing episode termination due to CARLA tick timeout\n"
            f"   Episode: {self.episode_count}, Step: {self.current_step}\n"
            f"   Last waypoint: {self.waypoint_manager.current_waypoint_idx}/{len(self.waypoint_manager.waypoints)}\n"
            f"   Recommendation: Check CARLA server logs for deadlock/error"
        )

        try:
            observation = self._get_observation()
        except Exception as e:
            self.logger.error(f"Failed to get observation during timeout recovery: {e}")
            observation = {
                "image": np.zeros((4, 84, 84), dtype=np.float32),
                "vector": np.zeros(self.observation_space["vector"].shape, dtype=np.float32),
            }

        reward = -100.0
        terminated = True
        truncated = False

        info = {
            "step": self.current_step,
            "episode": self.episode_count,
            "termination_reason": "carla_tick_timeout",
            "timeout_duration": 10.0,
            "reward_total": reward,
            "reward_components": {"timeout_penalty": -100.0},
        }

        if not hasattr(self, 'timeout_count'):
            self.timeout_count = 0
        self.timeout_count += 1

        self.logger.warning(
            f"   Total timeouts in session: {self.timeout_count}\n"
            f"   If timeouts persist, consider:\n"
            f"   1. Reducing NPC density\n"
            f"   2. Lowering sensor resolution\n"
            f"   3. Restarting CARLA server"
        )

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        image_obs = self.sensors.get_camera_data()

        vehicle_state = self._get_vehicle_state()

        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_heading_radians = np.radians(vehicle_transform.rotation.yaw)
        next_waypoints = self.waypoint_manager.get_next_waypoints(
            vehicle_location, vehicle_heading_radians
        )

        expected_num_waypoints = self.waypoint_manager.num_waypoints_ahead

        if len(next_waypoints) < expected_num_waypoints:
            if len(next_waypoints) > 0:
                last_waypoint = next_waypoints[-1]
                padding = np.tile(last_waypoint, (expected_num_waypoints - len(next_waypoints), 1))
                next_waypoints = np.vstack([next_waypoints, padding])
            else:
                next_waypoints = np.zeros((expected_num_waypoints, 2), dtype=np.float32)


        velocity_normalized = vehicle_state["velocity"] / 30.0

        lateral_deviation_normalized = vehicle_state["lateral_deviation"] / 3.5

        heading_error_normalized = vehicle_state["heading_error"] / np.pi

        lookahead_distance = self.carla_config.get("route", {}).get("lookahead_distance", 50.0)
        waypoints_normalized = next_waypoints / lookahead_distance

        parts = [
            [velocity_normalized],
            [lateral_deviation_normalized],
            [heading_error_normalized],
            waypoints_normalized.flatten(),
        ]
        for name in self._extra_vector:
            if name == "prev_action":
                parts.append(self._prev_action.astype(np.float32))
            elif name == "yaw_rate":
                parts.append([np.radians(self.vehicle.get_angular_velocity().z)])
            elif name == "lateral_velocity":
                _v = self.vehicle.get_velocity(); _r = vehicle_transform.get_right_vector()
                parts.append([(_v.x * _r.x + _v.y * _r.y) / 5.0])
            elif name == "target_speed_ahead":
                parts.append(self.speed_profile.ahead(self._route_s))
        vector_obs = np.concatenate(parts).astype(np.float32)

        if self.current_step % 10000 == 0:
            self.logger.info(
                f"OBSERVATION (Step {self.current_step}):\n"
                f"   Vehicle State (Raw):\n"
                f"      Velocity: {vehicle_state['velocity']:.2f} m/s ({vehicle_state['velocity']*3.6:.1f} km/h)\n"
                f"      Lateral deviation: {vehicle_state['lateral_deviation']:.3f} m\n"
                f"      Heading error: {np.degrees(vehicle_state['heading_error']):.2f}° ({vehicle_state['heading_error']:.3f} rad)\n"
                f"   Waypoints (Raw, vehicle frame):\n"
                f"      Total waypoints: {len(next_waypoints)}\n"
                f"      First 3 waypoints: {next_waypoints[:3].tolist() if len(next_waypoints) > 0 else 'None'}\n"
                f"      Lookahead distance: {lookahead_distance:.1f} m\n"
                f"   Normalized Vector Features (passed to TD3/CNN):\n"
                f"      Velocity (normalized): {velocity_normalized:.4f} (÷30.0)\n"
                f"      Lateral dev (normalized): {lateral_deviation_normalized:.4f} (÷3.5)\n"
                f"      Heading err (normalized): {heading_error_normalized:.4f} (÷π)\n"
                f"      Waypoints (normalized): shape={waypoints_normalized.shape}, range=[{waypoints_normalized.min():.3f}, {waypoints_normalized.max():.3f}] (÷{lookahead_distance})\n"
                f"   Final Observation Shapes:\n"
                f"      Image: {image_obs.shape} (dtype={image_obs.dtype}, range=[{image_obs.min():.2f}, {image_obs.max():.2f}])\n"
                f"      Vector: {vector_obs.shape} (dtype={vector_obs.dtype}, sum={vector_obs.sum():.3f})"
            )

        return {
            "image": image_obs,
            "vector": vector_obs,
        }

    def _get_vehicle_state(self) -> Dict[str, float]:
        velocity_vec = self.vehicle.get_velocity()
        velocity = np.sqrt(
            velocity_vec.x**2 + velocity_vec.y**2 + velocity_vec.z**2
        )

        accel_vec = self.vehicle.get_acceleration()
        acceleration = np.sqrt(
            accel_vec.x**2 + accel_vec.y**2 + accel_vec.z**2
        )

        angular_vel = self.vehicle.get_angular_velocity()
        omega_z_rad = np.radians(angular_vel.z)
        acceleration_lateral = abs(velocity * omega_z_rad) if velocity > 0.1 else 0.0

        location = self.vehicle.get_location()
        heading = self.vehicle.get_transform().rotation.yaw

        if self._lateral_reference == "route":
            lateral_deviation = self.waypoint_manager.get_signed_route_offset(location)
        else:
            lateral_deviation = self.waypoint_manager.get_lateral_deviation(location)
        target_heading = self.waypoint_manager.get_target_heading(location)
        heading_error = np.arctan2(
            np.sin(np.radians(heading) - target_heading),
            np.cos(np.radians(heading) - target_heading),
        )

        carla_map = self.world.get_map()
        waypoint = carla_map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        if waypoint is not None:
            lane_half_width = waypoint.lane_width / 2.0
        else:
            lane_half_width = self.reward_calculator.lateral_tolerance

        wrong_way_penalty = self._check_wrong_way_penalty(velocity)
        wrong_way = wrong_way_penalty != 0.0

        return {
            "velocity": velocity,
            "acceleration": acceleration,
            "acceleration_lateral": acceleration_lateral,
            "lateral_deviation": lateral_deviation,
            "heading_error": float(heading_error),
            "wrong_way": wrong_way,
            "wrong_way_penalty": wrong_way_penalty,
            "lane_half_width": lane_half_width,
            "dt": self.fixed_delta_seconds,
        }

    def _check_wrong_way_penalty(self, velocity: float) -> float:
        if not hasattr(self, 'waypoint_manager') or self.waypoint_manager is None:
            return 0.0

        waypoints = self.waypoint_manager.waypoints
        if waypoints is None or len(waypoints) < 2:
            return 0.0

        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_yaw = vehicle_transform.rotation.yaw

        route_direction_rad = self.waypoint_manager.get_target_heading(vehicle_location)
        route_direction_deg = np.degrees(route_direction_rad)

        heading_error = vehicle_yaw - route_direction_deg

        while heading_error > 180.0:
            heading_error -= 360.0
        while heading_error < -180.0:
            heading_error += 360.0

        abs_heading_error = abs(heading_error)

        if abs_heading_error > 90.0:
            severity = (abs_heading_error - 90.0) / 90.0

            base_penalty = -1.0 - severity * 4.0

            velocity_scale = min(velocity / 2.0, 1.0)

            penalty = base_penalty * velocity_scale
            penalty = max(penalty, -5.0)

            self.logger.warning(
                f"[WRONG-WAY] Heading error: {heading_error:.1f}° "
                f"(vehicle: {vehicle_yaw:.1f}°, route: {route_direction_deg:.1f}°), "
                f"Velocity: {velocity:.2f} m/s, "
                f"Severity: {severity:.2f}, "
                f"Penalty: {penalty:.2f}"
            )

            return penalty

        return 0.0

    def _route_heading_from_waypoints(self, vehicle_location) -> float:
        try:
            wm = self.waypoint_manager
            dense = getattr(wm, "dense_waypoints", None)
            if dense is None or len(dense) < 2:
                return 0.0
            i = int(getattr(wm, "dense_idx", 0))
            j = min(i + 200, len(dense) - 1)
            if j <= i:
                i, j = max(0, len(dense) - 2), len(dense) - 1
            a, b = dense[i], dense[j]
            ax, ay = (a.x, a.y) if hasattr(a, "x") else (a[0], a[1])
            bx, by = (b.x, b.y) if hasattr(b, "x") else (b[0], b[1])
            return float(math.atan2(by - ay, bx - ax))
        except Exception:
            return 0.0

    def _lead_vehicle(self):
        h = self._hazard_target
        if h is None or self.vehicle is None or not self.npcs:
            return None
        ego = self.vehicle.get_location()
        r2 = (h["horizon_m"] + 10.0) ** 2
        others = []
        for npc in self.npcs:
            try:
                if not npc.is_alive:
                    continue
                loc = npc.get_location()
                if (loc.x - ego.x) ** 2 + (loc.y - ego.y) ** 2 > r2:
                    continue
                vel = npc.get_velocity()
                others.append((loc.x, loc.y, vel.x, vel.y))
            except RuntimeError:
                continue
        if not others:
            return None
        return self.waypoint_manager.lead_vehicle_ahead(others, h["horizon_m"], h["corridor_m"])

    def _choose_start_index(self, options=None) -> int:
        override = (options or {}).get("random_start", None)
        cfg = (self.training_config.get("training", {}) or {}).get("random_start", {}) or {}
        enabled = cfg.get("enabled", False) if override is None else bool(override)
        if not enabled:
            return 0
        n = len(self.waypoint_manager.waypoints)
        min_remaining = int(cfg.get("min_remaining_waypoints", 15))
        hi = max(0, n - 1 - min_remaining)
        if hi <= 0:
            return 0
        idx = int(np.random.randint(0, hi + 1))
        self.logger.info(f"[RANDOM-START] episode begins at waypoint {idx}/{n-1} "
                         f"({100.0*idx/max(n-1,1):.1f}% along the route)")
        return idx

    def _is_offroad(self, vehicle_state: Dict) -> bool:
        return abs(vehicle_state.get("lateral_deviation", 0.0)) > OFFROAD_LATERAL_THRESHOLD_M

    def _check_termination(self, vehicle_state: Dict) -> Tuple[bool, str]:
        if self.sensors.is_collision_detected():
            self.logger.warning(f"[TERMINATION] Collision detected at step {self.current_step}")
            return True, "collision"

        if self._is_offroad(vehicle_state):
            lateral_deviation = abs(vehicle_state.get("lateral_deviation", 0.0))
            self.logger.warning(
                f"[TERMINATION] Off-road at step {self.current_step}: "
                f"lateral_deviation={lateral_deviation:.3f}m > "
                f"{OFFROAD_LATERAL_THRESHOLD_M}m threshold"
            )
            return True, "off_road"


        if self.waypoint_manager.is_route_finished():
            self.logger.info(
                f"[TERMINATION] Route completed at step {self.current_step}! "
                f"Dense waypoint {self.waypoint_manager.get_current_waypoint_index()}/{len(self.waypoint_manager.dense_waypoints)-1}"
            )
            return True, "route_completed"


        return False, "running"

    def _spawn_npc_traffic(self):
        scenarios = self.training_config.get("scenarios", [])

        self.logger.debug(f"training_config keys: {self.training_config.keys()}")
        self.logger.debug(f"scenarios type: {type(scenarios)}, length: {len(scenarios) if isinstance(scenarios, list) else 'N/A'}")

        scenario_idx = int(os.getenv('CARLA_SCENARIO_INDEX', '0'))
        self.logger.debug(f"CARLA_SCENARIO_INDEX from environment: {scenario_idx}")

        if isinstance(scenarios, list) and len(scenarios) > 0:
            scenario_idx = min(scenario_idx, len(scenarios) - 1)
            scenario = scenarios[scenario_idx]
            npc_count = scenario.get("num_vehicles", 50)
            self.logger.info(f"Using scenario: {scenario.get('name', 'unknown')} (index {scenario_idx})")
        else:
            npc_count = 50
            self.logger.warning(f"No scenarios found in config, using default NPC count: {npc_count}")

        self.logger.info(f"Spawning {npc_count} NPC vehicles...")

        try:
            if self.tm_port is None:
                self.traffic_manager = self.client.get_trafficmanager()
                self.logger.info("Using default Traffic Manager port (8000)")
            else:
                self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
                self.logger.info(f"Using custom Traffic Manager port ({self.tm_port})")

            self.traffic_manager.set_synchronous_mode(True)

            seed = self.training_config.get("seed", 42)
            self.traffic_manager.set_random_device_seed(seed)
            tm_port = self.traffic_manager.get_port()
            self.logger.info(f"Traffic Manager configured: synchronous=True, seed={seed}, port={tm_port}")

            vehicle_bp = self.world.get_blueprint_library().filter("vehicle")
            spawn_points = np.random.choice(
                self.spawn_points, min(npc_count, len(self.spawn_points)), replace=False
            )

            spawn_attempts = 0
            spawn_successes = 0
            ego_location = self.vehicle.get_location()
            if self._fix_npc_exclusion and self._ego_spawn_location is not None:
                ego_location = self._ego_spawn_location

            for spawn_point in spawn_points:
                if spawn_point.location.distance(ego_location) < 20.0:
                    continue

                spawn_attempts += 1
                try:
                    npc = self.world.spawn_actor(
                        np.random.choice(vehicle_bp), spawn_point
                    )

                    npc.set_autopilot(True, tm_port)

                    self.traffic_manager.update_vehicle_lights(npc, True)
                    self.traffic_manager.auto_lane_change(npc, True)
                    self.npcs.append(npc)
                    spawn_successes += 1

                except RuntimeError as e:
                    if "collision" in str(e).lower():
                        self.logger.debug(f"Spawn collision at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
                    else:
                        self.logger.warning(f"NPC spawn failed: {e}")

            success_rate = spawn_successes / spawn_attempts if spawn_attempts > 0 else 0
            self.logger.info(f"NPC spawning complete: {spawn_successes}/{spawn_attempts} successful ({success_rate*100:.1f}%)")

            if success_rate < 0.8 and spawn_attempts > 0:
                self.logger.warning(f"Low NPC spawn success rate: {success_rate*100:.1f}% (target: 80%)")

        except Exception as e:
            self.logger.warning(f"NPC traffic spawning failed: {e}")

    def _cleanup_episode(self):
        cleanup_errors = []

        if self.sensors:
            try:
                self.logger.debug("Destroying sensor suite...")
                success = self.sensors.destroy()
                self.sensors = None
                self.logger.debug("Sensor suite destroyed successfully")

            except Exception as e:
                cleanup_errors.append(f"Sensor cleanup error: {e}")
                self.logger.error(f"Error during sensor cleanup: {e}", exc_info=True)
                self.sensors = None

        if self.vehicle:
            try:
                self.logger.debug("Destroying ego vehicle...")

                if not hasattr(self.vehicle, 'is_alive') or not self.vehicle.is_alive:
                    self.logger.warning("Ego vehicle already destroyed, skipping cleanup")
                    self.vehicle = None
                else:
                    success = self.vehicle.destroy()
                    if success:
                        self.logger.debug("Ego vehicle destroyed successfully")
                    else:
                        cleanup_errors.append("Ego vehicle destruction returned False")
                        self.logger.warning("Ego vehicle destruction failed")
                    self.vehicle = None

            except RuntimeError as e:
                cleanup_errors.append(f"Vehicle cleanup RuntimeError: {e}")
                self.logger.warning(f"Vehicle destruction RuntimeError: {e}")
                self.vehicle = None
            except AttributeError as e:
                cleanup_errors.append(f"Vehicle attribute error: {e}")
                self.logger.warning(f"Vehicle attribute error during cleanup: {e}")
                self.vehicle = None
            except Exception as e:
                cleanup_errors.append(f"Vehicle cleanup error: {e}")
                self.logger.error(f"Unexpected error during vehicle cleanup: {e}", exc_info=True)
                self.vehicle = None

        npc_failures = 0
        for i, npc in enumerate(self.npcs):
            try:
                if not hasattr(npc, 'is_alive') or not npc.is_alive:
                    self.logger.debug(f"NPC {i} already destroyed")
                    continue

                success = npc.destroy()
                if not success:
                    npc_failures += 1
                    self.logger.debug(f"NPC {i} destruction returned False")

            except RuntimeError as e:
                npc_failures += 1
                self.logger.debug(f"NPC {i} RuntimeError during destruction: {e}")
            except AttributeError as e:
                npc_failures += 1
                self.logger.debug(f"NPC {i} AttributeError during destruction: {e}")
            except Exception as e:
                npc_failures += 1
                self.logger.debug(f"Failed to destroy NPC {i}: {e}")

        if npc_failures > 0:
            self.logger.debug(f"{npc_failures}/{len(self.npcs)} NPCs failed to destroy")

        self.npcs = []

        if cleanup_errors:
            error_msg = f"Critical cleanup issues encountered: {cleanup_errors}"
            self.logger.warning(error_msg)
        else:
            self.logger.debug("Episode cleanup completed successfully")

    def close(self):
        if getattr(self, '_closed', False):
            self.logger.debug("Environment already closed, skipping cleanup")
            return

        self.logger.info("Closing CARLA environment...")

        if self.ros_interface is not None:
            try:
                self.logger.info("[ROS BRIDGE] Closing ROS interface...")
                self.ros_interface.destroy()
                self.ros_interface = None
            except Exception as e:
                self.logger.warning(f"[ROS BRIDGE] Error closing ROS interface: {e}")

        self._cleanup_episode()

        if self.world:
            try:
                self.world.tick()
                time.sleep(0.02)
                self.logger.debug("Final world tick completed, callbacks flushed")
            except Exception as e:
                self.logger.warning(f"Final world tick failed: {e}")

        if self.traffic_manager:
            try:
                self.traffic_manager.set_synchronous_mode(False)
                self.logger.debug("Traffic Manager synchronous mode disabled")
            except Exception as e:
                self.logger.warning(f"Failed to disable TM sync mode: {e}")
            finally:
                self.traffic_manager = None

        if self.world and hasattr(self, '_original_settings'):
            try:
                self.world.apply_settings(self._original_settings)
                self.logger.debug("World settings restored to original state")
            except Exception as e:
                self.logger.warning(f"Failed to restore world settings: {e}")

        if self.client:
            self.client = None

        self._closed = True
        self.logger.info("CARLA environment closed")

    @property
    def is_closed(self):
        return getattr(self, '_closed', False)

    def render(self, mode: str = "human"):
        pass

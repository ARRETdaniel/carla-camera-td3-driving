"""Sensor Management for CARLA Autonomous Vehicle"""

import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict
import logging
import queue
import threading

import carla


class CARLACameraManager:

    def __init__(
        self,
        vehicle: carla.Actor,
        camera_config: Dict,
        world: carla.World,
    ):
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        self.camera_config = camera_config

        self.target_width = camera_config.get("target_size", 84)
        self.target_height = self.target_width

        self.width = camera_config.get("image_size_x", 336)
        self.height = camera_config.get("image_size_y", 336)
        self.fov = camera_config.get("fov", 90)
        if self.width != self.height:
            self.logger.warning(
                f"Camera capture is {self.width}x{self.height} (aspect "
                f"{self.width/self.height:.3f}) but the network input is square. "
                f"The resize will distort angles by "
                f"{(self.width/self.target_width)/(self.height/self.target_height):.2f}x. "
                "Set image_size_x == image_size_y in carla_config.yaml."
            )

        self.image_queue = queue.Queue()
        self.latest_image = None
        self.latest_rgb_image = None
        self.last_frame_id = None
        self.image_lock = threading.Lock()

        self.frame_step_counter = 0
        self.log_frequency = 100

        self.camera_sensor = None
        self._setup_camera()

        self.logger.info(
            f"Camera initialized: {self.width}×{self.height}, FOV={self.fov}°, "
            f"target size {self.target_width}×{self.target_height}"
        )

    def _setup_camera(self):
        camera_bp = self.world.get_blueprint_library().find("sensor.camera.rgb")

        camera_bp.set_attribute("image_size_x", str(self.width))
        camera_bp.set_attribute("image_size_y", str(self.height))
        camera_bp.set_attribute("fov", str(self.fov))

        tf = self.camera_config.get("transform", {})
        spawn_point = carla.Transform(
            carla.Location(
                x=tf.get("x", 0.8),
                y=tf.get("y", 0.0),
                z=tf.get("z", 1.7),
            ),
            carla.Rotation(
                pitch=tf.get("pitch", 0.0),
                yaw=tf.get("yaw", 0.0),
                roll=tf.get("roll", 0.0),
            ),
        )
        if "sensor_tick" in self.camera_config:
            camera_bp.set_attribute("sensor_tick", str(self.camera_config["sensor_tick"]))

        self.camera_sensor = self.world.spawn_actor(
            camera_bp, spawn_point, attach_to=self.vehicle
        )

        self.camera_sensor.listen(self._on_camera_frame)

    def _on_camera_frame(self, image: carla.Image):
        try:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            with self.image_lock:
                self.latest_rgb_image = array.copy()

            processed = self._preprocess(array)

            with self.image_lock:
                self.latest_image = processed

            self.image_queue.put((image.frame, processed))

        except Exception as e:
            self.logger.error(f"Error processing camera frame: {e}")

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        self.frame_step_counter += 1
        should_log = (self.frame_step_counter % self.log_frequency == 0)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   PREPROCESSING INPUT (Frame #{self.frame_step_counter}):\n"
                f"   Shape: {image.shape}\n"
                f"   Dtype: {image.dtype}\n"
                f"   Range: [{image.min():.2f}, {image.max():.2f}]\n"
                f"   Mean: {image.mean():.2f}, Std: {image.std():.2f}"
            )

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        resized = cv2.resize(
            gray, (self.target_width, self.target_height), interpolation=cv2.INTER_AREA
        )

        scaled = resized.astype(np.float32) / 255.0

        mean, std = 0.5, 0.5
        normalized = (scaled - mean) / std

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   PREPROCESSING OUTPUT:\n"
                f"   Shape: {normalized.shape}\n"
                f"   Dtype: {normalized.dtype}\n"
                f"   Range: [{normalized.min():.3f}, {normalized.max():.3f}]\n"
                f"   Mean: {normalized.mean():.3f}, Std: {normalized.std():.3f}\n"
                f"   Expected range: [-1, 1]"
            )

        return normalized

    def wait_for_frame(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        try:
            frame_id, frame = self.image_queue.get(timeout=timeout)
        except queue.Empty:
            self.logger.warning(
                f"No camera frame within {timeout}s. The frame stack is UNCHANGED - which "
                "immediately after a reset means the observation is four zero frames. "
                "Check that the server is in synchronous mode and that world.tick() is "
                "being called."
            )
            return None
        while True:
            try:
                frame_id, frame = self.image_queue.get_nowait()
            except queue.Empty:
                break
        self.last_frame_id = frame_id
        return frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self.image_lock:
            return self.latest_image

    def get_latest_rgb_frame(self) -> Optional[np.ndarray]:
        with self.image_lock:
            return self.latest_rgb_image

    def destroy(self):
        if self.camera_sensor is None:
            return

        try:
            if not hasattr(self.camera_sensor, 'is_alive') or not self.camera_sensor.is_alive:
                self.logger.warning("Camera sensor already destroyed, skipping cleanup")
                self.camera_sensor = None
                return

            if hasattr(self.camera_sensor, 'is_listening') and self.camera_sensor.is_listening:
                self.camera_sensor.stop()
                self.logger.debug("Camera sensor stopped listening")

            import time
            time.sleep(0.01)

            if self.camera_sensor.is_alive:
                success = self.camera_sensor.destroy()
                if success:
                    self.logger.info("Camera sensor destroyed")
                else:
                    self.logger.warning("Camera sensor destroy returned False")
            else:
                self.logger.warning("Camera sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Camera sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Camera sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying camera sensor: {e}", exc_info=True)
        finally:
            self.camera_sensor = None


class ImageStack:

    def __init__(self, frame_height: int = 84, frame_width: int = 84, num_frames: int = 4):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.num_frames = num_frames
        self.logger = logging.getLogger(__name__)

        self.push_step_counter = 0
        self.log_frequency = 100

        self.stack = deque(
            [
                np.zeros((frame_height, frame_width), dtype=np.float32)
                for _ in range(num_frames)
            ],
            maxlen=num_frames,
        )

    def push_frame(self, frame: np.ndarray):
        if frame.shape != (self.frame_height, self.frame_width):
            raise ValueError(
                f"Frame shape {frame.shape} != expected "
                f"({self.frame_height}, {self.frame_width})"
            )

        self.push_step_counter += 1
        should_log = (self.push_step_counter % self.log_frequency == 0)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            self.logger.debug(
                f"   FRAME STACKING (Push #{self.push_step_counter}):\n"
                f"   New frame shape: {frame.shape}\n"
                f"   New frame range: [{frame.min():.3f}, {frame.max():.3f}]\n"
                f"   Stack size before: {len(self.stack)}\n"
                f"   Stack filled: {self.is_filled()}"
            )

        self.stack.append(frame)

        if self.logger.isEnabledFor(logging.DEBUG) and should_log:
            stacked = self.get_stacked_frames()
            self.logger.debug(
                f"  STACK STATE AFTER PUSH:\n"
                f"   Stack size: {len(self.stack)}\n"
                f"   Stacked shape: {stacked.shape}\n"
                f"   Stacked range: [{stacked.min():.3f}, {stacked.max():.3f}]\n"
                f"   Per-frame ranges: {[f'[{frame.min():.3f}, {frame.max():.3f}]' for frame in self.stack]}"
            )

    def get_stacked_frames(self) -> np.ndarray:
        return np.array(list(self.stack), dtype=np.float32)

    def reset(self):
        self.stack.clear()
        for _ in range(self.num_frames):
            self.stack.append(
                np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
            )

    def is_filled(self) -> bool:
        return len(self.stack) == self.num_frames


class CollisionDetector:

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        self.collision_detected = False
        self.collision_event = None
        self.collision_impulse = 0.0
        self.collision_force = 0.0
        self.collision_lock = threading.Lock()

        self.step_collision_count = 0

        self.collision_sensor = None
        self._setup_collision_sensor()

    def _setup_collision_sensor(self):
        collision_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        spawn_point = carla.Transform(carla.Location())

        self.collision_sensor = self.world.spawn_actor(
            collision_bp, spawn_point, attach_to=self.vehicle
        )
        self.collision_sensor.listen(self._on_collision)
        self.logger.info("Collision sensor initialized")

    def _on_collision(self, event: carla.CollisionEvent):
        with self.collision_lock:
            self.collision_detected = True
            self.collision_event = event

            impulse_vector = event.normal_impulse
            self.collision_impulse = impulse_vector.length()

            collision_duration = 0.1
            self.collision_force = self.collision_impulse / collision_duration

            self.step_collision_count = 1

        self.logger.warning(
            f"Collision detected with {event.other_actor.type_id} "
            f"(impulse: {self.collision_impulse:.1f} N·s, force: ~{self.collision_force:.1f} N)"
        )

    def is_colliding(self) -> bool:
        with self.collision_lock:
            return self.collision_detected

    def get_collision_info(self) -> Optional[Dict]:
        with self.collision_lock:
            if self.collision_detected and self.collision_event:
                return {
                    "other_actor": str(self.collision_event.other_actor.type_id),
                    "impulse": self.collision_impulse,
                    "force": self.collision_force,
                }
            return None

    def get_step_collision_count(self) -> int:
        with self.collision_lock:
            return self.step_collision_count

    def reset(self):
        with self.collision_lock:
            self.collision_detected = False
            self.collision_event = None
            self.collision_impulse = 0.0
            self.collision_force = 0.0
            self.step_collision_count = 0

    def reset_step_counter(self):
        with self.collision_lock:
            self.step_collision_count = 0

    def destroy(self):
        if self.collision_sensor is None:
            return

        try:
            if not hasattr(self.collision_sensor, 'is_alive') or not self.collision_sensor.is_alive:
                self.logger.warning("Collision sensor already destroyed, skipping cleanup")
                self.collision_sensor = None
                return

            if hasattr(self.collision_sensor, 'is_listening') and self.collision_sensor.is_listening:
                self.collision_sensor.stop()
                self.logger.debug("Collision sensor stopped listening")

            import time
            time.sleep(0.01)

            if self.collision_sensor.is_alive:
                success = self.collision_sensor.destroy()
                if success:
                    self.logger.info("Collision sensor destroyed")
                else:
                    self.logger.warning("Collision sensor destroy returned False")
            else:
                self.logger.warning("Collision sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Collision sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Collision sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying collision sensor: {e}", exc_info=True)
        finally:
            self.collision_sensor = None


class LaneInvasionDetector:

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        self.lane_invaded = False
        self.invasion_event = None
        self.invasion_lock = threading.Lock()

        self.step_invasion_count = 0

        self.lane_sensor = None
        self._setup_lane_sensor()

    def _setup_lane_sensor(self):
        lane_bp = self.world.get_blueprint_library().find("sensor.other.lane_invasion")
        spawn_point = carla.Transform(carla.Location())

        self.lane_sensor = self.world.spawn_actor(
            lane_bp, spawn_point, attach_to=self.vehicle
        )
        self.lane_sensor.listen(self._on_lane_invasion)
        self.logger.info("Lane invasion sensor initialized")

    def _on_lane_invasion(self, event: carla.LaneInvasionEvent):
        with self.invasion_lock:
            self.lane_invaded = True
            self.invasion_event = event

            self.step_invasion_count = 1

        self.logger.warning(f"Lane invasion detected: {event.crossed_lane_markings}")

    def is_invading_lane(self, lateral_deviation: float = None, lane_half_width: float = None) -> bool:
        with self.invasion_lock:
            if lateral_deviation is not None and lane_half_width is not None:
                recovery_threshold = lane_half_width * 0.8

                if abs(lateral_deviation) < recovery_threshold:
                    if self.lane_invaded:
                        self.logger.info(
                            f"[LANE RECOVERY] Vehicle returned to lane center "
                            f"(deviation={lateral_deviation:.3f}m < threshold={recovery_threshold:.3f}m)"
                        )
                    self.lane_invaded = False
                    self.invasion_event = None

            return self.lane_invaded

    def get_step_invasion_count(self) -> int:
        with self.invasion_lock:
            return self.step_invasion_count

    def reset(self):
        with self.invasion_lock:
            self.lane_invaded = False
            self.invasion_event = None
            self.step_invasion_count = 0

    def reset_step_counter(self):
        with self.invasion_lock:
            self.step_invasion_count = 0

    def destroy(self):
        if self.lane_sensor is None:
            return

        try:
            if not hasattr(self.lane_sensor, 'is_alive') or not self.lane_sensor.is_alive:
                self.logger.warning("Lane invasion sensor already destroyed, skipping cleanup")
                self.lane_sensor = None
                return

            if hasattr(self.lane_sensor, 'is_listening') and self.lane_sensor.is_listening:
                self.lane_sensor.stop()
                self.logger.debug("Lane invasion sensor stopped listening")

            import time
            time.sleep(0.01)

            if self.lane_sensor.is_alive:
                success = self.lane_sensor.destroy()
                if success:
                    self.logger.info("Lane invasion sensor destroyed")
                else:
                    self.logger.warning("Lane invasion sensor destroy returned False")
            else:
                self.logger.warning("Lane invasion sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Lane invasion sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Lane invasion sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying lane invasion sensor: {e}", exc_info=True)
        finally:
            self.lane_sensor = None


class ObstacleDetector:

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        self.distance_to_obstacle = float('inf')
        self.other_actor = None
        self.last_detection_time = None
        self.obstacle_lock = threading.Lock()

        self.obstacle_sensor = None
        self._setup_obstacle_sensor()

    def _setup_obstacle_sensor(self):
        obstacle_bp = self.world.get_blueprint_library().find("sensor.other.obstacle")

        obstacle_bp.set_attribute("distance", "10.0")
        obstacle_bp.set_attribute("hit_radius", "0.5")
        obstacle_bp.set_attribute("only_dynamics", "False")
        obstacle_bp.set_attribute("debug_linetrace", "False")
        obstacle_bp.set_attribute("sensor_tick", "0.0")

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        self.obstacle_sensor = self.world.spawn_actor(
            obstacle_bp, spawn_point, attach_to=self.vehicle
        )
        self.obstacle_sensor.listen(self._on_obstacle_detection)
        self.logger.info("Obstacle detector initialized (distance=10m, hit_radius=0.5m)")

    def _on_obstacle_detection(self, event: carla.ObstacleDetectionEvent):
        import time

        with self.obstacle_lock:
            self.distance_to_obstacle = event.distance
            self.other_actor = event.other_actor
            self.last_detection_time = time.time()

        if event.distance < 5.0:
            self.logger.debug(
                f"Obstacle detected: {event.other_actor.type_id} at {event.distance:.2f}m"
            )

    def get_distance_to_nearest_obstacle(self) -> float:
        import time

        with self.obstacle_lock:
            if self.last_detection_time is not None:
                time_since_detection = time.time() - self.last_detection_time

                if time_since_detection > 0.2:
                    if self.distance_to_obstacle < float('inf'):
                        self.logger.debug(
                            f"[OBSTACLE CLEAR] No detection for {time_since_detection:.3f}s, "
                            f"clearing cached distance {self.distance_to_obstacle:.2f}m"
                        )
                    self.distance_to_obstacle = float('inf')
                    self.other_actor = None
                    self.last_detection_time = None

            return self.distance_to_obstacle

    def get_obstacle_info(self) -> Optional[Dict]:
        with self.obstacle_lock:
            if self.distance_to_obstacle < float('inf') and self.other_actor:
                return {
                    "distance": self.distance_to_obstacle,
                    "actor_type": str(self.other_actor.type_id),
                }
            return None

    def reset(self):
        with self.obstacle_lock:
            self.distance_to_obstacle = float('inf')
            self.other_actor = None
            self.last_detection_time = None

    def destroy(self):
        if self.obstacle_sensor is None:
            return

        try:
            if not hasattr(self.obstacle_sensor, 'is_alive') or not self.obstacle_sensor.is_alive:
                self.logger.warning("Obstacle sensor already destroyed, skipping cleanup")
                self.obstacle_sensor = None
                return

            if hasattr(self.obstacle_sensor, 'is_listening') and self.obstacle_sensor.is_listening:
                self.obstacle_sensor.stop()
                self.logger.debug("Obstacle sensor stopped listening")

            import time
            time.sleep(0.01)

            if self.obstacle_sensor.is_alive:
                success = self.obstacle_sensor.destroy()
                if success:
                    self.logger.info("Obstacle sensor destroyed")
                else:
                    self.logger.warning("Obstacle sensor destroy returned False")
            else:
                self.logger.warning("Obstacle sensor became invalid before destroy call")

        except RuntimeError as e:
            self.logger.warning(f"Obstacle sensor destruction RuntimeError: {e}")
        except AttributeError as e:
            self.logger.warning(f"Obstacle sensor attribute error during cleanup: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error destroying obstacle sensor: {e}", exc_info=True)
        finally:
            self.obstacle_sensor = None


class OffroadDetector:

    def __init__(self, vehicle: carla.Actor, world: carla.World):
        self.vehicle = vehicle
        self.world = world
        self.carla_map = world.get_map()
        self.logger = logging.getLogger(__name__)
        self.is_offroad = False
        self.offroad_lock = threading.Lock()

        self.logger.info("Offroad detector initialized using Waypoint API")

    def check_offroad(self) -> bool:
        with self.offroad_lock:
            try:
                location = self.vehicle.get_location()

                waypoint = self.carla_map.get_waypoint(
                    location,
                    project_to_road=False,
                    lane_type=carla.LaneType.Any
                )

                if waypoint is None:
                    self.is_offroad = True
                    self.logger.warning("[OFFROAD] Vehicle off map (no waypoint)")
                    return True

                drivable_lane_types = [
                    carla.LaneType.Driving,
                    carla.LaneType.Parking,
                    carla.LaneType.Bidirectional,
                ]

                if waypoint.lane_type not in drivable_lane_types:
                    self.is_offroad = True
                    lane_type_name = str(waypoint.lane_type).replace("LaneType.", "")
                    self.logger.warning(
                        f"[OFFROAD] Vehicle on {lane_type_name} (not drivable)"
                    )
                    return True

                self.is_offroad = False
                return False

            except Exception as e:
                self.logger.error(f"Error checking offroad status: {e}", exc_info=True)
                return False

    def reset(self):
        with self.offroad_lock:
            self.is_offroad = False
            self.logger.debug("Offroad detector state reset")


class SensorSuite:

    def __init__(
        self,
        vehicle: carla.Actor,
        carla_config: Dict,
        world: carla.World,
    ):
        self.vehicle = vehicle
        self.world = world
        self.logger = logging.getLogger(__name__)

        camera_config = carla_config.get("sensors", {}).get("camera_rgb", {})

        self.camera = CARLACameraManager(vehicle, camera_config, world)
        self.collision_detector = CollisionDetector(vehicle, world)
        self.lane_invasion_detector = LaneInvasionDetector(vehicle, world)
        self.obstacle_detector = ObstacleDetector(vehicle, world)
        self.offroad_detector = OffroadDetector(vehicle, world)

        self.image_stack = ImageStack(
            frame_height=84, frame_width=84, num_frames=4
        )

        self.logger.info("Complete sensor suite initialized (including offroad detector)")

    def tick(self):
        frame = self.camera.wait_for_frame()
        if frame is not None:
            self.image_stack.push_frame(frame)

    def get_camera_data(self) -> np.ndarray:
        stacked = self.image_stack.get_stacked_frames()

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"   FINAL CAMERA OBSERVATION:\n"
                f"   Shape: {stacked.shape}\n"
                f"   Dtype: {stacked.dtype}\n"
                f"   Range: [{stacked.min():.3f}, {stacked.max():.3f}]\n"
                f"   Mean: {stacked.mean():.3f}, Std: {stacked.std():.3f}\n"
                f"   Non-zero frames: {(stacked != 0).any(axis=(1,2)).sum()}/4\n"
                f"   Ready for CNN input (batch, 4, 84, 84)"
            )

        return stacked

    def get_rgb_camera_frame(self) -> Optional[np.ndarray]:
        return self.camera.get_latest_rgb_frame()

    def is_collision_detected(self) -> bool:
        return self.collision_detector.is_colliding()

    def is_lane_invaded(self, lateral_deviation: float = None, lane_half_width: float = None) -> bool:
        return self.lane_invasion_detector.is_invading_lane(lateral_deviation, lane_half_width)

    def get_collision_info(self) -> Optional[Dict]:
        return self.collision_detector.get_collision_info()

    def get_step_collision_count(self) -> int:
        return self.collision_detector.get_step_collision_count()

    def get_step_lane_invasion_count(self) -> int:
        return self.lane_invasion_detector.get_step_invasion_count()

    def get_distance_to_nearest_obstacle(self) -> float:
        return self.obstacle_detector.get_distance_to_nearest_obstacle()

    def is_offroad(self) -> bool:
        return self.offroad_detector.check_offroad()

    def get_obstacle_info(self) -> Optional[Dict]:
        return self.obstacle_detector.get_obstacle_info()

    def reset(self):
        self.image_stack.reset()
        self.collision_detector.reset()
        self.lane_invasion_detector.reset()
        self.obstacle_detector.reset()
        self.offroad_detector.reset()
        self.logger.info("All sensors reset (including offroad detector)")

    def reset_step_counters(self):
        self.collision_detector.reset_step_counter()
        self.lane_invasion_detector.reset_step_counter()

    def destroy(self):
        errors = []

        try:
            self.camera.destroy()
        except Exception as e:
            errors.append(f"Camera: {e}")
            self.logger.warning(f"Error destroying camera: {e}")

        try:
            self.collision_detector.destroy()
        except Exception as e:
            errors.append(f"Collision: {e}")
            self.logger.warning(f"Error destroying collision detector: {e}")

        try:
            self.lane_invasion_detector.destroy()
        except Exception as e:
            errors.append(f"Lane: {e}")
            self.logger.warning(f"Error destroying lane invasion detector: {e}")

        try:
            self.obstacle_detector.destroy()
        except Exception as e:
            errors.append(f"Obstacle: {e}")
            self.logger.warning(f"Error destroying obstacle detector: {e}")

        if errors:
            self.logger.warning(f"Sensor cleanup completed with {len(errors)} error(s)")
        else:
            self.logger.info("Sensor suite destroyed successfully")

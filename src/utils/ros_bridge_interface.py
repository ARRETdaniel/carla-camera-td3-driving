#!/usr/bin/env python3
"""ROS 2 Bridge Interface for CARLA Vehicle Control"""

import logging
from typing import Optional
import carla

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist, Vector3
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    logging.warning("[ROS BRIDGE] rclpy not available - ROS 2 control disabled")


class ROSBridgeInterface:

    MAX_LON_ACCELERATION = 10.0

    DEFAULT_MAX_STEER_ANGLE = 1.22

    def __init__(
        self,
        node_name: str = 'carla_vehicle_controller',
        role_name: str = 'ego_vehicle',
        use_docker_exec: bool = False
    ):
        self.logger = logging.getLogger(self.__class__.__name__)

        self.logger.info(f"[ROS BRIDGE] Attempting to initialize ROSBridgeInterface...")
        self.logger.info(f"[ROS BRIDGE] Node name: {node_name}, Role: {role_name}")

        if not ROS2_AVAILABLE:
            error_msg = (
                "rclpy not available. "
                "Ensure ROS 2 Humble is installed and sourced: "
                "source /opt/ros/humble/setup.bash"
            )
            self.logger.error(f"[ROS BRIDGE] {error_msg}")
            raise RuntimeError(error_msg)

        self.logger.info(f"[ROS BRIDGE] rclpy module found, proceeding with initialization...")

        self.node_name = node_name
        self.role_name = role_name
        self.node: Optional[Node] = None
        self.publisher = None
        self.max_steering_angle = self.DEFAULT_MAX_STEER_ANGLE
        self._message_count = 0

        self.logger.info(f"[ROS BRIDGE] Calling _initialize_ros()...")
        self._initialize_ros()
        self.logger.info(f"[ROS BRIDGE] _initialize_ros() completed successfully!")

        self.logger.info(f"[ROS BRIDGE] Initialized native rclpy Twist publisher")
        self.logger.info(f"[ROS BRIDGE] Node name: {node_name}")
        self.logger.info(f"[ROS BRIDGE] Topic: /carla/{role_name}/twist")
        self.logger.info(f"[ROS BRIDGE] Using geometry_msgs/Twist (standard ROS 2 message)")

    def _initialize_ros(self):
        try:
            self.logger.info("[ROS BRIDGE] Step 1: Checking rclpy context...")
            if not rclpy.ok():
                self.logger.info("[ROS BRIDGE] Step 2: Initializing rclpy context...")
                rclpy.init()
                self.logger.info("[ROS BRIDGE] Step 3: rclpy context initialized successfully!")
            else:
                self.logger.info("[ROS BRIDGE] Step 2: rclpy already initialized, skipping init()")

            self.logger.info(f"[ROS BRIDGE] Step 4: Creating ROS 2 node '{self.node_name}'...")
            self.node = Node(self.node_name)
            self.logger.info(f"[ROS BRIDGE] Step 5: Node created successfully!")

            topic_name = f'/carla/{self.role_name}/twist'
            self.logger.info(f"[ROS BRIDGE] Step 6: Creating publisher on topic: {topic_name}")

            self.publisher = self.node.create_publisher(
                Twist,
                topic_name,
                10
            )

            self.logger.info(f"[ROS BRIDGE] Step 7: Publisher created successfully!")
            self.logger.info(f"[ROS BRIDGE] Topic: {topic_name}")
            self.logger.info(f"[ROS BRIDGE] Message type: geometry_msgs/msg/Twist")

        except Exception as e:
            self.logger.error(f"[ROS BRIDGE] FAILED to initialize ROS 2: {type(e).__name__}: {e}")
            import traceback
            self.logger.error(f"[ROS BRIDGE] Traceback:\n{traceback.format_exc()}")
            raise

    def publish_control(
        self,
        control: Optional[carla.VehicleControl] = None,
        throttle: float = 0.0,
        steer: float = 0.0,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
        gear: int = 0,
        manual_gear_shift: bool = False
    ) -> bool:
        if not self.publisher:
            self.logger.error("[ROS BRIDGE] Publisher not initialized")
            return False

        try:
            if control is not None:
                throttle_val = float(control.throttle)
                steer_val = float(control.steer)
                brake_val = float(control.brake)
                hand_brake_val = bool(control.hand_brake)
                reverse_val = bool(control.reverse)
            else:
                throttle_val = float(throttle)
                steer_val = float(steer)
                brake_val = float(brake)
                hand_brake_val = bool(hand_brake)
                reverse_val = bool(reverse)

            msg = Twist()

            if hand_brake_val or brake_val >= 0.99:
                msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
                msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
            else:
                if brake_val > 0.01:
                    linear_x = -brake_val * self.MAX_LON_ACCELERATION
                elif reverse_val:
                    linear_x = -throttle_val * self.MAX_LON_ACCELERATION
                else:
                    linear_x = throttle_val * self.MAX_LON_ACCELERATION

                angular_z = -steer_val * self.max_steering_angle

                msg.linear = Vector3(x=linear_x, y=0.0, z=0.0)
                msg.angular = Vector3(x=0.0, y=0.0, z=angular_z)

            self.publisher.publish(msg)

            rclpy.spin_once(self.node, timeout_sec=0.0)

            self._message_count += 1

            if self._message_count <= 10:
                self.logger.info(
                    f"[ROS BRIDGE] Published Twist #{self._message_count}: "
                    f"linear.x={msg.linear.x:.2f} m/s, angular.z={msg.angular.z:.3f} rad "
                    f"(throttle={throttle_val:.2f}, steer={steer_val:.3f}, brake={brake_val:.2f})"
                )
            elif self._message_count % 100 == 0:
                self.logger.debug(
                    f"[ROS BRIDGE] Published Twist #{self._message_count}: "
                    f"linear.x={msg.linear.x:.2f} m/s, angular.z={msg.angular.z:.3f} rad"
                )
            else:
                self.logger.debug(
                    f"[ROS BRIDGE] Published Twist: "
                    f"linear.x={msg.linear.x:.2f} m/s, angular.z={msg.angular.z:.3f} rad "
                    f"(throttle={throttle_val:.2f}, steer={steer_val:.3f}, brake={brake_val:.2f})"
                )

            return True

        except Exception as e:
            self.logger.error(f"[ROS BRIDGE] Failed to publish Twist: {e}")
            return False

    def wait_for_topics(self, timeout: float = 10.0) -> bool:
        import time

        if not self.node:
            self.logger.error("[ROS BRIDGE] Node not initialized")
            return False

        self.logger.info(f"[ROS BRIDGE] Waiting for Twist topic (timeout: {timeout}s)...")

        start_time = time.time()
        topic_name = f'/carla/{self.role_name}/twist'

        while (time.time() - start_time) < timeout:
            topic_names_and_types = self.node.get_topic_names_and_types()
            topic_names = [name for name, _ in topic_names_and_types]

            check_topics = [
                f'/carla/{self.role_name}/vehicle_info',
                f'/carla/{self.role_name}/vehicle_control_cmd'
            ]

            found_count = sum(1 for t in check_topics if t in topic_names)

            if found_count >= 1:
                self.logger.info(
                    f"[ROS BRIDGE] ROS Bridge topics found ({found_count}/{len(check_topics)})!"
                )
                return True

            time.sleep(0.5)
            rclpy.spin_once(self.node, timeout_sec=0.0)

        self.logger.warning(
            f"[ROS BRIDGE] Timeout waiting for ROS Bridge topics\n"
            f"[ROS BRIDGE] Expected topics: {check_topics}\n"
            f"[ROS BRIDGE] Available topics (sample): {topic_names[:10]}\n"
            f"[ROS BRIDGE] Make sure CARLA ROS Bridge is running with carla_twist_to_control!"
        )
        return False

    def destroy(self):
        try:
            if self.node:
                self.node.destroy_node()
                self.logger.info("[ROS BRIDGE] Node destroyed")


        except Exception as e:
            self.logger.warning(f"[ROS BRIDGE] Cleanup warning: {e}")

    def __del__(self):
        self.destroy()

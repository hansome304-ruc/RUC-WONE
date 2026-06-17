from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Ros2CmdVelConfig:
    topic: str = "/cmd_vel"
    node_name: str = "airbot_movebase_cmd_vel"
    publish_rate_hz: float = 20.0
    wait_subscriber_timeout_s: float = 2.0
    require_subscriber: bool = True
    max_linear_mps: float = 0.20
    max_angular_radps: float = 0.50


class Ros2CmdVelBackend:
    """Small ROS 2 geometry_msgs/Twist backend.

    Imports ROS packages lazily so the rest of the repository can be imported in
    environments that do not have ROS sourced.
    """

    def __init__(self, config: Optional[Ros2CmdVelConfig] = None) -> None:
        self.config = config or Ros2CmdVelConfig()
        self._last_velocity = (0.0, 0.0, 0.0)
        self._closed = False

        try:
            import rclpy
            from geometry_msgs.msg import Twist
        except ImportError as exc:
            raise RuntimeError(
                "ROS 2 Python packages are not importable. Use the system ROS "
                "Python, for example:\n"
                "  source /opt/ros/humble/setup.bash\n"
                "  cd /home/ubuntu/out_dexmal\n"
                "  PYTHONPATH=/home/ubuntu/out_dexmal:$PYTHONPATH /usr/bin/python3 -m "
                "robots.airbots.movebase.cli stop"
            ) from exc

        self._rclpy = rclpy
        self._Twist = Twist
        self._owns_rclpy = False
        if not rclpy.ok():
            rclpy.init(args=None)
            self._owns_rclpy = True

        self._node = rclpy.create_node(self.config.node_name)
        self._publisher = self._node.create_publisher(Twist, self.config.topic, 10)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
        finally:
            self._node.destroy_node()
            if self._owns_rclpy and self._rclpy.ok():
                self._rclpy.shutdown()
            self._closed = True

    def wait_for_subscriber(self, timeout_s: Optional[float] = None) -> bool:
        timeout = (
            self.config.wait_subscriber_timeout_s
            if timeout_s is None
            else max(0.0, timeout_s)
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.1)
            if self._node.get_subscriptions_info_by_topic(self.config.topic):
                return True
        return bool(self._node.get_subscriptions_info_by_topic(self.config.topic))

    def send_velocity(self, x: float, y: float = 0.0, yaw: float = 0.0) -> None:
        x, y, yaw = self._bounded_velocity(x, y, yaw)
        if self.config.require_subscriber and not self.wait_for_subscriber():
            raise RuntimeError(
                f"No subscriber discovered on {self.config.topic}. "
                "Start the base driver first, or set require_subscriber=False."
            )
        self._publish_once(x, y, yaw)

    def move_at_velocity(
        self,
        x: float,
        y: float = 0.0,
        yaw: float = 0.0,
        duration_s: Optional[float] = None,
        stop_after: bool = True,
    ) -> None:
        x, y, yaw = self._bounded_velocity(x, y, yaw)
        if self.config.require_subscriber and not self.wait_for_subscriber():
            raise RuntimeError(
                f"No subscriber discovered on {self.config.topic}. "
                "Start the base driver first, or set require_subscriber=False."
            )

        if duration_s is None:
            self._publish_once(x, y, yaw)
            return

        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if self.config.publish_rate_hz <= 0:
            raise ValueError("publish_rate_hz must be positive")

        period = 1.0 / self.config.publish_rate_hz
        end = time.monotonic() + duration_s
        while time.monotonic() < end:
            self._publish_once(x, y, yaw)
            self._rclpy.spin_once(self._node, timeout_sec=0.0)
            time.sleep(period)

        if stop_after:
            self.stop()

    def stop(self) -> None:
        for _ in range(3):
            self._publish_once(0.0, 0.0, 0.0)
            time.sleep(0.05)

    def get_current_velocity(self) -> Tuple[float, float, float]:
        """Return the last commanded velocity.

        ROS /cmd_vel is command-only. For measured velocity, add an odometry
        subscriber once the base driver exposes its odom topic.
        """

        return self._last_velocity

    def _publish_once(self, x: float, y: float, yaw: float) -> None:
        msg = self._Twist()
        msg.linear.x = x
        msg.linear.y = y
        msg.angular.z = yaw
        self._publisher.publish(msg)
        self._last_velocity = (x, y, yaw)

    def _bounded_velocity(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        max_linear = self.config.max_linear_mps
        max_angular = self.config.max_angular_radps
        if max_linear < 0 or max_angular < 0:
            raise ValueError("velocity limits must be non-negative")
        return (
            self._bounded(x, max_linear, "linear.x"),
            self._bounded(y, max_linear, "linear.y"),
            self._bounded(yaw, max_angular, "angular.z"),
        )

    @staticmethod
    def _bounded(value: float, limit: float, name: str) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if abs(value) > limit:
            raise ValueError(f"{name}={value} exceeds limit {limit}")
        return float(value)

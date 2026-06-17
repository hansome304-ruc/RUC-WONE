from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from robots.airbots.movebase.backends.ros1_cmd_vel import (
    OdomState,
    Ros1CmdVelBackend,
    Ros1CmdVelConfig,
)
from robots.airbots.movebase.backends.ros2_cmd_vel import (
    Ros2CmdVelBackend,
    Ros2CmdVelConfig,
)


@dataclass(frozen=True)
class Velocity:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class MoveBaseConfig:
    backend: str = "ros1_cmd_vel"
    topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    ros_master_uri: str = "http://192.168.31.7:11311/"
    host_ip: Optional[str] = None
    node_name: str = "airbot_movebase"
    publish_rate_hz: float = 20.0
    wait_subscriber_timeout_s: float = 2.0
    require_subscriber: bool = True
    max_linear_mps: float = 0.20
    max_angular_radps: float = 0.50


class MoveBase:
    """High-level mobile base controller.

    The public methods mirror the old mobile-base wrapper but avoid the unused
    airbase_py dependency. The default backend publishes ROS 1 /cmd_vel Twist
    commands because the discovered base exposes a ROS 1 master on Ethernet.
    """

    def __init__(self, config: Optional[MoveBaseConfig] = None, **kwargs) -> None:
        if config is None:
            config = MoveBaseConfig()
        self.config = replace(config, **kwargs)
        self._backend = self._make_backend(self.config)

    def __enter__(self) -> "MoveBase":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        self._backend.close()

    def stop(self) -> None:
        self._backend.stop()

    def send_velocity(self, velocity: Tuple[float, float, float] | Velocity) -> None:
        vel = self._normalize_velocity(velocity)
        self._backend.send_velocity(vel.x, vel.y, vel.yaw)

    def move_at_velocity(
        self,
        velocity: Tuple[float, float, float] | Velocity,
        duration_s: Optional[float] = None,
        stop_after: bool = True,
    ) -> None:
        vel = self._normalize_velocity(velocity)
        self._backend.move_at_velocity(
            vel.x,
            vel.y,
            vel.yaw,
            duration_s=duration_s,
            stop_after=stop_after,
        )

    def move_at_velocity2D(
        self,
        velocity: Tuple[float, float],
        duration_s: Optional[float] = None,
        stop_after: bool = True,
    ) -> None:
        self.move_at_velocity(
            Velocity(x=velocity[0], yaw=velocity[1]),
            duration_s=duration_s,
            stop_after=stop_after,
        )

    def forward(self, speed: float = 0.05, duration_s: float = 0.5) -> None:
        self.move_at_velocity(Velocity(x=speed), duration_s=duration_s)

    def backward(self, speed: float = 0.05, duration_s: float = 0.5) -> None:
        self.move_at_velocity(Velocity(x=-speed), duration_s=duration_s)

    def left(self, speed: float = 0.05, duration_s: float = 0.5) -> None:
        self.move_at_velocity(Velocity(y=speed), duration_s=duration_s)

    def right(self, speed: float = 0.05, duration_s: float = 0.5) -> None:
        self.move_at_velocity(Velocity(y=-speed), duration_s=duration_s)

    def turn_left(self, yaw_speed: float = 0.15, duration_s: float = 0.5) -> None:
        self.move_at_velocity(Velocity(yaw=yaw_speed), duration_s=duration_s)

    def turn_right(self, yaw_speed: float = 0.15, duration_s: float = 0.5) -> None:
        self.move_at_velocity(Velocity(yaw=-yaw_speed), duration_s=duration_s)

    def get_current_velocity(self) -> Tuple[float, float, float]:
        return self._backend.get_current_velocity()

    def get_current_velocity2D(self) -> Tuple[float, float]:
        vel = self.get_current_velocity()
        return vel[0], vel[2]

    def get_odometry(self) -> OdomState:
        if not hasattr(self._backend, "get_odometry"):
            raise NotImplementedError("current backend does not expose odometry")
        return self._backend.get_odometry()

    def get_diagnostics(self) -> dict[str, str]:
        if not hasattr(self._backend, "get_diagnostics"):
            raise NotImplementedError("current backend does not expose diagnostics")
        return self._backend.get_diagnostics()

    def _make_backend(self, config: MoveBaseConfig):
        if config.backend == "ros1_cmd_vel":
            return Ros1CmdVelBackend(
                Ros1CmdVelConfig(
                    master_uri=config.ros_master_uri,
                    host_ip=config.host_ip,
                    topic=config.topic,
                    odom_topic=config.odom_topic,
                    node_name=_ros1_node_name(config.node_name),
                    publish_rate_hz=config.publish_rate_hz,
                    wait_subscriber_timeout_s=config.wait_subscriber_timeout_s,
                    require_subscriber=config.require_subscriber,
                    max_linear_mps=config.max_linear_mps,
                    max_angular_radps=config.max_angular_radps,
                )
            )
        if config.backend == "ros2_cmd_vel":
            return Ros2CmdVelBackend(
                Ros2CmdVelConfig(
                    topic=config.topic,
                    node_name=config.node_name,
                    publish_rate_hz=config.publish_rate_hz,
                    wait_subscriber_timeout_s=config.wait_subscriber_timeout_s,
                    require_subscriber=config.require_subscriber,
                    max_linear_mps=config.max_linear_mps,
                    max_angular_radps=config.max_angular_radps,
                )
            )
        raise ValueError(f"Unsupported movebase backend: {config.backend}")

    @staticmethod
    def _normalize_velocity(velocity: Tuple[float, float, float] | Velocity) -> Velocity:
        if isinstance(velocity, Velocity):
            return velocity
        if len(velocity) != 3:
            raise ValueError("velocity must be (x, y, yaw)")
        return Velocity(x=velocity[0], y=velocity[1], yaw=velocity[2])


def _ros1_node_name(node_name: str) -> str:
    return node_name if node_name.startswith("/") else f"/{node_name}"

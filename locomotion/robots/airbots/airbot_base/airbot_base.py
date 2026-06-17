from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

from robots.airbots.movebase import MoveBase, MoveBaseConfig


@dataclass(frozen=True)
class AIRBOTBaseConfig(object):
    """Compatibility config for old AIRBOTBase imports.

    `ip` and `velocity` are kept so older config files can still be loaded, but
    the current implementation controls the base through movebase.
    """

    ip: Optional[str] = None
    velocity: str = "high"
    backend: str = "ros1_cmd_vel"
    ros_master_uri: str = "http://192.168.31.7:11311/"
    host_ip: Optional[str] = None
    topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    publish_rate_hz: float = 20.0
    wait_subscriber_timeout_s: float = 2.0
    require_subscriber: bool = True
    max_linear_mps: float = 0.20
    max_angular_radps: float = 0.50


class AIRBOTBase(object):
    def __init__(self, config: Optional[AIRBOTBaseConfig] = None, **kwargs) -> None:
        if config is None:
            config = AIRBOTBaseConfig()
        self.config = replace(config, **kwargs)
        self._base = MoveBase(
            MoveBaseConfig(
                backend=self.config.backend,
                ros_master_uri=self.config.ros_master_uri,
                host_ip=self.config.host_ip,
                topic=self.config.topic,
                odom_topic=self.config.odom_topic,
                publish_rate_hz=self.config.publish_rate_hz,
                wait_subscriber_timeout_s=self.config.wait_subscriber_timeout_s,
                require_subscriber=self.config.require_subscriber,
                max_linear_mps=self.config.max_linear_mps,
                max_angular_radps=self.config.max_angular_radps,
            )
        )

    def close(self) -> None:
        self._base.close()

    def stop(self) -> None:
        self._base.stop()

    def move_by_key(self, key):
        if key == "w":
            self._base.forward()
        elif key == "s":
            self._base.backward()
        elif key == "a":
            self._base.turn_left()
        elif key == "d":
            self._base.turn_right()

    def move_at_velocity(self, velocity: Tuple[float, float, float]):
        self._base.move_at_velocity(velocity)

    def move_at_velocity2D(self, velocity: Tuple[float, float]):
        self._base.move_at_velocity2D(velocity)

    def get_current_velocity(self) -> Tuple[float, float, float]:
        return self._base.get_current_velocity()

    def get_current_velocity2D(self) -> Tuple[float, float]:
        return self._base.get_current_velocity2D()

    def get_odometry(self):
        return self._base.get_odometry()

    def lock_change(self):
        raise NotImplementedError("lock_change is not available in the /cmd_vel backend")

    def lock(self, lock_state):
        raise NotImplementedError("lock is not available in the /cmd_vel backend")

    def get_base_lock_state(self):
        raise NotImplementedError("get_base_lock_state is not available in the /cmd_vel backend")

    def build_map(self):
        raise NotImplementedError("build_map is not available in the /cmd_vel backend")

    def load_map(self):
        raise NotImplementedError("load_map is not available in the /cmd_vel backend")

    def move_to_origin(self):
        raise NotImplementedError("move_to_origin is not available in the /cmd_vel backend")

    def record_pose_once(self):
        raise NotImplementedError("record_pose_once is not available in the /cmd_vel backend")

    def save_pose(self, data_dir):
        raise NotImplementedError("save_pose is not available in the /cmd_vel backend")

    def get_current_pose(self):
        state = self._base.get_odometry()
        return state.x, state.y, state.yaw

    def init_behavior(self, track, data_path):
        raise NotImplementedError("init_behavior is not available in the /cmd_vel backend")

    def replay_pose_once(self, index, track=True, wait=True):
        raise NotImplementedError("replay_pose_once is not available in the /cmd_vel backend")

    def move_to_pose(self, x, y, angle, wait=True, backward=False):
        raise NotImplementedError("move_to_pose needs a navigation/action backend")

    def move_to_position(self, x, y, wait=True, backward=False):
        raise NotImplementedError("move_to_position needs a navigation/action backend")

    def move_by_angle(self, angle, wait=True):
        raise NotImplementedError("move_by_angle needs a navigation/action backend")

    def move_by_line(self, x, y, wait=True, backward=False):
        raise NotImplementedError("move_by_line needs a navigation/action backend")

    def move_by_points(self, points, wait=True, backward=False):
        raise NotImplementedError("move_by_points needs a navigation/action backend")


if __name__ == "__main__":
    import time

    airbase = AIRBOTBase(AIRBOTBaseConfig(require_subscriber=False))
    airbase.move_at_velocity((0.0, 0.0, -0.1))
    time.sleep(0.5)
    print(airbase.get_current_velocity())
    airbase.stop()

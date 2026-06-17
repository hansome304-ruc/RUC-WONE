# movebase

`movebase` is the AIRBOT mobile-base control wrapper used by this repository.
It does not depend on `airbase_py`.

The base connected on the wired `192.168.31.*` network exposes a ROS 1 master
at `http://192.168.31.7:11311/`. The default backend therefore publishes ROS 1
`geometry_msgs/Twist` messages to `/cmd_vel` and can read `/odom` without
requiring a local ROS 1 installation.

## Quick Check

Read the current odometry:

```bash
cd /home/ubuntu/out_dexmal
PYTHONPATH=/home/ubuntu/out_dexmal:$PYTHONPATH /usr/bin/python3 -m robots.airbots.movebase.cli state
```

Send a stop command:

```bash
PYTHONPATH=/home/ubuntu/out_dexmal:$PYTHONPATH /usr/bin/python3 -m robots.airbots.movebase.cli stop
```

Move slowly for two seconds. At `0.05 m/s`, this is about `0.1 m`:

```bash
PYTHONPATH=/home/ubuntu/out_dexmal:$PYTHONPATH /usr/bin/python3 -m robots.airbots.movebase.cli forward --speed 0.05 --duration 2.0 --yes
```

If the base master changes:

```bash
PYTHONPATH=/home/ubuntu/out_dexmal:$PYTHONPATH /usr/bin/python3 -m robots.airbots.movebase.cli state --ros-master-uri http://192.168.31.132:11311/
```

The old ROS 2 backend is still available:

```bash
source /opt/ros/humble/setup.bash
PYTHONPATH=/home/ubuntu/out_dexmal:$PYTHONPATH /usr/bin/python3 -m robots.airbots.movebase.cli --backend ros2_cmd_vel stop
```

## Python Usage

```python
from robots.airbots.movebase import MoveBase, MoveBaseConfig, Velocity

cfg = MoveBaseConfig(
    backend="ros1_cmd_vel",
    ros_master_uri="http://192.168.31.7:11311/",
    topic="/cmd_vel",
    odom_topic="/odom",
    max_linear_mps=0.2,
    max_angular_radps=0.5,
)

with MoveBase(cfg) as base:
    print(base.get_odometry())
    base.move_at_velocity(Velocity(x=0.05, y=0.0, yaw=0.0), duration_s=0.5)
    base.turn_left(yaw_speed=0.15, duration_s=0.5)
    base.stop()
```

## API

- `send_velocity((x, y, yaw))`: publish one velocity command.
- `move_at_velocity((x, y, yaw), duration_s=None, stop_after=True)`: publish a command once or continuously for `duration_s`.
- `move_at_velocity2D((x, yaw), duration_s=None, stop_after=True)`: two-dimensional helper.
- `forward()`, `backward()`, `left()`, `right()`, `turn_left()`, `turn_right()`: small convenience moves.
- `stop()`: publish zero velocity several times.
- `get_current_velocity()`: returns measured odometry velocity when available.
- `get_odometry()`: reads one `/odom` sample and returns pose and velocity.

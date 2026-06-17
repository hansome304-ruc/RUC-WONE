from __future__ import annotations

import math
import os
import socket
import struct
import threading
import time
import xmlrpc.client
from dataclasses import dataclass
from socketserver import ThreadingMixIn
from typing import Optional, Tuple
from urllib.parse import urlparse
from xmlrpc.server import SimpleXMLRPCServer


TWIST_TYPE = "geometry_msgs/Twist"
TWIST_MD5 = "9f195f881246fdfa2798d1d3eebca84a"
TWIST_DEFINITION = """# This expresses velocity in free space broken into its linear and angular parts.
Vector3  linear
Vector3  angular

================================================================================
MSG: geometry_msgs/Vector3
# This represents a vector in free space.
float64 x
float64 y
float64 z
"""
ODOM_TYPE = "nav_msgs/Odometry"
ODOM_MD5 = "cd5e73d190d741a2f92e81eda573aca7"
STRING_TYPE = "std_msgs/String"
STRING_MD5 = "992ce8a1687cec8c8bd883ec73ca41d1"


@dataclass(frozen=True)
class Ros1CmdVelConfig:
    master_uri: str = os.environ.get("ROS_MASTER_URI", "http://192.168.31.7:11311/")
    host_ip: Optional[str] = None
    topic: str = "/cmd_vel"
    odom_topic: str = "/odom"
    node_name: str = "/airbot_movebase_cmd_vel"
    publish_rate_hz: float = 20.0
    wait_subscriber_timeout_s: float = 2.0
    require_subscriber: bool = True
    max_linear_mps: float = 0.20
    max_angular_radps: float = 0.50


@dataclass(frozen=True)
class OdomState:
    stamp: float
    frame_id: str
    child_frame_id: str
    x: float
    y: float
    z: float
    yaw: float
    vx: float
    vy: float
    wz: float


class _ThreadingXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    daemon_threads = True
    allow_reuse_address = True


class Ros1CmdVelBackend:
    """Minimal ROS 1 TCPROS publisher for geometry_msgs/Twist.

    This backend intentionally avoids rospy so it can run on this ROS 2-only
    workstation while talking to the base's ROS 1 master over Ethernet.
    """

    def __init__(self, config: Optional[Ros1CmdVelConfig] = None) -> None:
        self.config = config or Ros1CmdVelConfig()
        self._master = xmlrpc.client.ServerProxy(self.config.master_uri)
        self._host_ip = self.config.host_ip or _detect_host_ip(self.config.master_uri)
        self._last_velocity = (0.0, 0.0, 0.0)
        self._connections: list[socket.socket] = []
        self._connections_lock = threading.Lock()
        self._closed = False

        self._tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._tcp_socket.bind((self._host_ip, 0))
        self._tcp_socket.listen(8)
        self._tcp_port = self._tcp_socket.getsockname()[1]

        self._xmlrpc_server = _ThreadingXMLRPCServer(
            (self._host_ip, 0), allow_none=True, logRequests=False
        )
        self._xmlrpc_port = self._xmlrpc_server.server_address[1]
        self._xmlrpc_uri = f"http://{self._host_ip}:{self._xmlrpc_port}/"
        self._register_xmlrpc_methods()

        self._xmlrpc_thread = threading.Thread(
            target=self._xmlrpc_server.serve_forever,
            name="ros1-cmd-vel-xmlrpc",
            daemon=True,
        )
        self._tcp_thread = threading.Thread(
            target=self._accept_tcp_connections,
            name="ros1-cmd-vel-tcpros",
            daemon=True,
        )
        self._xmlrpc_thread.start()
        self._tcp_thread.start()

        self._register_publisher()
        self._notify_subscribers()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
            self._unregister_publisher()
        finally:
            self._closed = True
            try:
                self._xmlrpc_server.shutdown()
            except Exception:
                pass
            try:
                self._xmlrpc_server.server_close()
            except Exception:
                pass
            try:
                self._tcp_socket.close()
            except Exception:
                pass
            with self._connections_lock:
                conns = list(self._connections)
                self._connections.clear()
            for conn in conns:
                _close_socket(conn)

    def wait_for_subscriber(self, timeout_s: Optional[float] = None) -> bool:
        timeout = (
            self.config.wait_subscriber_timeout_s
            if timeout_s is None
            else max(0.0, timeout_s)
        )
        deadline = time.monotonic() + timeout
        target_count = 1
        while time.monotonic() <= deadline:
            apis = self._subscriber_apis()
            target_count = max(target_count, len(apis), 1)
            if self._connection_count() >= target_count:
                return True
            self._notify_subscribers()
            time.sleep(0.1)
        return self._connection_count() > 0

    def send_velocity(self, x: float, y: float = 0.0, yaw: float = 0.0) -> None:
        x, y, yaw = self._bounded_velocity(x, y, yaw)
        if self.config.require_subscriber and not self.wait_for_subscriber():
            raise RuntimeError(
                f"No ROS 1 subscriber connected on {self.config.topic}. "
                f"Check ROS_MASTER_URI={self.config.master_uri} and host_ip={self._host_ip}."
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
                f"No ROS 1 subscriber connected on {self.config.topic}. "
                f"Check ROS_MASTER_URI={self.config.master_uri} and host_ip={self._host_ip}."
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
            time.sleep(period)

        if stop_after:
            self.stop()

    def stop(self) -> None:
        for _ in range(3):
            self._publish_once(0.0, 0.0, 0.0)
            time.sleep(0.05)

    def get_current_velocity(self) -> Tuple[float, float, float]:
        try:
            state = self.get_odometry()
            return state.vx, state.vy, state.wz
        except Exception:
            return self._last_velocity

    def get_odometry(self, timeout_s: float = 3.0) -> OdomState:
        return read_odom_once(
            self.config.master_uri,
            topic=self.config.odom_topic,
            caller_id=self.config.node_name + "_odom",
            timeout_s=timeout_s,
        )

    def get_diagnostics(self, timeout_s: float = 2.0) -> dict[str, str]:
        topics = [
            "/map_info_s",
            "/localization_state",
            "/localization_warn",
            "/sensor_warn",
            "/collision_state",
            "/nav_state_info",
            "/func_state",
            "/autocharge_state",
        ]
        data = {}
        for topic in topics:
            try:
                data[topic] = read_string_once(
                    self.config.master_uri,
                    topic=topic,
                    caller_id=self.config.node_name + "_doctor",
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                data[topic] = f"ERROR: {exc}"
        return data

    @property
    def publisher_uri(self) -> str:
        return self._xmlrpc_uri

    @property
    def host_ip(self) -> str:
        return self._host_ip

    def _register_xmlrpc_methods(self) -> None:
        self._xmlrpc_server.register_function(self._get_bus_stats, "getBusStats")
        self._xmlrpc_server.register_function(self._get_bus_info, "getBusInfo")
        self._xmlrpc_server.register_function(self._get_master_uri, "getMasterUri")
        self._xmlrpc_server.register_function(self._get_pid, "getPid")
        self._xmlrpc_server.register_function(self._get_publications, "getPublications")
        self._xmlrpc_server.register_function(self._get_subscriptions, "getSubscriptions")
        self._xmlrpc_server.register_function(self._publisher_update, "publisherUpdate")
        self._xmlrpc_server.register_function(self._request_topic, "requestTopic")
        self._xmlrpc_server.register_function(self._shutdown, "shutdown")

    def _register_publisher(self) -> None:
        code, msg, _subscribers = self._master.registerPublisher(
            self.config.node_name,
            self.config.topic,
            TWIST_TYPE,
            self._xmlrpc_uri,
        )
        if code != 1:
            raise RuntimeError(f"ROS 1 registerPublisher failed: {msg}")

    def _unregister_publisher(self) -> None:
        try:
            self._master.unregisterPublisher(
                self.config.node_name,
                self.config.topic,
                self._xmlrpc_uri,
            )
        except Exception:
            pass

    def _notify_subscribers(self) -> None:
        for api in self._subscriber_apis():
            try:
                xmlrpc.client.ServerProxy(api).publisherUpdate(
                    self.config.node_name, self.config.topic, [self._xmlrpc_uri]
                )
            except Exception:
                pass

    def _subscriber_apis(self) -> list[str]:
        try:
            code, _msg, state = self._master.getSystemState(self.config.node_name)
            if code != 1:
                return []
            subs = dict(state[1])
            apis = []
            for node_name in subs.get(self.config.topic, []):
                try:
                    code, _msg, api = self._master.lookupNode(
                        self.config.node_name, node_name
                    )
                    if code == 1:
                        apis.append(api)
                except Exception:
                    pass
            return apis
        except Exception:
            return []

    def _accept_tcp_connections(self) -> None:
        while not self._closed:
            try:
                conn, _addr = self._tcp_socket.accept()
            except OSError:
                break
            threading.Thread(
                target=self._handle_tcp_connection,
                args=(conn,),
                name="ros1-cmd-vel-subscriber",
                daemon=True,
            ).start()

    def _handle_tcp_connection(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(3.0)
            _read_tcpros_header(conn)
            conn.sendall(
                _pack_tcpros_header(
                    {
                        "callerid": self.config.node_name,
                        "md5sum": TWIST_MD5,
                        "message_definition": TWIST_DEFINITION,
                        "type": TWIST_TYPE,
                        "topic": self.config.topic,
                        "latching": "0",
                    }
                )
            )
            conn.settimeout(None)
            with self._connections_lock:
                self._connections.append(conn)
        except Exception:
            _close_socket(conn)

    def _publish_once(self, x: float, y: float, yaw: float) -> None:
        payload = struct.pack("<6d", x, y, 0.0, 0.0, 0.0, yaw)
        packet = struct.pack("<I", len(payload)) + payload
        with self._connections_lock:
            conns = list(self._connections)
        dead = []
        for conn in conns:
            try:
                conn.sendall(packet)
            except OSError:
                dead.append(conn)
        if dead:
            with self._connections_lock:
                self._connections = [c for c in self._connections if c not in dead]
            for conn in dead:
                _close_socket(conn)
        self._last_velocity = (x, y, yaw)

    def _connection_count(self) -> int:
        with self._connections_lock:
            return len(self._connections)

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

    def _get_bus_stats(self, _caller_id):
        return 1, "", [[], [], []]

    def _get_bus_info(self, _caller_id):
        return 1, "", []

    def _get_master_uri(self, _caller_id):
        return 1, "", self.config.master_uri

    def _get_pid(self, _caller_id):
        return 1, "", os.getpid()

    def _get_publications(self, _caller_id):
        return 1, "", [[self.config.topic, TWIST_TYPE]]

    def _get_subscriptions(self, _caller_id):
        return 1, "", []

    def _publisher_update(self, _caller_id, _topic, _publishers):
        return 1, "", 0

    def _request_topic(self, _caller_id, topic, protocols):
        if topic != self.config.topic:
            return 0, f"Unknown topic {topic}", []
        for protocol in protocols:
            if protocol and protocol[0] == "TCPROS":
                return 1, "", ["TCPROS", self._host_ip, self._tcp_port]
        return 0, "Only TCPROS is supported", []

    def _shutdown(self, _caller_id, _msg):
        return 1, "", 0


def read_odom_once(
    master_uri: str,
    topic: str = "/odom",
    caller_id: str = "/airbot_movebase_odom_probe",
    timeout_s: float = 3.0,
) -> OdomState:
    master = xmlrpc.client.ServerProxy(master_uri)
    code, msg, state = master.getSystemState(caller_id)
    if code != 1:
        raise RuntimeError(f"ROS 1 getSystemState failed: {msg}")
    publishers = dict(state[0]).get(topic)
    if not publishers:
        raise RuntimeError(f"No ROS 1 publisher found for {topic}")
    code, msg, api = master.lookupNode(caller_id, publishers[0])
    if code != 1:
        raise RuntimeError(f"ROS 1 lookupNode failed for {publishers[0]}: {msg}")
    pub = xmlrpc.client.ServerProxy(api)
    code, msg, proto = pub.requestTopic(caller_id, topic, [["TCPROS"]])
    if code != 1 or not proto or proto[0] != "TCPROS":
        raise RuntimeError(f"ROS 1 requestTopic failed for {topic}: {msg}")

    _, host, port = proto
    with socket.create_connection((host, int(port)), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(
            _pack_tcpros_header(
                {
                    "callerid": caller_id,
                    "md5sum": ODOM_MD5,
                    "topic": topic,
                    "type": ODOM_TYPE,
                    "tcp_nodelay": "1",
                }
            )
        )
        _read_tcpros_header(sock)
        msg_len = struct.unpack("<I", _read_exact(sock, 4))[0]
        return _parse_odom(_read_exact(sock, msg_len))


def read_string_once(
    master_uri: str,
    topic: str,
    caller_id: str = "/airbot_movebase_string_probe",
    timeout_s: float = 2.0,
) -> str:
    master = xmlrpc.client.ServerProxy(master_uri)
    code, msg, state = master.getSystemState(caller_id)
    if code != 1:
        raise RuntimeError(f"ROS 1 getSystemState failed: {msg}")
    publishers = dict(state[0]).get(topic)
    if not publishers:
        raise RuntimeError(f"No ROS 1 publisher found for {topic}")
    code, msg, api = master.lookupNode(caller_id, publishers[0])
    if code != 1:
        raise RuntimeError(f"ROS 1 lookupNode failed for {publishers[0]}: {msg}")
    pub = xmlrpc.client.ServerProxy(api)
    code, msg, proto = pub.requestTopic(caller_id, topic, [["TCPROS"]])
    if code != 1 or not proto or proto[0] != "TCPROS":
        raise RuntimeError(f"ROS 1 requestTopic failed for {topic}: {msg}")

    _, host, port = proto
    with socket.create_connection((host, int(port)), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        sock.sendall(
            _pack_tcpros_header(
                {
                    "callerid": caller_id,
                    "md5sum": STRING_MD5,
                    "topic": topic,
                    "type": STRING_TYPE,
                    "tcp_nodelay": "1",
                }
            )
        )
        _read_tcpros_header(sock)
        msg_len = struct.unpack("<I", _read_exact(sock, 4))[0]
        payload = _read_exact(sock, msg_len)
        string_len = struct.unpack_from("<I", payload, 0)[0]
        return payload[4 : 4 + string_len].decode("utf-8", errors="replace")


def _detect_host_ip(master_uri: str) -> str:
    parsed = urlparse(master_uri)
    host = parsed.hostname
    if not host:
        raise ValueError(f"Invalid ROS master URI: {master_uri}")
    port = parsed.port or 11311
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((host, port))
        return sock.getsockname()[0]
    finally:
        sock.close()


def _pack_tcpros_header(fields: dict[str, str]) -> bytes:
    chunks = []
    for key, value in fields.items():
        item = f"{key}={value}".encode("utf-8")
        chunks.append(struct.pack("<I", len(item)) + item)
    body = b"".join(chunks)
    return struct.pack("<I", len(body)) + body


def _read_tcpros_header(sock: socket.socket) -> dict[str, str]:
    length = struct.unpack("<I", _read_exact(sock, 4))[0]
    data = _read_exact(sock, length)
    fields: dict[str, str] = {}
    offset = 0
    while offset < len(data):
        item_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        item = data[offset : offset + item_len].decode("utf-8", errors="replace")
        offset += item_len
        if "=" in item:
            key, value = item.split("=", 1)
            fields[key] = value
    return fields


def _read_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("socket closed")
        data.extend(chunk)
    return bytes(data)


def _parse_odom(data: bytes) -> OdomState:
    offset = 0
    _seq, secs, nsecs = struct.unpack_from("<III", data, offset)
    offset += 12
    frame_id, offset = _read_ros_string(data, offset)
    child_frame_id, offset = _read_ros_string(data, offset)
    x, y, z, qx, qy, qz, qw = struct.unpack_from("<7d", data, offset)
    offset += 56
    offset += 36 * 8
    vx, vy, _vz, _wx, _wy, wz = struct.unpack_from("<6d", data, offset)
    yaw = _quat_to_yaw(qx, qy, qz, qw)
    return OdomState(
        stamp=secs + nsecs * 1e-9,
        frame_id=frame_id,
        child_frame_id=child_frame_id,
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        vx=vx,
        vy=vy,
        wz=wz,
    )


def _read_ros_string(data: bytes, offset: int) -> tuple[str, int]:
    length = struct.unpack_from("<I", data, offset)[0]
    offset += 4
    value = data[offset : offset + length].decode("utf-8", errors="replace")
    return value, offset + length


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass

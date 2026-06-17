from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import threading
import time
import xmlrpc.client
from dataclasses import dataclass
from socketserver import ThreadingMixIn
from typing import Optional
from urllib.parse import urlparse
from xmlrpc.server import SimpleXMLRPCServer


DEFAULT_MASTER_URI = "http://192.168.31.7:11311/"
POSE_STAMPED_TYPE = "geometry_msgs/PoseStamped"
POSE_STAMPED_MD5 = "d3812c3cbc69362b77dc0b19b345f8f5"
POSE_WITH_COVARIANCE_STAMPED_TYPE = "geometry_msgs/PoseWithCovarianceStamped"
POSE_WITH_COVARIANCE_STAMPED_MD5 = "953b798c0f514ff060a53a3498ce6246"


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float
    frame_id: str = "map"


class _ThreadingXMLRPCServer(ThreadingMixIn, SimpleXMLRPCServer):
    daemon_threads = True
    allow_reuse_address = True


class Ros1PosePublisher:
    def __init__(
        self,
        master_uri: str,
        topic: str,
        msg_type: str,
        md5sum: str,
        node_name: str,
        host_ip: Optional[str] = None,
        latch: bool = True,
    ) -> None:
        self.master_uri = master_uri
        self.topic = topic
        self.msg_type = msg_type
        self.md5sum = md5sum
        self.node_name = node_name if node_name.startswith("/") else f"/{node_name}"
        self.latch = latch
        self._master = xmlrpc.client.ServerProxy(master_uri)
        self._host_ip = host_ip or _detect_host_ip(master_uri)
        self._connections: list[socket.socket] = []
        self._connections_lock = threading.Lock()
        self._last_packet: bytes | None = None
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
            name="ros1-pose-pub-xmlrpc",
            daemon=True,
        )
        self._tcp_thread = threading.Thread(
            target=self._accept_tcp_connections,
            name="ros1-pose-pub-tcpros",
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

    def wait_for_subscriber(self, timeout_s: float = 2.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while time.monotonic() <= deadline:
            self._notify_subscribers()
            if self._connection_count() > 0:
                return True
            time.sleep(0.05)
        return self._connection_count() > 0

    def publish(self, payload: bytes, repeat: int = 8, period_s: float = 0.05) -> None:
        packet = struct.pack("<I", len(payload)) + payload
        if self.latch:
            self._last_packet = packet
        for _ in range(max(1, repeat)):
            self._publish_packet(packet)
            time.sleep(period_s)

    def _publish_packet(self, packet: bytes) -> None:
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

    def _connection_count(self) -> int:
        with self._connections_lock:
            return len(self._connections)

    def _register_xmlrpc_methods(self) -> None:
        self._xmlrpc_server.register_function(lambda _caller_id: (1, "", [[], [], []]), "getBusStats")
        self._xmlrpc_server.register_function(lambda _caller_id: (1, "", []), "getBusInfo")
        self._xmlrpc_server.register_function(lambda _caller_id: (1, "", self.master_uri), "getMasterUri")
        self._xmlrpc_server.register_function(lambda _caller_id: (1, "", os.getpid()), "getPid")
        self._xmlrpc_server.register_function(lambda _caller_id: (1, "", [[self.topic, self.msg_type]]), "getPublications")
        self._xmlrpc_server.register_function(lambda _caller_id: (1, "", []), "getSubscriptions")
        self._xmlrpc_server.register_function(lambda _caller_id, _topic, _publishers: (1, "", 0), "publisherUpdate")
        self._xmlrpc_server.register_function(self._request_topic, "requestTopic")
        self._xmlrpc_server.register_function(lambda _caller_id, _msg: (1, "", 0), "shutdown")

    def _register_publisher(self) -> None:
        code, msg, _subscribers = self._master.registerPublisher(
            self.node_name,
            self.topic,
            self.msg_type,
            self._xmlrpc_uri,
        )
        if code != 1:
            raise RuntimeError(f"ROS1 registerPublisher failed: {msg}")

    def _unregister_publisher(self) -> None:
        try:
            self._master.unregisterPublisher(self.node_name, self.topic, self._xmlrpc_uri)
        except Exception:
            pass

    def _notify_subscribers(self) -> None:
        for api in self._subscriber_apis():
            try:
                xmlrpc.client.ServerProxy(api).publisherUpdate(
                    self.node_name, self.topic, [self._xmlrpc_uri]
                )
            except Exception:
                pass

    def _subscriber_apis(self) -> list[str]:
        try:
            code, _msg, state = self._master.getSystemState(self.node_name)
            if code != 1:
                return []
            subs = dict(state[1])
            out = []
            for node_name in subs.get(self.topic, []):
                try:
                    code, _msg, api = self._master.lookupNode(self.node_name, node_name)
                    if code == 1:
                        out.append(api)
                except Exception:
                    pass
            return out
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
                name="ros1-pose-pub-subscriber",
                daemon=True,
            ).start()

    def _handle_tcp_connection(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(3.0)
            _read_tcpros_header(conn)
            conn.sendall(
                _pack_tcpros_header(
                    {
                        "callerid": self.node_name,
                        "md5sum": self.md5sum,
                        "type": self.msg_type,
                        "topic": self.topic,
                        "latching": "1" if self.latch else "0",
                    }
                )
            )
            conn.settimeout(None)
            with self._connections_lock:
                self._connections.append(conn)
            if self._last_packet is not None:
                conn.sendall(self._last_packet)
        except Exception:
            _close_socket(conn)

    def _request_topic(self, _caller_id, topic, protocols):
        if topic != self.topic:
            return 0, f"Unknown topic {topic}", []
        for protocol in protocols:
            if protocol and protocol[0] == "TCPROS":
                return 1, "", ["TCPROS", self._host_ip, self._tcp_port]
        return 0, "Only TCPROS is supported", []


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send initial pose or navigation goal to the base ROS1 stack.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("set-initial", "goto"):
        p = sub.add_parser(name)
        p.add_argument("--x", type=float, required=True)
        p.add_argument("--y", type=float, required=True)
        p.add_argument("--yaw", type=float, required=True, help="Yaw in radians by default.")
        p.add_argument("--deg", action="store_true", help="Interpret --yaw as degrees.")
        p.add_argument("--frame", default="map")
        p.add_argument("--ros-master-uri", default=DEFAULT_MASTER_URI)
        p.add_argument("--host-ip", default=None)
        p.add_argument("--wait-subscriber", type=float, default=2.0)
        p.add_argument("--repeat", type=int, default=8)
        p.add_argument("--yes", action="store_true", help="Required for goto.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    yaw = math.radians(args.yaw) if args.deg else args.yaw
    pose = Pose2D(x=args.x, y=args.y, yaw=yaw, frame_id=args.frame)

    if args.command == "set-initial":
        publish_initial_pose(args, pose)
        print(f"sent initial pose: frame={pose.frame_id} x={pose.x:.3f} y={pose.y:.3f} yaw={pose.yaw:.3f}rad")
        return 0

    if args.command == "goto":
        if not args.yes:
            raise SystemExit("Refusing to send navigation goal without --yes.")
        publish_goal(args, pose)
        print(f"sent goal: frame={pose.frame_id} x={pose.x:.3f} y={pose.y:.3f} yaw={pose.yaw:.3f}rad")
        return 0

    raise SystemExit(f"unknown command: {args.command}")


def publish_initial_pose(args: argparse.Namespace, pose: Pose2D) -> None:
    pub = Ros1PosePublisher(
        master_uri=args.ros_master_uri,
        topic="/initialpose",
        msg_type=POSE_WITH_COVARIANCE_STAMPED_TYPE,
        md5sum=POSE_WITH_COVARIANCE_STAMPED_MD5,
        node_name="/ruc_wone_initialpose_pub",
        host_ip=args.host_ip,
    )
    try:
        if not pub.wait_for_subscriber(args.wait_subscriber):
            raise RuntimeError("No subscriber connected on /initialpose")
        pub.publish(_pack_pose_with_covariance_stamped(pose), repeat=args.repeat)
    finally:
        pub.close()


def publish_goal(args: argparse.Namespace, pose: Pose2D) -> None:
    pub = Ros1PosePublisher(
        master_uri=args.ros_master_uri,
        topic="/move_base_simple/goal",
        msg_type=POSE_STAMPED_TYPE,
        md5sum=POSE_STAMPED_MD5,
        node_name="/ruc_wone_goal_pub",
        host_ip=args.host_ip,
    )
    try:
        if not pub.wait_for_subscriber(args.wait_subscriber):
            raise RuntimeError("No subscriber connected on /move_base_simple/goal")
        pub.publish(_pack_pose_stamped(pose), repeat=args.repeat)
    finally:
        pub.close()


def _pack_pose_stamped(pose: Pose2D) -> bytes:
    return _pack_header(pose.frame_id) + _pack_pose(pose)


def _pack_pose_with_covariance_stamped(pose: Pose2D) -> bytes:
    covariance = [0.0] * 36
    covariance[0] = 0.25
    covariance[7] = 0.25
    covariance[35] = 0.06853891945200942
    return _pack_header(pose.frame_id) + _pack_pose(pose) + struct.pack("<36d", *covariance)


def _pack_header(frame_id: str) -> bytes:
    now = time.time()
    secs = int(now)
    nsecs = int((now - secs) * 1_000_000_000)
    encoded = frame_id.encode("utf-8")
    return struct.pack("<III", 0, secs, nsecs) + struct.pack("<I", len(encoded)) + encoded


def _pack_pose(pose: Pose2D) -> bytes:
    qz = math.sin(pose.yaw / 2.0)
    qw = math.cos(pose.yaw / 2.0)
    return struct.pack("<7d", pose.x, pose.y, 0.0, 0.0, 0.0, qz, qw)


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


def _close_socket(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except Exception:
        pass
    try:
        sock.close()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

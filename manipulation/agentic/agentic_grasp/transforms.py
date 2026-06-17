"""Pure-numpy SE(3) helpers used to bridge camera frame ↔ arm base frame."""
from __future__ import annotations

import numpy as np


def quat_xyzw_from_R(R: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix → quaternion (x, y, z, w). AIRBOTPlay expects xyzw order."""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0.0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return float(x), float(y), float(z), float(w)


def R_from_quat_xyzw(q: tuple[float, float, float, float]) -> np.ndarray:
    """Quaternion (x, y, z, w) → 3x3 rotation matrix."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_T(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def apply_tcp_offset(T_grasp: np.ndarray, tcp_offset_m: float) -> np.ndarray:
    """Convert a fingertip-frame grasp pose to a flange-frame pose.

    The grasp returned by ZeroGrasp/GraspNet is the fingertip pose of the
    gripper. AIRBOTPlay.move_to_cart_pose expects the *flange* pose. So we
    translate ``tcp_offset_m`` along the gripper's local -z axis.
    """
    T_back = np.eye(4, dtype=np.float64)
    T_back[2, 3] = -float(tcp_offset_m)
    return T_grasp @ T_back


def offset_along_pose_z(T: np.ndarray, dist: float) -> np.ndarray:
    """Translate `T` by `dist` metres along its own +z axis (signed)."""
    out = T.copy()
    out[:3, 3] += float(dist) * T[:3, 2]
    return out


def pose_for_airbot(T: np.ndarray) -> list[list[float]]:
    """Format a 4x4 pose for ``AIRBOTPlay.move_to_cart_pose``."""
    xyz = T[:3, 3].tolist()
    quat = list(quat_xyzw_from_R(T[:3, :3]))
    return [xyz, quat]


def deproject_pixel(K: np.ndarray, u: float, v: float, z: float) -> np.ndarray:
    """Inverse projection of a pixel + depth back to a 3D point in camera frame."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def project_point(K: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Project a 3D point in camera frame into pixel coords."""
    p = np.asarray(p, dtype=np.float64).reshape(3)
    if p[2] <= 0:
        raise ValueError(f"Point is behind camera (z={p[2]:.3f})")
    u = K[0, 0] * p[0] / p[2] + K[0, 2]
    v = K[1, 1] * p[1] / p[2] + K[1, 2]
    return float(u), float(v)

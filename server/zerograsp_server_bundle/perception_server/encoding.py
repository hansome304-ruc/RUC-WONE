"""Mirror of `agentic_grasp/perception_client.py::_b64 / _from_b64`.

We use raw-bytes base64 of the underlying ndarray buffer, NOT PNG/JPEG, so the
encoding is lossless and the same on both sides.
"""
from __future__ import annotations

import base64

import numpy as np


def np_to_b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def b64_to_np(b64: str, dtype, shape) -> np.ndarray:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=dtype)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError(
            f"b64 payload size mismatch: got {arr.size} elements of dtype "
            f"{dtype}, expected {expected} for shape {shape}"
        )
    # frombuffer yields a read-only view of the bytes object; copy so callers
    # can write into it (cv2 / torch.from_numpy / etc).
    return arr.reshape(shape).copy()

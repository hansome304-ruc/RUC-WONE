from __future__ import annotations

import numpy as np

from medicine_agentic.act_export import (
    StationaryDedupConfig,
    _select_stationary_dedup_indices,
)


def _timestamps(count: int, fps: float = 30.0) -> np.ndarray:
    return np.arange(count, dtype=np.float64) / fps


def test_stationary_run_keeps_boundaries_and_every_fifth_frame() -> None:
    values = np.zeros((31, 14), dtype=np.float32)
    selected, stats = _select_stationary_dedup_indices(
        values,
        values,
        _timestamps(31),
        StationaryDedupConfig(min_duration_seconds=0.4, keep_every_n_frames=5),
    )

    assert selected.tolist() == [0, 5, 10, 15, 20, 25, 30]
    assert stats["sample_count_before"] == 31
    assert stats["sample_count_after"] == 7
    assert stats["compressed_run_count"] == 1
    assert stats["keep_every_n_frames"] == 5


def test_short_stationary_run_is_not_compressed() -> None:
    values = np.zeros((10, 14), dtype=np.float32)
    selected, stats = _select_stationary_dedup_indices(
        values,
        values,
        _timestamps(10),
        StationaryDedupConfig(min_duration_seconds=0.4),
    )

    assert selected.tolist() == list(range(10))
    assert stats["removed_sample_count"] == 0


def test_anchor_guard_preserves_slow_motion() -> None:
    values = np.zeros((31, 14), dtype=np.float32)
    values[:, 0] = np.arange(31, dtype=np.float32) * 1.5e-4
    selected, stats = _select_stationary_dedup_indices(
        values,
        values,
        _timestamps(31),
        StationaryDedupConfig(
            joint_tolerance_rad=2e-4,
            min_duration_seconds=0.4,
        ),
    )

    assert selected.tolist() == list(range(31))
    assert stats["compressed_run_count"] == 0


def test_any_arm_or_gripper_motion_prevents_compression() -> None:
    timestamps = _timestamps(31)
    for group, dimension, step in (
        ("qpos", 0, 3e-3),
        ("qpos", 13, 6e-4),
        ("action", 7, 3e-3),
        ("action", 6, 6e-4),
    ):
        qpos = np.zeros((31, 14), dtype=np.float32)
        action = np.zeros((31, 14), dtype=np.float32)
        values = qpos if group == "qpos" else action
        values[:, dimension] = np.arange(31, dtype=np.float32) * step
        selected, stats = _select_stationary_dedup_indices(
            qpos,
            action,
            timestamps,
            StationaryDedupConfig(),
        )

        assert selected.tolist() == list(range(31))
        assert stats["compressed_run_count"] == 0


def test_stationary_stride_must_be_a_positive_integer() -> None:
    for value in (0, -1, 2.5, True):
        try:
            StationaryDedupConfig(keep_every_n_frames=value).validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid keep_every_n_frames={value!r}")


def test_dedup_can_be_disabled() -> None:
    values = np.zeros((31, 14), dtype=np.float32)
    selected, stats = _select_stationary_dedup_indices(
        values,
        values,
        _timestamps(31),
        StationaryDedupConfig(enabled=False),
    )

    assert selected.tolist() == list(range(31))
    assert stats["enabled"] is False

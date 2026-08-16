from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from medicine_agentic.p0_task1_cli import run as run_cli
from medicine_agentic.p0_task1_plan import (
    DEFAULT_REQUIRED_POSES,
    FailureCode,
    P0State,
    P0Task1DryRunPlanner,
    load_plan_config,
)
from medicine_agentic.pose_store import document_sha256, new_pose_document


def _arm_record(offset: float = 0.0) -> dict:
    return {
        "joint_position_rad": [offset, -0.2, 0.5, 0.0, 0.0, 0.0],
        "flange_pose_in_base": {
            "position_m": [0.3, 0.0, 0.2],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "eef_feedback_m": [0.07],
        "driver_state": "IDLE",
        "control_mode": "SERVO_JOINT_POS",
        "capture_metrics": {"stable": True},
    }


def _pose_record() -> dict:
    return {
        "kind": "paired_joint_pose",
        "status": "validated",
        "captured_at": "2026-07-30T00:00:00.000Z",
        "source": "unit_test",
        "instruction": "",
        "scene_id": "unit_test",
        "context": {"base_docked": True, "lift_height_mm": 0.0},
        "tooling": {},
        "arms": {"left": _arm_record(), "right": _arm_record(0.1)},
        "validation": {"stable": True, "collision_free": True},
    }


def _candidate() -> dict:
    pixels = [[380, 200], [372, 200], [388, 200], [380, 192], [380, 208]]
    samples = [820.0, 820.2, 819.8, 820.1, 819.9]
    return {
        "ok": True,
        "candidate": {
            "center_px": [380.0, 200.0],
            "suction_px": [380, 200],
            "polygon_px": [
                [330.0, 175.0],
                [430.0, 175.0],
                [430.0, 225.0],
                [330.0, 225.0],
            ],
            "long_side_px": 100.0,
            "short_side_px": 50.0,
            "angle_deg": 0.0,
            "rectangularity": 0.95,
            "bright_fill": 0.92,
            "edge_clearance_px": 22.0,
            "score": 0.91,
            "provider": "unit_test_visual_prompt",
            "face_type": "front_large",
            "face_score": 0.96,
            "reference_face_id": "front_large_01",
            "graspable": True,
            "grasp_blockers": [],
        },
        "depth": {
            "median_mm": 820.0,
            "spread_mm": 0.4,
            "valid_samples": 5,
            "samples_mm": samples,
            "sample_pixels_px": pixels,
            "frame_age_s": 0.02,
        },
        "point_camera_m": [0.07, -0.04, 0.82],
        "point_left_base_m": [0.35, -0.12, 0.16],
        "physical_size_m": [0.12, 0.06],
        "surface_normal_left_base": [0.0, 0.0, 1.0],
        "surface_tilt_deg": 0.0,
        "plane_residual_mm": 0.2,
        "reachable": True,
        "blockers": [],
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.pose_store = root / "poses.json"
        self.detection = root / "detection.json"
        self.box_config = root / "task1_box.json"
        self.suction_tcp = root / "left_suction_tcp.json"
        self.calibration = root / "cam_to_left.json"
        self.plan_config = root / "p0_plan.json"
        self.logs = root / "logs"

    def write(self) -> None:
        document = new_pose_document("dosw1")
        document["revision"] = 1
        document["poses"] = {
            name: _pose_record() for name in DEFAULT_REQUIRED_POSES
        }
        document["content_sha256"] = document_sha256(document)
        self.pose_store.write_text(json.dumps(document), encoding="utf-8")

        self.calibration.write_text(
            json.dumps(
                {
                    "cam_to_base": [
                        [1.0, 0.0, 0.0, 0.1],
                        [0.0, 1.0, 0.0, 0.2],
                        [0.0, 0.0, 1.0, 0.3],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                }
            ),
            encoding="utf-8",
        )
        self.box_config.write_text(
            json.dumps(
                {
                    "camera": {"cam_to_left_path": str(self.calibration)},
                    "pick": {"tcp_calibrated": True},
                }
            ),
            encoding="utf-8",
        )
        self.suction_tcp.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "calibrated": True,
                    "usable_for_motion": True,
                    "translation_m": [0.0, 0.0, 0.12],
                }
            ),
            encoding="utf-8",
        )
        self.detection.write_text(
            json.dumps(
                {
                    "ok": True,
                    "timestamp": time.time(),
                    "selected": _candidate(),
                }
            ),
            encoding="utf-8",
        )
        self.plan_config.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "inputs": {
                        "pose_store": str(self.pose_store),
                        "detection_report": str(self.detection),
                        "task1_box_config": str(self.box_config),
                        "suction_tcp": str(self.suction_tcp),
                        "log_root": str(self.logs),
                    },
                    "safety": {
                        "dry_run": True,
                        "motion_commands_enabled": False,
                        "suction_commands_enabled": False,
                        "robot_adapter": "none",
                    },
                    "gates": {
                        "robot_id": "dosw1",
                        "required_poses": list(DEFAULT_REQUIRED_POSES),
                        "required_pose_status": "validated",
                        "require_pose_stable": True,
                        "require_collision_free": True,
                        "require_camera_extrinsic": True,
                        "require_suction_tcp": True,
                        "max_detection_age_s": 30.0,
                    },
                    "plan": {
                        "pre_contact_height_m": 0.08,
                        "passive_compression_m": 0.004,
                        "probe_lift_m": 0.02,
                        "full_lift_m": 0.08,
                        "release_clearance_m": 0.004,
                        "pick_descent_speed_m_s": 0.01,
                        "place_descent_speed_m_s": 0.01,
                        "speed_scale": 0.1,
                        "slot_id": "slot_0",
                    },
                }
            ),
            encoding="utf-8",
        )

    def read_plan_config(self) -> dict:
        return json.loads(self.plan_config.read_text(encoding="utf-8"))

    def update_plan_config(self, payload: dict) -> None:
        self.plan_config.write_text(json.dumps(payload), encoding="utf-8")


class P0Task1PlanTests(unittest.TestCase):
    def test_happy_path_is_ready_but_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write()

            report = P0Task1DryRunPlanner(
                load_plan_config(fixture.plan_config),
                run_id="ready_case",
            ).run()

            self.assertTrue(report.ready, report.message)
            self.assertEqual(report.state, P0State.DRY_RUN_COMPLETE)
            self.assertEqual(report.failure_code, FailureCode.NONE)
            self.assertGreater(len(report.actions), 10)
            self.assertTrue(all(not action.executed for action in report.actions))
            summary = json.loads(
                (report.log_dir / "summary.json").read_text(encoding="utf-8")
            )
            self.assertFalse(summary["task_physically_completed"])
            self.assertEqual(
                summary["safety_accounting"]["motion_commands_issued"], 0
            )
            self.assertEqual(
                summary["safety_accounting"]["suction_commands_issued"], 0
            )
            events = [
                json.loads(line)
                for line in (report.log_dir / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(events[0]["to_state"], "CONFIG_GATE")
            self.assertEqual(events[-1]["to_state"], "DRY_RUN_COMPLETE")
            self.assertTrue(
                all(
                    event["outcome"] != "EXECUTED"
                    for event in events
                )
            )

    def test_motion_enable_is_fail_closed_before_pose_or_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write()
            raw = fixture.read_plan_config()
            raw["safety"]["motion_commands_enabled"] = True
            fixture.update_plan_config(raw)

            report = P0Task1DryRunPlanner(
                load_plan_config(fixture.plan_config),
                run_id="motion_block",
            ).run()

            self.assertFalse(report.ready)
            self.assertEqual(
                report.failure_code, FailureCode.MOTION_MUST_BE_DISABLED
            )
            self.assertEqual(report.state, P0State.FAILED)
            self.assertEqual(report.actions, ())

    def test_missing_pose_reports_explicit_failure_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write()
            document = json.loads(fixture.pose_store.read_text(encoding="utf-8"))
            del document["poses"]["recovery_high"]
            document["content_sha256"] = document_sha256(document)
            fixture.pose_store.write_text(json.dumps(document), encoding="utf-8")

            report = P0Task1DryRunPlanner(
                load_plan_config(fixture.plan_config),
                run_id="missing_pose",
            ).run()

            self.assertFalse(report.ready)
            self.assertEqual(report.failure_code, FailureCode.POSE_MISSING)
            self.assertIn("missing", report.message)

    def test_perception_blocker_stops_before_pick_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write()
            detection = json.loads(fixture.detection.read_text(encoding="utf-8"))
            detection["selected"]["blockers"] = ["target surface outside workspace"]
            fixture.detection.write_text(json.dumps(detection), encoding="utf-8")

            report = P0Task1DryRunPlanner(
                load_plan_config(fixture.plan_config),
                run_id="target_blocked",
            ).run()

            self.assertFalse(report.ready)
            self.assertEqual(report.failure_code, FailureCode.TARGET_BLOCKED)
            self.assertEqual(
                [action.state for action in report.actions],
                [P0State.PLAN_HOME, P0State.PLAN_OBSERVE],
            )

    def test_legacy_report_without_grasp_authorization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write()
            detection = json.loads(fixture.detection.read_text(encoding="utf-8"))
            candidate = detection["selected"]["candidate"]
            for key in (
                "provider",
                "face_type",
                "face_score",
                "reference_face_id",
                "graspable",
                "grasp_blockers",
            ):
                candidate.pop(key)
            fixture.detection.write_text(json.dumps(detection), encoding="utf-8")

            report = P0Task1DryRunPlanner(
                load_plan_config(fixture.plan_config),
                run_id="legacy_detection_blocked",
            ).run()

            self.assertFalse(report.ready)
            self.assertEqual(report.failure_code, FailureCode.TARGET_BLOCKED)
            self.assertIn("authorization", report.message)

    def test_cli_writes_report_and_returns_zero_only_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write()
            report_out = fixture.root / "cli_report.json"

            code = run_cli(
                [
                    "--config",
                    str(fixture.plan_config),
                    "--run-id",
                    "cli_ready",
                    "--report-out",
                    str(report_out),
                ]
            )

            self.assertEqual(code, 0)
            payload = json.loads(report_out.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "DRY_RUN_READY")
            self.assertFalse(payload["task_physically_completed"])


if __name__ == "__main__":
    unittest.main()

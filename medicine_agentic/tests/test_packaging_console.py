from __future__ import annotations

import copy
import http.client
import io
import json
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from medicine_agentic.cartesian_jog import CartesianJogSafetyViolation
from medicine_agentic.packaging_camera import (
    CameraFrame,
    CameraUnavailable,
    create_camera,
)
from medicine_agentic.packaging_console import (
    DEFAULT_BIND,
    DEFAULT_PORT,
    PackagingConsoleApp,
    PackagingHTTPServer,
    PackagingRequestHandler,
    RESERVED_PORTS,
    Task3FrontPanelProjector,
    _is_loopback,
    apply_fixed_suction_depth_plane_fallback,
    apply_task1_slot_progress,
    apply_top_view_clockwise_yaw_xyzw,
    axial_orientation_error_deg,
    build_task1_detector_config,
    build_task2_detector_config,
    build_task3_detector_config,
    classify_carton_height,
    cluster_carton_instances,
    compute_fixed_suction_axis_preview,
    describe_act_start_pose,
    detect_task1_staged_carton_top_rgbd,
    estimate_carton_layer,
    flange_offset_for_orientation,
    locate_open_shipping_box_rgbd,
    measure_task1_top_barcode_stripes,
    normalize_task3_front_face_geometry,
    normalized_polygon_contains_point,
    normalized_roi_contains_point,
    normalized_roi_intersects_polygon,
    physical_instance_center_depth_m,
    project_pixel_to_base_z_plane,
    recover_task3_verified_flat_face_geometry,
    run_server,
    segmented_linear_positions,
    select_highest_layer_nearest_base,
    should_recover_task_row,
    shipping_box_detections_consistent,
    shipping_box_image_detections_consistent,
    shipping_box_rim_plane_depth_m,
    shipping_box_region_depth_statistics,
)
from medicine_agentic.task1_box import (
    BoxCandidate,
    estimate_candidate_physical_size_rgbd,
    plan_dual_suction_target,
    propose_task1_surface_grid,
    propose_task2_single_row,
    split_carton_grid_candidate,
)
from medicine_agentic.teleop_launcher import TeleopLaunchConflict, TeleopLauncher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "packaging_console.json"


class TaskRowRecoveryPolicyTests(unittest.TestCase):
    def test_task2_recovers_row_after_individual_identity_match(self) -> None:
        self.assertTrue(
            should_recover_task_row(
                "task2",
                individual_front_similarity=True,
                task_profile={},
            )
        )

    def test_task2_row_recovery_can_be_disabled_explicitly(self) -> None:
        self.assertFalse(
            should_recover_task_row(
                "task2",
                individual_front_similarity=True,
                task_profile={"recover_row_from_verified_identity": False},
            )
        )

    def test_task3_keeps_existing_individual_similarity_behavior(self) -> None:
        self.assertFalse(
            should_recover_task_row(
                "task3",
                individual_front_similarity=True,
                task_profile={},
            )
        )


class FixedSuctionDepthPlaneFallbackTests(unittest.TestCase):
    def support(
        self,
        left_ratio: float,
        right_ratio: float,
        left_depth_m: float,
        right_depth_m: float,
    ) -> dict:
        return {
            "available": True,
            "valid": False,
            "minimum_valid_ratio": 0.8,
            "cups": [
                {
                    "valid_ratio": left_ratio,
                    "median_depth_m": left_depth_m,
                    "valid": left_ratio >= 0.8,
                },
                {
                    "valid_ratio": right_ratio,
                    "median_depth_m": right_depth_m,
                    "valid": right_ratio >= 0.8,
                },
            ],
        }

    def test_accepts_one_sided_hole_only_on_consistent_carton_plane(
        self,
    ) -> None:
        result = apply_fixed_suction_depth_plane_fallback(
            self.support(0.746, 0.192, 0.905, 0.909)
        )
        self.assertTrue(result["valid"])
        self.assertTrue(result["plane_fallback"]["used"])

    def test_rejects_too_little_secondary_depth(self) -> None:
        result = apply_fixed_suction_depth_plane_fallback(
            self.support(1.0, 0.14, 0.968, 0.976)
        )
        self.assertFalse(result["valid"])
        self.assertFalse(result["plane_fallback"]["used"])

    def test_rejects_weak_primary_depth_support(self) -> None:
        result = apply_fixed_suction_depth_plane_fallback(
            self.support(0.69, 0.30, 0.968, 0.976)
        )
        self.assertFalse(result["valid"])
        self.assertFalse(result["plane_fallback"]["used"])

    def test_rejects_different_depth_planes(self) -> None:
        result = apply_fixed_suction_depth_plane_fallback(
            self.support(1.0, 0.25, 0.968, 0.990)
        )
        self.assertFalse(result["valid"])
        self.assertFalse(result["plane_fallback"]["used"])


class Task1SlotPlanPersistenceTests(unittest.TestCase):
    @staticmethod
    def detection() -> dict:
        slots = []
        for identifier in range(1, 21):
            center = [0.2 + identifier * 0.001, -0.3, -0.095]
            slots.append(
                {
                    "slot_id": identifier,
                    "occupied": False,
                    "occupied_at_first_detection": False,
                    "floor_center_left_base_m": center,
                    "release_surface_center_left_base_m": [
                        center[0], center[1], -0.01
                    ],
                    "approach_center_left_base_m": [
                        center[0], center[1], 0.05
                    ],
                    "placement_completion_center_right_base_m": [
                        center[0], 0.3, -0.01
                    ],
                    "approach_center_right_base_m": [center[0], 0.3, 0.05],
                    "carton_long_axis_yaw_right_base_deg": -1.0,
                    "center_px": [100.0 + identifier, 200.0],
                }
            )
        return {
            "id": "grid-1",
            "quality": {
                "high_confidence": True,
                "sample_count": 15,
                "maximum_anchor_peak_to_peak_px": 3.0,
            },
            "metric_grid": {
                "bottom_plane_rms_residual_mm": 3.0,
                "layout_top_view_clockwise_rotation_deg": 90.0,
            },
            "slots": slots,
            "candidate": {"slots": copy.deepcopy(slots)},
            "candidates": [{"slots": copy.deepcopy(slots)}],
        }

    def test_progress_selects_next_slot_without_redetection(self) -> None:
        progressed = apply_task1_slot_progress(self.detection(), [1, 2, 3])

        self.assertEqual(progressed["next_slot"]["slot_id"], 4)
        self.assertEqual(progressed["slot_plan_progress"]["placed_count"], 3)
        self.assertEqual(
            progressed["slot_plan_progress"]["next_placement_number"], 4
        )
        self.assertTrue(progressed["slots"][1]["placed_by_workflow"])

    def test_plan_survives_service_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "task1-slot-plan.json"
            app = PackagingConsoleApp.__new__(PackagingConsoleApp)
            app.task1_slot_plan_state_path = state_path
            app.task1_slot_plan_reuse_enabled = True
            app.task1_slot_grid_cfg = {
                "layout_top_view_clockwise_rotation_deg": 90.0,
            }
            app.task1_box_placement_cfg = {
                "placement_flange_top_view_clockwise_yaw_deg": 90.0,
                "minimum_depth_consensus_samples": 12,
                "maximum_depth_plane_rms_mm": 8.0,
                "maximum_anchor_peak_to_peak_px": 7.0,
            }
            app._workflow_detection_cache = {}
            app._persist_task1_slot_plan(
                self.detection(), [1, 2], detected_at=1234.0
            )

            restored = PackagingConsoleApp.__new__(PackagingConsoleApp)
            restored.task1_slot_plan_state_path = state_path
            restored.task1_slot_plan_reuse_enabled = True
            restored.task1_slot_grid_cfg = copy.deepcopy(app.task1_slot_grid_cfg)
            restored.task1_box_placement_cfg = copy.deepcopy(
                app.task1_box_placement_cfg
            )
            restored._workflow_detection_cache = {}
            restored._restore_task1_slot_plan()

            cached = restored._workflow_detection_cache["task1_box_slots"]
            self.assertTrue(cached["persistent_plan"])
            self.assertEqual(cached["detection"]["next_slot"]["slot_id"], 3)
            self.assertEqual(
                cached["detection"]["slot_plan_progress"]["placed_count"], 2
            )


class Task1StagedCartonTopTests(unittest.TestCase):
    @staticmethod
    def scene(*, with_barcode: bool) -> tuple[np.ndarray, np.ndarray]:
        bgr = np.zeros((240, 400, 3), dtype=np.uint8)
        depth = np.zeros((240, 400), dtype=np.uint16)
        bgr[108:133, 135:265] = (225, 215, 250)
        depth[108:133, 135:265] = 1000
        if with_barcode:
            for x in range(210, 251, 5):
                bgr[112:129, x : x + 2] = (20, 20, 20)
        return bgr, depth

    @staticmethod
    def config() -> dict:
        return {
            "roi_norm": [0.2, 0.3, 0.8, 0.7],
            "expected_center_norm": [0.5, 0.5],
            "maximum_center_distance_norm": 0.1,
            "surface_z_range_left_base_m": [0.95, 1.05],
            "expected_top_size_mm": [130.0, 25.0],
            "top_size_tolerance_mm": [5.0, 3.0],
            "minimum_area_px": 1500.0,
            "minimum_depth_samples": 1500,
            "minimum_pink_fraction": 0.05,
            "pink_hsv_lower": [130, 8, 100],
            "pink_hsv_upper": [179, 180, 255],
            "barcode_roi_norm": [0.48, 0.08, 0.98, 0.92],
            "barcode_minimum_stripe_count": 8,
            "barcode_minimum_anisotropy": 1.1,
        }

    def test_accepts_metric_top_with_barcode_and_returns_geometric_center(
        self,
    ) -> None:
        bgr, depth = self.scene(with_barcode=True)
        detection, _overlay = detect_task1_staged_carton_top_rgbd(
            bgr,
            depth,
            0.001,
            np.asarray(
                [[1000.0, 0.0, 200.0], [0.0, 1000.0, 120.0], [0.0, 0.0, 1.0]]
            ),
            np.eye(4),
            self.config(),
        )

        self.assertTrue(detection["target_ready"])
        candidate = detection["candidate"]
        self.assertTrue(candidate["barcode_evidence"]["valid"])
        np.testing.assert_allclose(
            candidate["physical_size_m"], [0.129, 0.024], atol=0.002
        )
        np.testing.assert_allclose(
            detection["point_left_base_m"], [-0.0005, 0.0, 1.0], atol=0.002
        )

    def test_rejects_same_geometry_without_barcode(self) -> None:
        bgr, depth = self.scene(with_barcode=False)
        detection, _overlay = detect_task1_staged_carton_top_rgbd(
            bgr,
            depth,
            0.001,
            np.asarray(
                [[1000.0, 0.0, 200.0], [0.0, 1000.0, 120.0], [0.0, 0.0, 1.0]]
            ),
            np.eye(4),
            self.config(),
        )

        self.assertFalse(detection["target_ready"])
        self.assertIn(
            "staged_top_barcode_not_found",
            detection["candidates"][0]["blockers"],
        )

    def test_fixed_station_fallback_reconstructs_center_from_partial_face(
        self,
    ) -> None:
        bgr = np.zeros((240, 400, 3), dtype=np.uint8)
        depth = np.zeros((240, 400), dtype=np.uint16)
        bgr[108:133, 135:205] = (225, 215, 250)
        depth[108:133, 135:205] = 1000
        config = {
            **self.config(),
            "fixed_station_partial_surface_fallback_enabled": True,
            "fixed_surface_long_axis_angle_deg": 0.0,
            "fixed_surface_z_left_base_m": 0.99,
            "fixed_surface_z_tolerance_m": 0.02,
            "fixed_station_anchor_maximum_center_distance_norm": 0.10,
        }

        detection, _overlay = detect_task1_staged_carton_top_rgbd(
            bgr,
            depth,
            0.001,
            np.asarray(
                [[1000.0, 0.0, 200.0], [0.0, 1000.0, 120.0], [0.0, 0.0, 1.0]]
            ),
            np.eye(4),
            config,
        )

        self.assertTrue(detection["target_ready"])
        self.assertEqual(
            detection["geometry_source"],
            "fixed_station_visible_rgbd_anchor_and_calibrated_full_face_center",
        )
        candidate = detection["candidate"]
        self.assertTrue(candidate["fixed_station_partial_surface_fallback"]["used"])
        np.testing.assert_allclose(candidate["center_px"], [200.0, 120.0], atol=0.01)
        np.testing.assert_allclose(
            detection["point_left_base_m"], [0.0, 0.0, 0.99], atol=0.002
        )
        np.testing.assert_allclose(candidate["physical_size_m"], [0.13, 0.025])
        self.assertAlmostEqual(
            candidate["fixed_station_partial_surface_fallback"][
                "observed_surface_z_left_base_m"
            ],
            1.0,
        )
        self.assertAlmostEqual(
            candidate["fixed_station_partial_surface_fallback"][
                "target_surface_z_left_base_m"
            ],
            0.99,
        )

    def test_fixed_station_fallback_still_requires_pink_material(self) -> None:
        bgr = np.zeros((240, 400, 3), dtype=np.uint8)
        depth = np.zeros((240, 400), dtype=np.uint16)
        bgr[108:133, 135:205] = (220, 220, 220)
        depth[108:133, 135:205] = 1000
        config = {
            **self.config(),
            "fixed_station_partial_surface_fallback_enabled": True,
            "fixed_station_anchor_maximum_center_distance_norm": 0.10,
        }

        detection, _overlay = detect_task1_staged_carton_top_rgbd(
            bgr,
            depth,
            0.001,
            np.asarray(
                [[1000.0, 0.0, 200.0], [0.0, 1000.0, 120.0], [0.0, 0.0, 1.0]]
            ),
            np.eye(4),
            config,
        )

        self.assertFalse(detection["target_ready"])

    def test_barcode_measurement_uses_right_side_stripe_structure(self) -> None:
        bgr, _depth = self.scene(with_barcode=True)
        evidence = measure_task1_top_barcode_stripes(
            bgr,
            np.asarray([[135, 108], [264, 108], [264, 132], [135, 132]]),
            self.config(),
        )

        self.assertTrue(evidence["valid"])
        self.assertGreaterEqual(evidence["stripe_count"], 8)

    def test_top_view_clockwise_yaw_is_negative_base_z_rotation(self) -> None:
        original = np.asarray([0.0, 0.70710678, 0.0, 0.70710678])
        rotated = apply_top_view_clockwise_yaw_xyzw(original, 5.0)
        expected = np.asarray(
            [0.03084356, 0.70643377, -0.03084356, 0.70643377]
        )
        np.testing.assert_allclose(rotated, expected, atol=1e-7)

    def test_step5_keeps_flange_xy_on_top_center_and_uses_dynamic_height(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task1_staged_top_enabled = True
        app.task1_staged_top_error = ""
        app.task1_staged_top_calibration = {
            "locked_flange_quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071],
            "contact_sample": {"surface_to_flange_z_offset_m": 0.018},
        }
        app.task1_staged_top_cfg = {
            "target_workspace_left_base_m": {
                "x": [-0.05, 0.1],
                "y": [-0.28, -0.08],
                "z": [0.06, 0.13],
            },
            "flange_center_offset_left_base_m": [0.0, 0.0],
            "pre_contact_clearance_m": 0.025,
            "transit_z_m": 0.14,
            "test_lift_m": 0.05,
            "calibrated_workspace_profile": "task1_pick",
            "flange_top_view_clockwise_yaw_deg": 5.0,
        }
        app._motion_transition_lock = threading.RLock()
        app.trajectory_recorder = SimpleNamespace(status=lambda: {"active": False})
        app.trajectory_replay = SimpleNamespace(status=lambda: {"active": False})
        app.act_rollout = SimpleNamespace(status=lambda: {"active": False})
        app.teleop_launcher = SimpleNamespace(status=lambda: {})
        app._system_follow_ownership_active = lambda: False
        app._teleop_status_blocks_cartesian_jog = lambda _status: False
        app.detect_task1_staged_top = mock.Mock(
            return_value={
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.016, -0.181, 0.091],
                    "candidate": {
                        "barcode_evidence": {"valid": True},
                    },
                },
            }
        )
        app._verify_task1_staged_top_suction_axis = mock.Mock(
            return_value={"centers_valid": True}
        )
        app.cartesian_jog = SimpleNamespace(
            status=lambda: {"busy": False},
            move_to_fixed_orientation_entry=mock.Mock(
                return_value={"executed": True}
            ),
            move_fixed_orientation_path=mock.Mock(
                side_effect=[{"executed": True}, {"executed": True}]
            ),
        )
        app.suction = SimpleNamespace(
            status=lambda: {"available": True, "engaged": False},
            set_engaged=mock.Mock(
                return_value={"available": True, "engaged": True}
            ),
            settle_s=0.0,
        )
        app._cartesian_jog_snapshot = lambda: {}

        response = app.run_task1_pick_staged_top_step({})

        rotated = apply_top_view_clockwise_yaw_xyzw(
            [0.0, 0.7071, 0.0, 0.7071], 5.0
        )
        expected_contact = np.asarray([0.016, -0.181, 0.109])
        np.testing.assert_allclose(
            response["result"]["contact_flange_position_m"], expected_contact
        )
        self.assertEqual(
            response["result"]["flange_center_offset_left_base_m"],
            [0.0, 0.0],
        )
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            (expected_contact + np.asarray([0.0, 0.0, 0.025])).tolist(),
            rotated.tolist(),
            transit_z_m=0.14,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task1_staged_top_direct_entry",
            calibrated_workspace_profile="task1_pick",
            use_configured_safe_transit=False,
        )

    def test_step5_right_clearance_sequence_opens_waits_and_moves_y_negative(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task1_box_placement_enabled = True
        app.task1_box_placement_cfg = {
            "post_suction_wait_s": 1.0,
            "post_right_gripper_open_wait_s": 0.5,
            "right_gripper_fully_open_m": 0.06,
            "right_gripper_open_tolerance_m": 0.012,
            "right_y_negative_clearance_m": 0.15,
            "right_enable_token": "ENABLE_RIGHT_ARM_HOME",
        }
        initial = {
            "position_m": [0.033, 0.413, 0.056],
            "quaternion_xyzw": [0.0, 0.0, 0.7, 0.7141428],
            "joint_positions_rad": [0.0] * 6,
            "captured_at": 123.0,
        }
        settled = {
            **initial,
            "gripper_position_m": 0.0587,
            "captured_at": 124.5,
        }
        app.right_arm_home = SimpleNamespace(
            read_current_pose=mock.Mock(side_effect=[initial, settled]),
            move_gripper_to_position=mock.Mock(
                return_value={
                    "executed": True,
                    # Reproduce the real transient: the blocking call returns
                    # before the fingers have mechanically settled.
                    "actual_gripper_position_m": 0.04,
                    "gripper_error_m": 0.02,
                }
            ),
            move_to_fixed_orientation_entry=mock.Mock(
                return_value={
                    "executed": True,
                    "actual_position_m": [0.033, 0.263, 0.056],
                }
            ),
        )
        app._task1_right_clearance_state = None

        with mock.patch(
            "medicine_agentic.packaging_console.time.sleep"
        ) as sleep:
            result = app._prepare_task1_right_arm_clearance()

        self.assertEqual(sleep.call_args_list, [mock.call(1.0), mock.call(0.5)])
        app.right_arm_home.move_gripper_to_position.assert_called_once_with(
            0.06,
            operation="task1_right_gripper_fully_open",
            speed_profile="DEFAULT",
        )
        app.right_arm_home.move_to_fixed_orientation_entry.assert_called_once_with(
            [0.033, 0.263, 0.056],
            initial["quaternion_xyzw"],
            transit_z_m=0.056,
            enable_token="ENABLE_RIGHT_ARM_HOME",
            area_clear=True,
            estop_ready=True,
            operation="task1_right_y_negative_clearance",
            use_configured_safe_transit=False,
        )
        self.assertTrue(result["ready_for_parallel_retreat"])
        self.assertAlmostEqual(
            result["right_gripper"]["settled_actual_gripper_position_m"],
            0.0587,
        )
        self.assertAlmostEqual(
            result["right_gripper"]["settled_gripper_error_m"],
            0.0013,
        )
        self.assertEqual(
            app._task1_right_clearance_state["clearance_position_m"],
            [0.033, 0.263, 0.056],
        )

    def test_step6_directly_places_left_while_right_resets_system_home(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task1_box_placement_enabled = True
        app.task1_box_placement_cfg = {
            "calibrated_workspace_profile": "task1_pick",
            "placement_flange_top_view_clockwise_yaw_deg": 90.0,
            "left_transfer_flange_z_m": 0.16,
            "left_transfer_max_step_m": 0.05,
            "left_approach_max_step_m": 0.04,
            "right_z_positive_retreat_m": 0.15,
            "right_enable_token": "ENABLE_RIGHT_ARM_HOME",
            "left_slot_workspace_m": {
                "x": [0.15, 0.5],
                "y": [-0.4, -0.25],
                "z": [-0.1, 0.1],
            },
        }
        app.task1_staged_top_cfg = {
            "fixed_contact_flange_position_m": [0.0217, -0.2127, 0.088572],
            "fixed_surface_z_left_base_m": 0.085,
            "fixed_contact_flange_quaternion_xyzw": [0.09, 0.702, -0.094, 0.7],
        }
        app.task1_staged_top_calibration = {
            "flange_to_tcp": {
                "translation_m": [0.0180152535, -0.0029797121, 0.0082506053]
            }
        }
        app.task1_staged_top_error = ""
        live_flange_position = [0.0217, -0.2127, 0.208572]
        live_quaternion = [0.09, 0.702, -0.094, 0.7]
        app._workflow_detection_cache = {
            "task1_box_slots": {
                "cached_at": time.time(),
                "detection": {
                    "quality": {
                        "high_confidence": True,
                        "sample_count": 12,
                        "maximum_anchor_peak_to_peak_px": 3.2,
                    },
                    "metric_grid": {
                        "bottom_plane_rms_residual_mm": 3.9,
                        "carton_height_mm": 85.0,
                        "box_center_left_base_m": [0.24, -0.28, -0.095],
                    },
                    "next_slot": {
                        "slot_id": 6,
                        "floor_center_left_base_m": [0.30, -0.33, -0.095],
                        "release_surface_center_left_base_m": [0.30, -0.33, -0.01],
                        "approach_center_left_base_m": [0.30, -0.33, 0.05],
                    },
                    "slot_plan_progress": {
                        "placed_count": 5,
                        "placed_slot_ids": [1, 2, 3, 4, 5],
                    },
                }
            }
        }
        app._task1_right_clearance_state = {
            "ready_for_parallel_retreat": True,
            "initial_position_m": [0.033, 0.413, 0.056],
            "initial_quaternion_xyzw": [0.0, 0.0, 0.7, 0.7141428],
            "clearance_position_m": [0.033, 0.263, 0.056],
        }
        app._motion_transition_lock = threading.RLock()
        app.trajectory_recorder = SimpleNamespace(status=lambda: {"active": False})
        app.trajectory_replay = SimpleNamespace(status=lambda: {"active": False})
        app.act_rollout = SimpleNamespace(status=lambda: {"active": False})
        app.teleop_launcher = SimpleNamespace(status=lambda: {})
        app._teleop_status_blocks_cartesian_jog = lambda _status: False
        app._system_follow_ownership_active = lambda: False
        app.suction = SimpleNamespace(status=lambda: {"engaged": True})
        app.cartesian_jog = SimpleNamespace(
            read_current_pose=mock.Mock(
                return_value={
                    "position_m": live_flange_position,
                    "quaternion_xyzw": live_quaternion,
                }
            ),
            move_to_fixed_orientation_entry=mock.Mock(
                return_value={"executed": True}
            ),
            move_fixed_orientation_path=mock.Mock(
                return_value={"executed": True}
            ),
        )
        app.right_arm_home = SimpleNamespace(
            move_fixed_orientation_path=mock.Mock(
                return_value={"executed": True}
            ),
            reset_home=mock.Mock(
                return_value={"executed": True}
            ),
        )
        app._cartesian_jog_snapshot = lambda: {}
        app._right_arm_home_snapshot = lambda: {}
        app._advance_task1_slot_plan = mock.Mock(
            return_value={
                "slot_plan_progress": {
                    "placed_count": 6,
                    "capacity": 20,
                    "remaining_count": 14,
                    "next_slot_id": 7,
                }
            }
        )

        response = app.run_task1_place_in_box_step({})

        normalized_live_quaternion = (
            np.asarray(live_quaternion) / np.linalg.norm(live_quaternion)
        )
        current_tcp_to_flange = flange_offset_for_orientation(
            app.task1_staged_top_calibration,
            normalized_live_quaternion,
        )
        placement_quaternion = apply_top_view_clockwise_yaw_xyzw(
            normalized_live_quaternion,
            90.0,
        )
        placement_tcp_to_flange = flange_offset_for_orientation(
            app.task1_staged_top_calibration,
            placement_quaternion,
        )
        current_suction = (
            np.asarray(live_flange_position) - current_tcp_to_flange
        )
        orientation_rotation_flange = (
            current_suction + placement_tcp_to_flange
        ).tolist()
        release = (
            np.asarray([0.30, -0.33, -0.01]) + placement_tcp_to_flange
        ).tolist()
        approach = (
            np.asarray([0.30, -0.33, 0.05]) + placement_tcp_to_flange
        ).tolist()
        transfer = [0.24, -0.28, 0.16]
        segmented_transfer = segmented_linear_positions(
            orientation_rotation_flange,
            transfer,
            0.05,
        )
        segmented_approach = segmented_linear_positions(
            transfer,
            approach,
            0.04,
        )
        self.assertLessEqual(len(segmented_transfer), 8)
        self.assertLessEqual(len(segmented_approach), 8)
        self.assertEqual(response["result"]["slot_id"], 6)
        self.assertEqual(
            response["result"]["execution"],
            "parallel_left_place_right_retreat_and_system_home",
        )
        np.testing.assert_allclose(
            response["result"]["left_release_flange_position_m"],
            release,
        )
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            orientation_rotation_flange,
            placement_quaternion.tolist(),
            transit_z_m=live_flange_position[2],
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task1_left_box_slot_clockwise90_about_suction_center",
            calibrated_workspace_profile="task1_pick",
            use_configured_safe_transit=False,
        )
        self.assertEqual(
            app.cartesian_jog.move_fixed_orientation_path.call_args_list,
            [
                mock.call(
                    segmented_transfer,
                    operation="task1_left_box_slot_segmented_transfer",
                    calibrated_workspace_profile="task1_pick",
                ),
                mock.call(
                    segmented_approach,
                    operation="task1_left_box_slot_segmented_approach",
                    calibrated_workspace_profile="task1_pick",
                ),
                mock.call(
                    [release],
                    operation="task1_left_box_slot_descent",
                    calibrated_workspace_profile="task1_pick",
                ),
            ],
        )
        app.right_arm_home.move_fixed_orientation_path.assert_called_once_with(
            [[0.033, 0.263, 0.206]],
            operation="task1_right_z_positive_retreat",
        )
        app.right_arm_home.reset_home.assert_called_once_with(
            speed_profile="DEFAULT",
        )
        self.assertEqual(
            response["result"]["left_entry_mode"],
            "segmented_via_box_center_without_lift",
        )
        self.assertEqual(
            response["result"]["left_segmented_approach_waypoint_count"],
            len(segmented_transfer) + len(segmented_approach),
        )
        self.assertEqual(
            response["result"]["right_return_target"],
            "system_initial_joint_pose",
        )
        self.assertIsNone(app._task1_right_clearance_state)
        app._advance_task1_slot_plan.assert_called_once_with(6)
        self.assertEqual(response["result"]["placement_sequence_number"], 6)
        self.assertEqual(response["result"]["next_slot_id"], 7)
        self.assertEqual(response["result"]["placed_count_after"], 6)
        np.testing.assert_allclose(
            response["result"]["target_suction_center_left_base_m"],
            [0.30, -0.33, -0.01],
        )
        np.testing.assert_allclose(
            response["result"]["current_suction_center_left_base_m"],
            current_suction,
        )
        np.testing.assert_allclose(
            response["result"]["placement_flange_quaternion_xyzw"],
            placement_quaternion,
        )

    def test_step5_uses_exact_operator_taught_pose_and_requested_heights(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task1_staged_top_enabled = True
        app.task1_staged_top_error = ""
        app.task1_staged_top_calibration = {
            "locked_flange_quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071],
            "contact_sample": {"surface_to_flange_z_offset_m": 0.018},
        }
        contact = [0.0217032041, -0.2127215880, 0.0885721154]
        quaternion = [0.0896763548, 0.7024134008, -0.0936777142, 0.6998557363]
        app.task1_staged_top_cfg = {
            "fixed_contact_pose_enabled": True,
            "fixed_contact_pose_frame": "left_base",
            "fixed_contact_flange_position_m": contact,
            "fixed_contact_flange_quaternion_xyzw": quaternion,
            "target_workspace_left_base_m": {
                "x": [-0.05, 0.1],
                "y": [-0.28, -0.08],
                "z": [0.06, 0.13],
            },
            "pre_contact_clearance_m": 0.08,
            "transit_z_m": 0.14,
            "test_lift_m": 0.12,
            "lift_speed_profile": "DEFAULT",
            "calibrated_workspace_profile": "task1_pick",
            "flange_top_view_clockwise_yaw_deg": 15.0,
        }
        app._motion_transition_lock = threading.RLock()
        app.trajectory_recorder = SimpleNamespace(status=lambda: {"active": False})
        app.trajectory_replay = SimpleNamespace(status=lambda: {"active": False})
        app.act_rollout = SimpleNamespace(status=lambda: {"active": False})
        app.teleop_launcher = SimpleNamespace(status=lambda: {})
        app._system_follow_ownership_active = lambda: False
        app._teleop_status_blocks_cartesian_jog = lambda _status: False
        app.detect_task1_staged_top = mock.Mock(
            side_effect=AssertionError("fixed pose must not depend on vision")
        )
        app._verify_task1_staged_top_suction_axis = mock.Mock()
        app.cartesian_jog = SimpleNamespace(
            status=lambda: {"busy": False},
            move_to_fixed_orientation_entry=mock.Mock(
                return_value={"executed": True}
            ),
            move_fixed_orientation_path=mock.Mock(
                side_effect=[{"executed": True}, {"executed": True}]
            ),
        )
        app.suction = SimpleNamespace(
            status=lambda: {"available": True, "engaged": False},
            set_engaged=mock.Mock(
                return_value={"available": True, "engaged": True}
            ),
            settle_s=0.0,
        )
        app._cartesian_jog_snapshot = lambda: {}

        response = app.run_task1_pick_staged_top_step({})

        self.assertEqual(
            response["result"]["flange_center_alignment"],
            "operator_taught_fixed_contact_pose",
        )
        np.testing.assert_allclose(
            response["result"]["contact_flange_position_m"], contact
        )
        np.testing.assert_allclose(
            response["result"]["pre_contact_flange_position_m"],
            [contact[0], contact[1], contact[2] + 0.08],
        )
        np.testing.assert_allclose(
            response["result"]["test_lift_flange_position_m"],
            [contact[0], contact[1], contact[2] + 0.12],
        )
        self.assertNotIn(
            "test_lift_midpoint_flange_position_m",
            response["result"],
        )
        self.assertEqual(response["result"]["test_lift_speed_profile"], "DEFAULT")
        app.detect_task1_staged_top.assert_not_called()
        app._verify_task1_staged_top_suction_axis.assert_not_called()
        normalized_quaternion = (
            np.asarray(quaternion, dtype=np.float64)
            / np.linalg.norm(np.asarray(quaternion, dtype=np.float64))
        ).tolist()
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            [contact[0], contact[1], contact[2] + 0.08],
            normalized_quaternion,
            transit_z_m=contact[2] + 0.08,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task1_staged_top_direct_entry",
            calibrated_workspace_profile="task1_pick",
            use_configured_safe_transit=False,
        )
        self.assertEqual(
            app.cartesian_jog.move_fixed_orientation_path.call_args_list[1],
            mock.call(
                [[contact[0], contact[1], contact[2] + 0.12]],
                operation="task1_staged_top_test_lift",
                calibrated_workspace_profile="task1_pick",
                speed_profile="DEFAULT",
            ),
        )

class Task2ShippingBoxPlacementPlanTests(unittest.TestCase):
    @staticmethod
    def make_app(*, cache_age_s: float = 1.0) -> PackagingConsoleApp:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task2_shipping_box_placement_cfg = {
            "target_cache_max_age_s": 30.0,
            "approach_flange_z_m": 0.22,
            "minimum_opening_margin_m": 0.015,
            "orientation_pose": "system_home",
            "speed_profile": "SLOW",
            "flange_workspace": {
                "x_min": 0.45,
                "x_max": 0.8,
                "y_min": -0.5,
                "y_max": 0.0,
                "z_min": 0.1,
                "z_max": 0.3,
            },
        }
        app.task2_pick_calibration = {
            "usable_for_motion": True,
            "translation_m": [0.018, -0.003, 0.008],
            "flange_to_tcp": {
                "translation_m": [0.018, -0.003, 0.008]
            },
            "locked_flange_quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071],
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": [
                    -0.008,
                    0.003,
                    0.018,
                ]
            },
        }
        app._workflow_detection_cache = {
            "task2_shipping_box": {
                "cached_at": time.time() - cache_age_s,
                "detection": {
                    "id": "box-1",
                    "target_ready": True,
                    "opening_center_left_base_m": [0.63, -0.267, 0.075],
                    "opening_size_m": [0.255, 0.173],
                    "rim_z_m": 0.075,
                },
            }
        }
        app.suction = SimpleNamespace(
            status=lambda: {"available": True, "engaged": True}
        )
        app.trajectory_recorder = SimpleNamespace(status=lambda: {"active": False})
        app.trajectory_replay = SimpleNamespace(status=lambda: {"active": False})
        app.act_rollout = SimpleNamespace(status=lambda: {"active": False})
        app.teleop_launcher = SimpleNamespace(status=lambda: {})
        app.cartesian_jog = SimpleNamespace(
            status=lambda: {
                "busy": False,
                "home_joint_pose": {
                    "joint_positions_rad": [0.0, 0.0, 0.0, 1.5, 0.0, -1.5],
                    "position_tolerance_rad": 0.12,
                },
            },
            read_current_pose=lambda: {
                "joint_positions_rad": [0.0, 0.0, 0.0, 1.5, 0.0, -1.5],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        )
        return app

    def test_preflight_uses_approach_height_as_release_height(self) -> None:
        app = self.make_app()
        with mock.patch.object(
            PackagingConsoleApp,
            "_system_follow_ownership_active",
            return_value=False,
        ):
            response = app.task2_place_shipping_box_preflight({})

        preflight = response["preflight"]
        self.assertTrue(preflight["ready"])
        self.assertEqual(preflight["blockers"], [])
        plan = preflight["plan"]
        self.assertAlmostEqual(plan["release_tcp_left_base_m"][0], 0.63)
        self.assertAlmostEqual(plan["release_tcp_left_base_m"][1], -0.267)
        self.assertAlmostEqual(plan["release_tcp_left_base_m"][2], 0.228)
        self.assertAlmostEqual(plan["release_flange_left_base_m"][0], 0.612)
        self.assertAlmostEqual(plan["release_flange_left_base_m"][1], -0.264)
        self.assertAlmostEqual(plan["release_flange_left_base_m"][2], 0.22)
        self.assertEqual(plan["approach_flange_left_base_m"][2], 0.22)
        self.assertNotIn("post_release_lift_flange_left_base_m", plan)

    def test_tcp_offset_rotates_with_taught_holding_pose(self) -> None:
        calibration = {
            "flange_to_tcp": {"translation_m": [0.018, -0.003, 0.008]}
        }
        identity = flange_offset_for_orientation(
            calibration, [0.0, 0.0, 0.0, 1.0]
        )
        rotated = flange_offset_for_orientation(
            calibration, [0.0, 0.0, 1.0, 0.0]
        )
        np.testing.assert_allclose(identity, [-0.018, 0.003, -0.008])
        np.testing.assert_allclose(rotated, [0.018, -0.003, -0.008])

    def test_preflight_rejects_expired_target_and_released_suction(self) -> None:
        app = self.make_app(cache_age_s=31.0)
        app.suction = SimpleNamespace(
            status=lambda: {"available": True, "engaged": False}
        )
        with mock.patch.object(
            PackagingConsoleApp,
            "_system_follow_ownership_active",
            return_value=False,
        ):
            preflight = app.task2_place_shipping_box_preflight({})["preflight"]

        self.assertFalse(preflight["ready"])
        self.assertIn("task2_shipping_box_target_expired", preflight["blockers"])
        self.assertIn("task2_suction_must_be_engaged", preflight["blockers"])

    def test_preflight_does_not_depend_on_static_enabled_switch(self) -> None:
        app = self.make_app()
        app.task2_shipping_box_placement_cfg.pop("enabled", None)
        with mock.patch.object(
            PackagingConsoleApp,
            "_system_follow_ownership_active",
            return_value=False,
        ):
            preflight = app.task2_place_shipping_box_preflight({})["preflight"]

        self.assertTrue(preflight["ready"])
        self.assertNotIn(
            "task2_shipping_box_placement_disabled",
            preflight["blockers"],
        )

    def test_execute_approaches_once_and_releases_without_descent(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task2_shipping_box_placement_cfg = {
            "confirmation_token": "PLACE_TASK2_IN_SHIPPING_BOX",
            "orientation_pose": "system_home",
            "speed_profile": "SLOW",
            "maximum_system_home_orientation_error_deg": 3.0,
        }
        plan = {
            "approach_flange_left_base_m": [0.622, -0.264, 0.22],
            "release_flange_left_base_m": [0.622, -0.264, 0.22],
            "orientation_quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071],
        }
        app.task2_place_shipping_box_preflight = mock.Mock(
            return_value={
                "preflight": {"ready": True, "blockers": [], "plan": plan}
            }
        )
        app._motion_transition_lock = threading.RLock()
        app.task2_pick_calibration = {
            "locked_flange_quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071]
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.reset_home.return_value = {"executed": True}
        app.cartesian_jog.capture_orientation.return_value = {
            "quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071]
        }
        app.cartesian_jog.enable.return_value = {
            "enabled": True,
            "current_position_m": [0.259, -0.101, 0.151],
        }
        app.cartesian_jog.move_fixed_orientation_path.return_value = {
            "operation": "approach",
            "executed": True,
        }
        app.suction = mock.Mock()
        app.suction.settle_s = 0.0
        app.suction.status.side_effect = [
            {"engaged": True},
            {"engaged": True},
            {"engaged": False},
        ]
        app.suction.set_engaged.return_value = {"engaged": False}
        app._workflow_detection_cache = {"task2_shipping_box": {}}

        result = app.run_task2_place_shipping_box_step(
            {"confirmation": "PLACE_TASK2_IN_SHIPPING_BOX"}
        )

        self.assertTrue(result["placement"]["released"])
        app.cartesian_jog.reset_home.assert_called_once_with(
            speed_profile="SLOW"
        )
        first_path = app.cartesian_jog.move_fixed_orientation_path.call_args_list[
            0
        ].args[0]
        self.assertEqual(first_path[0], [0.259, -0.101, 0.22])
        self.assertEqual(first_path[1], plan["approach_flange_left_base_m"])
        self.assertEqual(len(first_path), 2)
        app.cartesian_jog.move_fixed_orientation_path.assert_called_once()
        app.suction.set_engaged.assert_called_once_with(False)
        self.assertNotIn("task2_shipping_box", app._workflow_detection_cache)


class Task2ShippingBoxConsensusTests(unittest.TestCase):
    @staticmethod
    def detection(*, x: float, y: float, z: float, u: float, v: float) -> dict:
        return {
            "point_left_base_m": [x, y, z],
            "opening_size_m": [0.255, 0.185],
            "candidate": {"center_px": [u, v]},
        }

    def test_rim_z_noise_does_not_reject_stable_opening_xy(self) -> None:
        result = shipping_box_detections_consistent(
            self.detection(x=0.592, y=-0.050, z=0.080, u=790, v=245),
            self.detection(x=0.588, y=-0.056, z=0.130, u=796, v=249),
            xy_tolerance_m=0.025,
            pixel_tolerance_px=35.0,
            size_tolerance_m=0.04,
        )

        self.assertTrue(result["valid"])
        self.assertGreater(result["z_delta_m"], 0.04)

    def test_large_xy_jump_is_still_rejected(self) -> None:
        result = shipping_box_detections_consistent(
            self.detection(x=0.592, y=-0.050, z=0.080, u=790, v=245),
            self.detection(x=0.650, y=-0.120, z=0.081, u=795, v=248),
            xy_tolerance_m=0.025,
            pixel_tolerance_px=35.0,
            size_tolerance_m=0.04,
        )

        self.assertFalse(result["valid"])

    def test_stable_image_geometry_survives_one_bad_depth_frame(self) -> None:
        first = self.detection(x=0.57, y=0.03, z=0.07, u=370, v=185)
        second = self.detection(x=0.0, y=0.0, z=0.0, u=373, v=187)
        first["candidate"].update({"long_side_px": 202.0, "short_side_px": 186.0})
        second["candidate"].update({"long_side_px": 207.0, "short_side_px": 184.0})

        result = shipping_box_image_detections_consistent(
            first,
            second,
            pixel_tolerance_px=35.0,
            side_tolerance_fraction=0.08,
        )

        self.assertTrue(result["valid"])

    def test_image_geometry_rejects_wrong_size_box(self) -> None:
        first = self.detection(x=0.57, y=0.03, z=0.07, u=370, v=185)
        second = self.detection(x=0.57, y=0.03, z=0.07, u=372, v=186)
        first["candidate"].update({"long_side_px": 202.0, "short_side_px": 186.0})
        second["candidate"].update({"long_side_px": 260.0, "short_side_px": 184.0})

        result = shipping_box_image_detections_consistent(
            first,
            second,
            pixel_tolerance_px=35.0,
            side_tolerance_fraction=0.08,
        )

        self.assertFalse(result["valid"])

    def test_rim_plane_median_rejects_one_near_depth_outlier(self) -> None:
        result = shipping_box_rim_plane_depth_m(
            [0.797, 1.085, 1.080, 1.078]
        )

        self.assertAlmostEqual(result, 1.079, places=3)

    def test_rim_plane_requires_three_valid_probes(self) -> None:
        result = shipping_box_rim_plane_depth_m([1.08, float("nan")])

        self.assertIsNone(result)

    def test_pixel_ray_projects_to_configured_base_plane(self) -> None:
        intrinsics = np.asarray(
            [[500.0, 0.0, 120.0], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]]
        )
        transform = np.eye(4, dtype=np.float64)

        result = project_pixel_to_base_z_plane(
            [170.0, 75.0],
            1.0,
            intrinsics,
            transform,
        )

        np.testing.assert_allclose(result, [0.1, -0.05, 1.0], atol=1e-9)

    def test_region_depth_follows_shifted_box_polygon(self) -> None:
        depth = np.full((120, 160), 1.00, dtype=np.float64)
        inner = np.asarray([[40, 25], [130, 25], [130, 95], [40, 95]], dtype=np.float64)
        depth[25:96, 40:131] = 0.92
        depth[38:83, 57:114] = 0.82
        intrinsics = np.asarray(
            [[100.0, 0.0, 80.0], [0.0, 100.0, 60.0], [0.0, 0.0, 1.0]]
        )
        transform = np.eye(4, dtype=np.float64)

        result = shipping_box_region_depth_statistics(
            depth,
            inner,
            intrinsics,
            transform,
            minimum_depth_m=0.3,
            maximum_depth_m=1.5,
            bottom_region_scale=0.55,
            rim_height_quantile=0.8,
            bottom_height_quantile=0.2,
        )

        self.assertAlmostEqual(result["rim_z_m"], 0.92, places=3)
        self.assertAlmostEqual(result["bottom_z_m"], 0.82, places=3)
        self.assertGreater(result["rim_z_m"] - result["bottom_z_m"], 0.09)

    def test_rim_only_shipping_box_target_ignores_hidden_bottom(self) -> None:
        bgr = np.zeros((200, 240, 3), dtype=np.uint8)
        cv2.rectangle(bgr, (40, 30), (200, 150), (40, 100, 160), -1)
        depth_z16 = np.full((200, 240), 1000, dtype=np.uint16)
        intrinsics = np.asarray(
            [[500.0, 0.0, 120.0], [0.0, 500.0, 100.0], [0.0, 0.0, 1.0]]
        )
        cam_to_left = np.eye(4, dtype=np.float64)
        cam_to_left[2, 2] = -1.0
        cam_to_left[2, 3] = 1.1
        config = {
            "roi_norm": [0.0, 0.0, 1.0, 1.0],
            "hsv_lower": [5, 45, 45],
            "hsv_upper": [22, 230, 225],
            "minimum_area_fraction": 0.12,
            "maximum_area_fraction": 0.85,
            "minimum_rectangularity": 0.45,
            "minimum_aspect_ratio": 1.0,
            "maximum_aspect_ratio": 2.4,
            "minimum_opening_size_m": 0.10,
            "maximum_opening_size_m": 0.48,
            "minimum_score": 0.65,
            "minimum_depth_valid_ratio": 0.65,
            "minimum_cavity_depth_m": 0.025,
            "maximum_cavity_depth_m": 0.20,
        }

        strict, _ = locate_open_shipping_box_rgbd(
            bgr,
            depth_z16,
            0.001,
            intrinsics,
            cam_to_left,
            config,
        )
        rim_only, _ = locate_open_shipping_box_rgbd(
            bgr,
            depth_z16,
            0.001,
            intrinsics,
            cam_to_left,
            {**config, "require_cavity_depth": False},
        )

        self.assertFalse(strict["target_ready"])
        self.assertIn("shipping_box_cavity_depth_invalid", strict["blockers"])
        self.assertTrue(rim_only["target_ready"])
        self.assertEqual(rim_only["blockers"], [])
        self.assertFalse(rim_only["cavity_depth_required"])
        self.assertIsNotNone(rim_only["opening_center_left_base_m"])
        self.assertIsNotNone(rim_only["opening_size_m"])

        occluded_depth = np.full((200, 240), 750, dtype=np.uint16)
        fixed_plane, _ = locate_open_shipping_box_rgbd(
            bgr,
            occluded_depth,
            0.001,
            intrinsics,
            cam_to_left,
            {
                **config,
                "require_cavity_depth": False,
                "fixed_rim_z_left_base_m": 0.1,
            },
        )
        self.assertTrue(fixed_plane["target_ready"])
        self.assertAlmostEqual(fixed_plane["rim_z_m"], 0.1, places=6)
        self.assertEqual(fixed_plane["rim_height_source"], "configured_base_plane")
        self.assertAlmostEqual(
            fixed_plane["opening_center_left_base_m"][2],
            0.1,
            places=6,
        )

def camera_config(image_name: str) -> dict:
    return {
        "name": "front",
        "mode": "offline",
        "serial": "test-offline-camera",
        "offline_image": image_name,
        "color": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "format": "bgr8",
        },
        "depth": {
            "width": 1280,
            "height": 720,
            "fps": 30,
            "format": "z16",
        },
    }


def detector_config() -> dict:
    # The endpoint contract is tested independently from a particular carton
    # score. A permissive full-frame profile keeps the synthetic fixture useful.
    return {
        "roi_norm": [0.0, 0.0, 1.0, 1.0],
        "bright_value_min": 120,
        "bright_saturation_max": 255,
        "pink_hue_min": 130,
        "pink_hue_max": 179,
        "pink_saturation_min": 1,
        "pink_value_min": 80,
        "open_kernel_px": 1,
        "close_kernel_px": 3,
        "min_area_fraction": 0.0001,
        "max_area_fraction": 0.20,
        "min_short_side_fraction": 0.005,
        "max_short_side_fraction": 0.60,
        "expected_aspect_ratio": 2.0,
        "min_aspect_ratio": 1.1,
        "max_aspect_ratio": 4.0,
        "min_rectangularity": 0.2,
        "min_bright_fill": 0.2,
        "min_edge_clearance_px": 1,
        "cup_radius_px": 5,
        "dual_suction": {
            "enabled": True,
            "carton_face_size_mm": [130.0, 85.0],
            "cup_diameter_mm": 25.0,
            "cup_edge_gap_mm": 25.0,
            "cup_center_spacing_mm": 50.0,
            "assembly_outer_span_mm": 75.0,
            "safety_margin_mm": 8.0,
            "required_face_types": ["front_large", "back_large"],
        },
        "min_detection_score": 0.1,
    }


def box_candidate(*, score: float, graspable: bool) -> BoxCandidate:
    return BoxCandidate(
        center_px=(210.0, 120.0),
        suction_px=(210, 120),
        polygon_px=(
            (160.0, 90.0),
            (260.0, 90.0),
            (260.0, 150.0),
            (160.0, 150.0),
        ),
        long_side_px=100.0,
        short_side_px=60.0,
        angle_deg=0.0,
        rectangularity=0.95,
        bright_fill=0.9,
        edge_clearance_px=25.0,
        score=score,
        provider="fake",
        face_type="front_large" if graspable else "unknown",
        face_score=0.92 if graspable else 0.0,
        reference_face_id="front_01" if graspable else None,
        graspable=graspable,
        grasp_blockers=() if graspable else ("face_unverified",),
    )


class FakeSelectionProvider:
    name = "fake"

    def detect(self, rgb):
        del rgb
        return [
            box_candidate(score=0.99, graspable=False),
            box_candidate(score=0.88, graspable=True),
        ]

    def status(self):
        return {
            "name": self.name,
            "ok": True,
            "reference_bank": {"ready": True},
        }


class FakeDepthSelectionProvider:
    name = "fake-depth-selection"

    def detect(self, rgb):
        del rgb
        first = box_candidate(score=0.99, graspable=True)
        second = replace(
            box_candidate(score=0.88, graspable=True),
            center_px=(380.0, 120.0),
            suction_px=(380, 120),
            polygon_px=(
                (330.0, 90.0),
                (430.0, 90.0),
                (430.0, 150.0),
                (330.0, 150.0),
            ),
        )
        return [first, second]

    def status(self):
        return {
            "name": self.name,
            "ok": True,
            "reference_bank": {"ready": True},
        }


class CartonGridSplitTests(unittest.TestCase):
    def test_task1_surface_grid_completes_nine_individual_cells(self) -> None:
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        # Three slightly shifted rows emulate placement tolerance while each
        # cell remains an independent physical candidate.
        row_offsets = (8, 0, -7)
        for row, offset in enumerate(row_offsets):
            for column in range(3):
                x0 = 35 + offset + column * 70
                y0 = 75 + row * 78
                rgb[y0 : y0 + 62, x0 : x0 + 58] = [255, 205, 225]
        seed = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(70.0, 105.0),
            face_type="front_large",
            face_score=0.95,
            reference_face_id="front_large_01",
            grasp_blockers=(),
        )

        cells = propose_task1_surface_grid(
            rgb,
            [seed],
            roi_norm=[0.0, 0.1, 0.7, 1.0],
            config={
                "pink_hue_min": 130,
                "pink_hue_max": 175,
                "pink_saturation_min": 8,
                "pink_saturation_max": 130,
                "pink_value_min": 130,
                "task1_grid_minimum_cell_side_px": 30,
                "task1_grid_minimum_cell_pink_fraction": 0.01,
            },
        )

        self.assertEqual(len(cells), 9)
        self.assertEqual({cell.grid_shape for cell in cells}, {(3, 3)})
        self.assertEqual(
            {cell.grid_index for cell in cells},
            {(row, column) for row in range(3) for column in range(3)},
        )
        first_centres = [
            cell.center_px[0] for cell in cells if cell.grid_index[0] == 0
        ]
        last_centres = [
            cell.center_px[0] for cell in cells if cell.grid_index[0] == 2
        ]
        self.assertGreater(np.mean(first_centres), np.mean(last_centres))

    def test_task2_short_axis_places_both_cups_horizontally_inside_one_carton(self) -> None:
        candidate = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            polygon_px=((597.5, 295.0), (682.5, 295.0), (682.5, 425.0), (597.5, 425.0)),
            long_side_px=130.0,
            short_side_px=85.0,
            angle_deg=90.0,
        )
        config = detector_config()
        config["dual_suction"]["alignment"] = "carton_short_axis"
        config["dual_suction"]["safety_margin_mm"] = 2.0

        target = plan_dual_suction_target(
            candidate,
            config,
            image_shape=(720, 1280),
        )

        self.assertIsNotNone(target)
        assert target is not None
        self.assertTrue(target.valid_2d)
        self.assertEqual(target.alignment, "carton_short_axis")
        self.assertAlmostEqual(target.cup_centers_px[0][1], 360.0, places=4)
        self.assertAlmostEqual(target.cup_centers_px[1][1], 360.0, places=4)
        cup_x = sorted(point[0] for point in target.cup_centers_px)
        self.assertLess(cup_x[0], 640.0)
        self.assertGreater(cup_x[1], 640.0)

    def test_graspable_task_candidate_can_skip_redundant_face_type_gate(self) -> None:
        candidate = replace(
            box_candidate(score=0.95, graspable=True),
            polygon_px=(
                (597.5, 295.0),
                (682.5, 295.0),
                (682.5, 425.0),
                (597.5, 425.0),
            ),
            center_px=(640.0, 360.0),
            long_side_px=130.0,
            short_side_px=85.0,
            angle_deg=90.0,
            face_type="unknown",
            reference_face_id=None,
        )
        config = detector_config()
        config["dual_suction"]["alignment"] = "carton_short_axis"
        config["dual_suction"]["safety_margin_mm"] = 2.0

        strict_target = plan_dual_suction_target(
            candidate,
            config,
            image_shape=(720, 1280),
        )
        self.assertIsNotNone(strict_target)
        assert strict_target is not None
        self.assertFalse(strict_target.valid_2d)
        self.assertIn("dual_suction_face_not_allowed", strict_target.blockers)

        config["dual_suction"]["enforce_required_face_types"] = False
        relaxed_target = plan_dual_suction_target(
            candidate,
            config,
            image_shape=(720, 1280),
        )
        self.assertIsNotNone(relaxed_target)
        assert relaxed_target is not None
        self.assertTrue(relaxed_target.valid_2d)
        self.assertNotIn(
            "dual_suction_face_not_allowed",
            relaxed_target.blockers,
        )

    def test_task2_detector_keeps_only_front_motifs_inside_task_roi(self) -> None:
        source = {
            "provider": "reference_feature",
            "reference_feature_ratio": 0.78,
            "reference_feature_min_matches": 6,
            "reference_feature_min_inliers": 6,
            "provider_options": {
                "motif_templates": [
                    {"image": "front.png", "face_type": "front_large"},
                    {"image": "back.png", "face_type": "back_large"},
                ],
                "device": "cpu",
            },
        }
        task2 = build_task2_detector_config(
            source,
            {"include_roi_norm": [0.42, 0.52, 0.72, 0.82]},
        )
        task2_motifs = task2["provider_options"]["motif_templates"]
        self.assertEqual(len(task2_motifs), 1)
        self.assertEqual(task2_motifs[0]["face_type"], "front_large")
        self.assertEqual(task2_motifs[0]["angle_step_deg"], 5.0)
        self.assertEqual(task2_motifs[0]["scale_steps"], 5)
        self.assertEqual(
            task2["provider_options"]["roi_norm"],
            [0.42, 0.52, 0.72, 0.82],
        )
        self.assertEqual(task2["reference_feature_face_types"], ["front_large"])
        self.assertEqual(task2["reference_feature_ratio"], 0.78)
        self.assertEqual(task2["reference_feature_min_matches"], 6)
        self.assertEqual(task2["reference_feature_min_inliers"], 6)
        self.assertEqual(
            source["provider_options"]["motif_templates"],
            [
                {"image": "front.png", "face_type": "front_large"},
                {"image": "back.png", "face_type": "back_large"},
            ],
        )

    def test_task1_detector_reuses_task3_front_face_method(self) -> None:
        source = {
            "provider": "reference_feature",
            "provider_options": {
                "motif_templates": [{"image": "task1_stack.png"}],
            },
        }
        profile = {
            "include_roi_norm": [0.05, 0.25, 0.37, 0.85],
            "include_polygon_norm": [
                [0.05, 0.25],
                [0.255, 0.25],
                [0.215, 0.85],
                [0.05, 0.85],
            ],
            "adaptive_profile_name": "task1_3x3",
            "adaptive_allowed_counts": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "adaptive_recovery_enabled": False,
            "adaptive_reject_glare_matches": False,
            "adaptive_homography_attempt_multiplier": 4,
            "adaptive_sift_ratio": 0.86,
            "adaptive_slot_grid_shape": [3, 3],
            "adaptive_slot_polygon_norm": [
                [0.15, 0.325],
                [0.3525, 0.325],
                [0.338, 0.79],
                [0.11, 0.79],
            ],
            "adaptive_slot_sift_ratio": 0.95,
            "adaptive_slot_min_matches": 6,
            "adaptive_slot_min_inliers": 5,
            "front_similarity_slot_mode": "template",
            "front_similarity_slot_columns": 3,
        }
        task1 = build_task1_detector_config(source, profile)
        self.assertEqual(
            task1["provider_options"]["motif_templates"],
            [],
        )
        self.assertEqual(task1["adaptive_profile_name"], "task1_3x3")
        self.assertEqual(
            task1["adaptive_allowed_counts"],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        self.assertFalse(task1["adaptive_recovery_enabled"])
        self.assertFalse(task1["adaptive_reject_glare_matches"])
        self.assertEqual(task1["adaptive_homography_attempt_multiplier"], 4)
        self.assertEqual(task1["adaptive_sift_ratio"], 0.86)
        self.assertEqual(task1["adaptive_slot_grid_shape"], [3, 3])
        self.assertEqual(
            task1["adaptive_slot_polygon_norm"],
            profile["adaptive_slot_polygon_norm"],
        )
        self.assertEqual(task1["adaptive_slot_sift_ratio"], 0.95)
        self.assertEqual(task1["adaptive_slot_min_matches"], 6)
        self.assertEqual(task1["adaptive_slot_min_inliers"], 5)
        self.assertEqual(
            task1["adaptive_include_polygon_norm"],
            profile["include_polygon_norm"],
        )
        self.assertEqual(
            task1["adaptive_roi_norm"],
            [0.05, 0.25, 0.37, 0.85],
        )
        self.assertIsNot(task1, source)
        self.assertIsNot(task1["provider_options"], source["provider_options"])
        self.assertEqual(
            source["provider_options"]["motif_templates"],
            [{"image": "task1_stack.png"}],
        )

    def test_task3_detector_uses_front_motif_inside_left_roi(self) -> None:
        source = {
            "provider": "reference_feature",
            "provider_options": {
                "motif_templates": [
                    {"image": "front.png", "face_type": "front_large"},
                    {"image": "back.png", "face_type": "back_large"},
                ],
            },
        }
        task3 = build_task3_detector_config(
            source,
            {"include_roi_norm": [0.03, 0.42, 0.36, 0.90]},
        )
        task3_motifs = task3["provider_options"]["motif_templates"]
        self.assertEqual(len(task3_motifs), 1)
        self.assertEqual(task3_motifs[0]["face_type"], "front_large")
        self.assertEqual(task3_motifs[0]["angle_step_deg"], 5.0)
        self.assertEqual(task3_motifs[0]["scale_steps"], 5)
        self.assertEqual(
            task3["provider_options"]["roi_norm"],
            [0.03, 0.42, 0.36, 0.90],
        )
        self.assertEqual(task3["reference_feature_face_types"], ["front_large"])
        self.assertEqual(task3["adaptive_profile_name"], "task3")
        self.assertEqual(task3["adaptive_allowed_counts"], [1, 2])
        self.assertFalse(task3["adaptive_recovery_enabled"])
        self.assertEqual(task3["adaptive_minimum_quad_fill"], 0.70)
        self.assertEqual(
            task3["adaptive_roi_norm"],
            [0.03, 0.42, 0.36, 0.90],
        )
        self.assertEqual(
            source["provider_options"]["motif_templates"],
            [
                {"image": "front.png", "face_type": "front_large"},
                {"image": "back.png", "face_type": "back_large"},
            ],
        )

    def test_task3_projects_only_the_printed_front_inside_flat_dieline(self) -> None:
        reference = np.full((170, 260), 235, dtype=np.uint8)
        cv2.rectangle(reference, (8, 8), (250, 160), 40, 3)
        cv2.putText(
            reference,
            "FRONT 130x85",
            (18, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            30,
            2,
            cv2.LINE_AA,
        )
        for index in range(18):
            center = (20 + (index * 37) % 220, 85 + (index * 23) % 65)
            cv2.circle(reference, center, 3 + index % 5, 20 + index * 8, -1)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "front.png"
            self.assertTrue(cv2.imwrite(str(image_path), reference))
            bank = SimpleNamespace(
                faces=(
                    SimpleNamespace(
                        id="front_large_01",
                        face_type="front_large",
                        pick_allowed=True,
                        image_path=image_path,
                    ),
                )
            )
            projector = Task3FrontPanelProjector(bank)
            self.assertTrue(projector.ready)
            rgb = np.zeros((400, 600, 3), dtype=np.uint8)
            rgb[:, :] = [30, 120, 95]
            rgb[110:280, 160:420] = cv2.cvtColor(
                reference,
                cv2.COLOR_GRAY2RGB,
            )
            full_dieline = replace(
                box_candidate(score=0.95, graspable=True),
                center_px=(290.0, 200.0),
                polygon_px=(
                    (80.0, 60.0),
                    (500.0, 60.0),
                    (500.0, 340.0),
                    (80.0, 340.0),
                ),
                long_side_px=420.0,
                short_side_px=280.0,
                edge_clearance_px=140.0,
            )

            panel = projector.project(rgb, full_dieline)

        self.assertIsNotNone(panel)
        assert panel is not None
        self.assertEqual(panel.face_type, "front_large")
        self.assertEqual(panel.reference_face_id, "front_large_01")
        self.assertAlmostEqual(panel.center_px[0], 289.5, delta=4.0)
        self.assertAlmostEqual(panel.center_px[1], 194.5, delta=4.0)
        self.assertLess(panel.long_side_px, full_dieline.long_side_px)
        target = plan_dual_suction_target(
            panel,
            detector_config(),
            image_shape=rgb.shape[:2],
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertTrue(target.valid_2d)
        contour = np.asarray(panel.polygon_px, dtype=np.float32)
        for cup in target.cup_centers_px:
            self.assertGreater(
                cv2.pointPolygonTest(contour, cup, True),
                target.projected_cup_radius_px,
            )

    def test_center_depth_projection_matches_task2_depth_path(self) -> None:
        report = {
            "method": "inner_axis_center_depth_projected",
            "samples": {
                "long": [[100.0, 200.0, 947.0], [100.0, 300.0, 947.0]],
                "short": [[60.0, 250.0, 947.0], [140.0, 250.0, 947.0]],
            },
        }

        self.assertAlmostEqual(
            physical_instance_center_depth_m(report),
            0.947,
        )
        self.assertIsNone(
            physical_instance_center_depth_m(
                {"method": "inner_axis_rgbd", "samples": report["samples"]}
            )
        )

    def test_task3_rgbd_face_rect_keeps_two_mm_cup_margin(self) -> None:
        candidate = replace(
            box_candidate(score=1.0, graspable=True),
            center_px=(264.3, 485.3),
            suction_px=(264, 485),
            polygon_px=(
                (225.0, 420.0),
                (306.0, 432.0),
                (303.0, 555.0),
                (242.0, 548.0),
            ),
            long_side_px=137.3,
            short_side_px=82.9,
            angle_deg=-80.6,
            provider="reference_feature:task3_front_homography",
            face_type="front_large",
            reference_face_id="front_large_01",
        )
        report = {
            "method": "inner_axis_center_depth_projected",
            "samples": {
                "long": [[0.0, 0.0, 947.0], [0.0, 0.0, 947.0]],
                "short": [[0.0, 0.0, 947.0], [0.0, 0.0, 947.0]],
            },
        }
        intrinsics = np.asarray(
            [[912.0, 0.0, 640.0], [0.0, 912.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        normalized = normalize_task3_front_face_geometry(
            candidate,
            report,
            intrinsics,
            [130.0, 85.0],
        )

        self.assertIsNotNone(normalized)
        assert normalized is not None
        self.assertEqual(normalized.center_px, candidate.center_px)
        self.assertTrue(normalized.provider.endswith(":rgbd_face_rect"))
        config = detector_config()
        config["dual_suction"]["alignment"] = "carton_short_axis"
        config["dual_suction"]["safety_margin_mm"] = 2.0
        config["dual_suction"]["required_face_types"] = ["front_large"]
        target = plan_dual_suction_target(
            normalized,
            config,
            image_shape=(720, 1280),
        )
        self.assertIsNotNone(target)
        assert target is not None
        self.assertTrue(target.valid_2d)
        self.assertEqual(target.blockers, ())

    def test_task3_roi_stack_projects_front_face_without_full_dieline_size(self) -> None:
        stack = replace(
            box_candidate(score=1.0, graspable=True),
            center_px=(240.0, 466.5),
            polygon_px=(
                (178.0, 369.0),
                (302.0, 369.0),
                (302.0, 564.0),
                (178.0, 564.0),
            ),
            long_side_px=195.0,
            short_side_px=124.0,
            angle_deg=-90.0,
            provider="reference_feature:task3_row_cell",
            face_type="back_large",
            reference_face_id="back_large_01",
        )
        folded_size_report = {
            "valid": False,
            "method": "inner_axis_center_depth_projected",
            "samples": {
                "long": [[0.0, 0.0, 950.0], [0.0, 0.0, 950.0]],
                "short": [[0.0, 0.0, 950.0], [0.0, 0.0, 950.0]],
            },
        }
        intrinsics = np.asarray(
            [[912.0, 0.0, 640.0], [0.0, 912.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        self.assertIsNone(
            recover_task3_verified_flat_face_geometry(
                stack,
                folded_size_report,
                intrinsics,
                [130.0, 85.0],
            )
        )
        face = recover_task3_verified_flat_face_geometry(
            stack,
            folded_size_report,
            intrinsics,
            [130.0, 85.0],
            require_verified_front=False,
        )

        self.assertIsNotNone(face)
        assert face is not None
        self.assertEqual(face.center_px, stack.center_px)
        self.assertEqual(face.face_type, "front_large")
        self.assertIsNone(face.reference_face_id)
        self.assertTrue(face.provider.endswith("task3_roi_front_face:rgbd_face_rect"))
        self.assertLess(face.long_side_px, stack.long_side_px)
        self.assertLess(face.short_side_px, stack.short_side_px)

    def test_instance_clustering_keeps_separate_task_layouts_apart(self) -> None:
        first = box_candidate(score=0.9, graspable=True)
        candidates = [
            replace(first, center_px=(100.0, 100.0)),
            replace(first, center_px=(210.0, 100.0)),
            replace(first, center_px=(320.0, 100.0)),
            replace(first, center_px=(900.0, 450.0)),
            replace(first, center_px=(1010.0, 450.0)),
        ]

        groups = cluster_carton_instances(candidates)

        self.assertEqual([len(group) for group in groups], [3, 2])

    def setUp(self) -> None:
        self.depth = np.full((720, 1280), 1000, dtype=np.uint16)
        self.intrinsics = np.asarray(
            [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.config = detector_config()
        self.config["grid_split"] = {
            "enabled": True,
            "maximum_count_per_axis": 3,
            "integer_count_tolerance": 0.55,
            "center_depth_radius_px": 5,
            "minimum_depth_samples": 8,
            "minimum_cell_side_px": 24,
            "maximum_single_carton_area_ratio": 1.45,
            "require_face_color_support": True,
            "minimum_cell_pink_fraction": 0.01,
            "minimum_supported_cell_ratio": 0.5,
        }

    def test_rgbd_physical_gate_accepts_one_face_and_rejects_merged_row(self) -> None:
        config = detector_config()
        config["physical_instance_gate"] = {
            "enabled": True,
            "expected_face_size_mm": [130.0, 85.0],
            "tolerance_mm": [25.0, 20.0],
            "interior_span_fraction": 0.70,
            "sample_radius_px": 3,
            "minimum_depth_samples": 4,
        }
        single = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            long_side_px=130.0,
            short_side_px=85.0,
            angle_deg=0.0,
        )
        merged = replace(
            single,
            long_side_px=390.0,
            polygon_px=(
                (445.0, 317.5),
                (835.0, 317.5),
                (835.0, 402.5),
                (445.0, 402.5),
            ),
        )

        accepted = estimate_candidate_physical_size_rgbd(
            single,
            self.depth,
            0.001,
            self.intrinsics,
            config,
        )
        rejected = estimate_candidate_physical_size_rgbd(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            config,
        )

        self.assertTrue(accepted["valid"])
        np.testing.assert_allclose(
            accepted["estimated_size_mm"], [130.0, 85.0], atol=2.0
        )
        self.assertFalse(rejected["valid"])
        self.assertIn("physical_size_long_mismatch", rejected["blockers"])

    def test_center_depth_projection_ignores_oblique_endpoint_depth_gradient(self) -> None:
        candidate = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            long_side_px=130.0,
            short_side_px=85.0,
            angle_deg=90.0,
        )
        depth = np.full_like(self.depth, 1000)
        depth[:340, :] = 960
        depth[380:, :] = 1040
        config = detector_config()
        config["physical_instance_gate"] = {
            "enabled": True,
            "depth_mode": "center_depth_projected",
            "expected_face_size_mm": [130.0, 85.0],
            "tolerance_mm": [15.0, 15.0],
            "interior_span_fraction": 0.70,
            "sample_radius_px": 3,
            "minimum_depth_samples": 4,
        }

        report = estimate_candidate_physical_size_rgbd(
            candidate,
            depth,
            0.001,
            self.intrinsics,
            config,
        )

        self.assertTrue(report["valid"])
        self.assertEqual(report["method"], "inner_axis_center_depth_projected")
        np.testing.assert_allclose(
            report["estimated_size_mm"], [130.0, 85.0], atol=2.0
        )

    def test_three_by_three_merged_candidate_is_split_into_single_boxes(self) -> None:
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            polygon_px=((445.0, 232.5), (835.0, 232.5), (835.0, 487.5), (445.0, 487.5)),
            long_side_px=390.0,
            short_side_px=255.0,
            angle_deg=0.0,
        )

        cells = split_carton_grid_candidate(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            self.config,
        )

        self.assertEqual(len(cells), 9)
        self.assertEqual(cells[0].grid_shape, (3, 3))
        self.assertEqual(cells[0].grid_index, (1, 1))
        self.assertEqual(cells[0].center_px, (640.0, 360.0))
        self.assertAlmostEqual(cells[0].long_side_px, 130.0)
        self.assertAlmostEqual(cells[0].short_side_px, 85.0)
        self.assertTrue(all(cell.graspable for cell in cells))
        self.assertTrue(all(cell.provider.endswith(":grid_cell") for cell in cells))

    def test_single_carton_is_not_split(self) -> None:
        single = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            long_side_px=130.0,
            short_side_px=85.0,
        )
        cells = split_carton_grid_candidate(
            single,
            self.depth,
            0.001,
            self.intrinsics,
            self.config,
        )
        self.assertEqual(cells, [single])

    def test_two_cartons_side_by_side_are_split_with_swapped_axes(self) -> None:
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            polygon_px=(
                (555.0, 295.0),
                (725.0, 295.0),
                (725.0, 425.0),
                (555.0, 425.0),
            ),
            long_side_px=170.0,
            short_side_px=130.0,
            angle_deg=0.0,
        )

        cells = split_carton_grid_candidate(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            self.config,
        )

        self.assertEqual(len(cells), 2)
        self.assertTrue(all(cell.grid_shape == (2, 1) for cell in cells))
        self.assertEqual(
            {round(cell.center_px[0], 1) for cell in cells},
            {597.5, 682.5},
        )
        self.assertTrue(all(abs(cell.angle_deg) == 90.0 for cell in cells))
        self.assertTrue(all(cell.long_side_px == 130.0 for cell in cells))
        self.assertTrue(all(cell.short_side_px == 85.0 for cell in cells))

    def test_task2_preferred_four_by_one_wins_ambiguous_grid(self) -> None:
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            polygon_px=(
                (469.0, 286.0),
                (811.0, 286.0),
                (811.0, 434.0),
                (469.0, 434.0),
            ),
            long_side_px=342.0,
            short_side_px=148.0,
            angle_deg=0.0,
        )
        depth = np.full_like(self.depth, 1130)
        config = json.loads(json.dumps(self.config))
        config["grid_split"]["maximum_count_per_axis"] = 4
        config["grid_split"]["preferred_grid_shape"] = [4, 1]
        config["grid_split"]["preferred_grid_count_tolerance"] = 1.25

        cells = split_carton_grid_candidate(
            merged,
            depth,
            0.001,
            self.intrinsics,
            config,
        )

        self.assertEqual(len(cells), 4)
        self.assertTrue(all(cell.grid_shape == (4, 1) for cell in cells))
        self.assertEqual(
            {round(cell.center_px[0], 1) for cell in cells},
            {511.8, 597.2, 682.8, 768.2},
        )

    def test_task2_dynamic_single_row_never_turns_three_cartons_into_two_by_two(self) -> None:
        # Reproduces the live Task2 failure: three touching cartons occupy a
        # 257 x 147 px parent at about 0.98 m.  Unconstrained physical scoring
        # can mistake this perspective-distorted row for a 2 x 2 array.
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(779.5, 467.5),
            suction_px=(780, 468),
            polygon_px=(
                (651.0, 394.0),
                (908.0, 394.0),
                (908.0, 541.0),
                (651.0, 541.0),
            ),
            long_side_px=257.0,
            short_side_px=147.0,
            angle_deg=0.0,
        )
        depth = np.full_like(self.depth, 980)
        config = json.loads(json.dumps(self.config))
        config["grid_split"]["maximum_count_per_axis"] = 4
        config["grid_split"]["shape_policy"] = "single_axis_dynamic"

        cells = split_carton_grid_candidate(
            merged,
            depth,
            0.001,
            self.intrinsics,
            config,
        )

        self.assertEqual(len(cells), 3)
        self.assertTrue(all(1 in cell.grid_shape for cell in cells))
        self.assertTrue(all(cell.grid_shape == (3, 1) for cell in cells))
        self.assertEqual(
            {round(cell.center_px[0], 1) for cell in cells},
            {693.8, 779.5, 865.2},
        )
        self.assertEqual(
            {round(cell.center_px[1], 1) for cell in cells},
            {467.5},
        )

    def test_task2_row_geometry_recovers_four_touching_vertical_cartons(self) -> None:
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        rgb[:, :] = [20, 95, 75]
        rgb[340:470, 560:900] = [245, 245, 245]
        seed = replace(
            box_candidate(score=0.97, graspable=True),
            center_px=(730.0, 405.0),
            suction_px=(730, 405),
        )
        config = detector_config()
        config.update(
            {
                "bright_value_min": 220,
                "bright_saturation_max": 30,
                "task2_row_minimum_fill": 0.50,
            }
        )

        cells = propose_task2_single_row(
            rgb,
            self.depth,
            0.001,
            self.intrinsics,
            [seed],
            roi_norm=[0.40, 0.35, 0.80, 0.80],
            config=config,
            maximum_count=4,
        )

        self.assertEqual(len(cells), 4)
        self.assertTrue(all(cell.grid_shape == (4, 1) for cell in cells))
        self.assertTrue(all(cell.angle_deg == -90.0 for cell in cells))
        self.assertEqual(
            [round(cell.center_px[0], 1) for cell in cells],
            [602.5, 687.5, 772.5, 857.5],
        )

    def test_task3_flat_stacks_are_not_split_as_formed_carton_faces(self) -> None:
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        rgb[:, :] = [20, 95, 75]
        rgb[363:542, 59:196] = [245, 245, 245]
        rgb[377:556, 206:335] = [245, 245, 245]
        seed = replace(
            box_candidate(score=0.97, graspable=True),
            center_px=(127.5, 452.5),
            suction_px=(128, 453),
        )
        config = detector_config()
        config["dual_suction"]["carton_face_size_mm"] = [205.0, 130.0]
        config.update(
            {
                "bright_value_min": 220,
                "bright_saturation_max": 30,
                "task2_row_minimum_fill": 0.50,
            }
        )
        depth = np.full_like(self.depth, 950)

        cells = propose_task2_single_row(
            rgb,
            depth,
            0.001,
            self.intrinsics,
            [seed],
            roi_norm=[0.03, 0.32, 0.42, 0.92],
            config=config,
            maximum_count=2,
        )

        self.assertEqual(len(cells), 2)
        self.assertEqual(
            [round(cell.center_px[0], 1) for cell in cells],
            [127.5, 270.5],
        )
        self.assertTrue(all(cell.grid_shape == (2, 1) for cell in cells))
        self.assertTrue(all(cell.long_side_px > 180.0 for cell in cells))
        self.assertTrue(all(cell.short_side_px > 115.0 for cell in cells))

    def test_task2_row_geometry_joins_three_slightly_shifted_cartons(self) -> None:
        rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
        rgb[:, :] = [20, 95, 75]
        for index, (x, y) in enumerate(
            ((600, 350), (689, 346), (778, 354))
        ):
            del index
            rgb[y : y + 130, x : x + 85] = [245, 245, 245]
        seed = replace(
            box_candidate(score=0.97, graspable=True),
            center_px=(730.0, 410.0),
            suction_px=(730, 410),
        )
        config = detector_config()
        config.update(
            {
                "bright_value_min": 220,
                "bright_saturation_max": 30,
                "task2_row_minimum_fill": 0.50,
            }
        )

        cells = propose_task2_single_row(
            rgb,
            self.depth,
            0.001,
            self.intrinsics,
            [seed],
            roi_norm=[0.40, 0.35, 0.80, 0.80],
            config=config,
            maximum_count=4,
        )

        self.assertEqual(len(cells), 3)
        self.assertTrue(all(cell.grid_shape == (3, 1) for cell in cells))
        self.assertLess(max(cell.center_px[1] for cell in cells) - min(
            cell.center_px[1] for cell in cells
        ), 10.0)

    def test_task2_vertical_axis_gate_wraps_plus_and_minus_ninety(self) -> None:
        self.assertAlmostEqual(axial_orientation_error_deg(89.0, 90.0), 1.0)
        self.assertAlmostEqual(axial_orientation_error_deg(-89.0, 90.0), 1.0)
        self.assertAlmostEqual(axial_orientation_error_deg(0.0, 90.0), 90.0)

    def test_task_regions_include_sources_and_exclude_finished_carton(self) -> None:
        shape = (720, 1280)
        task1_source = [0.72, 0.30, 1.00, 1.00]
        task3_source = [0.03, 0.32, 0.42, 0.92]
        finished_carton = [0.23, 0.06, 0.49, 0.46]
        self.assertTrue(
            normalized_roi_contains_point((1150.0, 500.0), shape, task1_source)
        )
        self.assertTrue(
            normalized_roi_contains_point((250.0, 430.0), shape, task3_source)
        )
        self.assertTrue(
            normalized_roi_contains_point((500.0, 220.0), shape, finished_carton)
        )
        self.assertFalse(
            normalized_roi_contains_point((500.0, 220.0), shape, task1_source)
        )
        # The centre is outside the finished-goods ROI, but the carton itself
        # crosses the boundary and must still be excluded from picking.
        boundary_carton = (
            (600.0, 200.0),
            (650.0, 200.0),
            (650.0, 270.0),
            (600.0, 270.0),
        )
        self.assertTrue(
            normalized_roi_intersects_polygon(
                boundary_carton,
                shape,
                finished_carton,
            )
        )
        self.assertFalse(
            normalized_roi_intersects_polygon(
                ((900.0, 500.0), (950.0, 500.0), (950.0, 550.0), (900.0, 550.0)),
                shape,
                finished_carton,
            )
        )

    def test_task1_slanted_source_polygon_includes_all_three_columns(self) -> None:
        shape = (720, 1280)
        source = [
            [0.15, 0.325],
            [0.3525, 0.325],
            [0.338, 0.79],
            [0.11, 0.79],
        ]
        for center in (
            (213.0, 293.0), (282.0, 293.0), (350.0, 293.0),
            (196.0, 410.0), (265.0, 410.0), (333.0, 410.0),
            (187.0, 526.0), (256.0, 526.0), (324.0, 526.0),
        ):
            self.assertTrue(
                normalized_polygon_contains_point(center, shape, source)
            )
        self.assertFalse(
            normalized_polygon_contains_point((500.0, 410.0), shape, source)
        )

    def test_oversized_unsplit_candidate_is_never_graspable(self) -> None:
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            long_side_px=127.0,
            short_side_px=127.0,
            angle_deg=0.0,
        )

        cells = split_carton_grid_candidate(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            self.config,
        )

        self.assertEqual(len(cells), 1)
        self.assertFalse(cells[0].graspable)
        self.assertIn("touching_cartons_unsplit", cells[0].grasp_blockers)

    def test_non_pink_floor_candidate_is_not_expanded_into_a_grid(self) -> None:
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            long_side_px=390.0,
            short_side_px=255.0,
            angle_deg=0.0,
        )
        beige_floor = np.full((720, 1280, 3), (190, 180, 160), dtype=np.uint8)
        cells = split_carton_grid_candidate(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            self.config,
            rgb=beige_floor,
        )
        self.assertEqual(len(cells), 1)
        self.assertFalse(cells[0].graspable)
        self.assertIn(
            "touching_cartons_split_unverified",
            cells[0].grasp_blockers,
        )

    def test_sparse_grid_drops_cells_whose_carton_face_is_absent(self) -> None:
        merged = replace(
            box_candidate(score=0.95, graspable=True),
            center_px=(640.0, 360.0),
            suction_px=(640, 360),
            polygon_px=(
                (445.0, 232.5),
                (835.0, 232.5),
                (835.0, 487.5),
                (445.0, 487.5),
            ),
            long_side_px=390.0,
            short_side_px=255.0,
            angle_deg=0.0,
        )
        unfiltered = split_carton_grid_candidate(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            self.config,
        )
        rgb = np.full((720, 1280, 3), (20, 130, 110), dtype=np.uint8)
        occupied_indices = {(0, 0), (0, 1), (1, 0), (2, 2)}
        for cell in unfiltered:
            if cell.grid_index in occupied_indices:
                polygon = np.round(np.asarray(cell.polygon_px)).astype(np.int32)
                cv2.fillConvexPoly(rgb, polygon, (255, 180, 205))
        config = json.loads(json.dumps(self.config))
        config["grid_split"]["drop_unsupported_cells"] = True
        config["grid_split"]["minimum_cell_pink_fraction"] = 0.12

        filtered = split_carton_grid_candidate(
            merged,
            self.depth,
            0.001,
            self.intrinsics,
            config,
            rgb=rgb,
        )

        self.assertEqual(len(filtered), 4)
        self.assertEqual(
            {cell.grid_index for cell in filtered},
            occupied_indices,
        )

    def test_selection_prefers_highest_layer_then_nearest_left_base(self) -> None:
        near_layer_two = object()
        far_layer_three = object()
        near_layer_three = object()
        candidates = [near_layer_two, far_layer_three, near_layer_three]
        layers = {
            id(near_layer_two): {"valid": True, "layer": 2},
            id(far_layer_three): {"valid": True, "layer": 3},
            id(near_layer_three): {"valid": True, "layer": 3},
        }
        base_points = {
            id(near_layer_two): (0.10, 0.10, 0.05),
            id(far_layer_three): (0.40, 0.30, 0.08),
            id(near_layer_three): (0.20, 0.10, 0.08),
        }
        selected = select_highest_layer_nearest_base(
            candidates,
            layers,
            base_points,
        )
        self.assertIs(selected, near_layer_three)


class RunningConsole:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.static_dir = self.root / "static"
        self.static_dir.mkdir()
        (self.static_dir / "index.html").write_text(
            "<!doctype html><title>Packaging test</title>",
            encoding="utf-8",
        )
        (self.root / "outside-secret.txt").write_text(
            "must not be served",
            encoding="utf-8",
        )

        image = np.full((240, 424, 3), (30, 95, 30), dtype=np.uint8)
        cv2.rectangle(image, (120, 70), (310, 165), (220, 190, 235), -1)
        self.image_path = self.root / "offline.jpg"
        if not cv2.imwrite(str(self.image_path), image):
            raise RuntimeError("failed to write synthetic offline camera image")

        self.config_path = self.root / "packaging_console.json"
        config = {
            "server": {"bind": "127.0.0.1", "port": 8899},
            "static_dir": "static",
            "camera": camera_config(self.image_path.name),
            "wrist_cameras": {
                "enabled": True,
                "base_url": "http://127.0.0.1:8877",
                "left_name": "left",
                "right_name": "right",
            },
            "detector": detector_config(),
        }
        self.config_path.write_text(
            json.dumps(config),
            encoding="utf-8",
        )
        self.app = PackagingConsoleApp(
            config,
            config_path=self.config_path,
            bind="127.0.0.1",
            port=0,
        )
        self.server = PackagingHTTPServer(
            ("127.0.0.1", 0),
            PackagingRequestHandler,
            self.app,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = int(self.server.server_address[1])

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3.0)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            payload = response.read()
            return response.status, dict(response.getheaders()), payload
        finally:
            connection.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3.0)
        self.app.close()
        self.temporary.cleanup()


class TeleopLauncherTests(unittest.TestCase):
    def test_console_hard_restart_applies_motion_interlocks_and_delegates(
        self,
    ) -> None:
        app = object.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.act_rollout = mock.Mock()
        app.trajectory_replay = mock.Mock()
        app.trajectory_recorder = mock.Mock()
        for controller in (
            app.act_rollout,
            app.trajectory_replay,
            app.trajectory_recorder,
        ):
            controller.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.status.return_value = {
            "enabled": False,
            "busy": False,
        }
        app.teleop_launcher = mock.Mock()
        accepted = {
            "confirm": "HARD_RESTART_TELEOP",
            "area_clear": True,
            "estop_ready": True,
            "master_arms_stable": True,
        }
        app.teleop_launcher.hard_restart.return_value = {
            "ok": True,
            "operation_id": "restart-1",
            "teleop": {"state": "hard-restarting"},
        }
        app._augment_teleop_status = mock.Mock(
            return_value={"state": "hard-restarting", "operator_mode": "dual"}
        )

        result = app.hard_restart_teleop(accepted)

        app.teleop_launcher.hard_restart.assert_called_once_with(accepted)
        app.cartesian_jog.close.assert_called_once_with()
        self.assertEqual(result["teleop"]["state"], "hard-restarting")

        app.act_rollout.status.return_value = {"active": True}
        with self.assertRaisesRegex(TeleopLaunchConflict, "ACT rollout"):
            app.hard_restart_teleop(accepted)

    def test_start_requires_explicit_operator_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = TeleopLauncher(
                {
                    "enabled": True,
                    "repo_root": ".",
                    "arm_runtime_config": "arm_services.json",
                },
                config_dir=root,
            )
            with self.assertRaisesRegex(
                ValueError,
                "request fields must be exactly",
            ):
                launcher.start({"area_clear": True})
            with self.assertRaisesRegex(
                ValueError,
                "request fields must be exactly",
            ):
                launcher.start({"confirm": "START_FOLLOW"})

    def test_start_reuses_saved_endpoints_and_existing_scripts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm_services.json").write_text(
                json.dumps(
                    {
                        "remote_host": "10.20.30.40",
                        "ssh_target": "robot@10.20.30.40",
                        "remote_left_iface": "can0",
                        "remote_right_iface": "can1",
                    }
                ),
                encoding="utf-8",
            )
            launcher = TeleopLauncher(
                {
                    "enabled": True,
                    "repo_root": ".",
                    "arm_runtime_config": "arm_services.json",
                    "follow_start_script": "server/start_teleop_follow.sh",
                    "follow_check_script": "server/check_teleop_follow.sh",
                    "follow_stop_script": "server/stop_teleop_follow.sh",
                },
                config_dir=root,
            )
            steps: list[tuple[str, dict[str, str]]] = []
            follow_ready = False

            ready_endpoints = {
                "lead_host": "10.20.30.40",
                "lead_ports": {"50050": True, "50052": True},
                "follower_ports": {"50051": True, "50053": True},
                "lead_ready": True,
                "follower_ready": True,
            }

            def fake_check() -> dict:
                return {
                    "ok": follow_ready,
                    "tmux": follow_ready,
                    "state": "running" if follow_ready else "missing",
                    "heartbeat_age_s": 0.1 if follow_ready else None,
                    "error": "",
                }

            def fake_run_script(
                script: Path,
                env: dict[str, str],
                timeout_s: float,
            ) -> None:
                nonlocal follow_ready
                self.assertGreater(timeout_s, 0)
                steps.append((script.name, dict(env)))
                if script.name == "start_teleop_follow.sh":
                    follow_ready = True

            with mock.patch.object(
                launcher,
                "_check_follow",
                side_effect=fake_check,
            ), mock.patch.object(
                launcher,
                "_endpoints",
                return_value=ready_endpoints,
            ), mock.patch.object(
                launcher,
                "_stable_endpoints",
                return_value=ready_endpoints,
            ), mock.patch.object(
                launcher,
                "_run_script",
                side_effect=fake_run_script,
            ), mock.patch.object(
                launcher,
                "_desired_matches",
                return_value=True,
            ):
                response = launcher.start(
                    {
                        "confirm": "START_FOLLOW",
                        "area_clear": True,
                        "estop_ready": True,
                        "initial_pose_aligned": True,
                    }
                )
                self.assertTrue(response["ok"])
                deadline = time.monotonic() + 2.0
                while launcher.status()["busy"] and time.monotonic() < deadline:
                    time.sleep(0.01)
                status = launcher.status()

            self.assertTrue(status["running"])
            self.assertEqual(status["state"], "running")
            self.assertEqual(
                [step[0] for step in steps],
                ["start_teleop_follow.sh"],
            )
            for _, env in steps:
                self.assertEqual(env["LEAD_URL"], "10.20.30.40")
                self.assertEqual(env["FOLLOW_URL"], "localhost")
                self.assertEqual(env["LEFT_LEAD_PORT"], "50050")
                self.assertEqual(env["RIGHT_LEAD_PORT"], "50052")
                self.assertEqual(env["LEFT_PORT"], "50051")
                self.assertEqual(env["RIGHT_PORT"], "50053")

    def test_start_does_not_recover_or_arm_when_an_endpoint_is_missing(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm_services.json").write_text(
                json.dumps({"remote_host": "10.20.30.40"}),
                encoding="utf-8",
            )
            launcher = TeleopLauncher(
                {
                    "enabled": True,
                    "repo_root": ".",
                    "arm_runtime_config": "arm_services.json",
                },
                config_dir=root,
            )
            missing_endpoint = {
                "lead_host": "10.20.30.40",
                "lead_ports": {"50050": True, "50052": False},
                "follower_ports": {"50051": True, "50053": True},
                "lead_ready": False,
                "follower_ready": True,
            }
            with mock.patch.object(
                launcher,
                "_check_follow",
                return_value={
                    "ok": False,
                    "tmux": False,
                    "state": "missing",
                    "heartbeat_age_s": None,
                    "error": "",
                },
            ), mock.patch.object(
                launcher,
                "_endpoints",
                return_value=missing_endpoint,
            ), mock.patch.object(
                launcher,
                "_stable_endpoints",
                return_value=missing_endpoint,
            ), mock.patch.object(
                launcher,
                "_run_script",
            ) as run_script:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "all lead and follower endpoints",
                ):
                    launcher.start(
                        {
                            "confirm": "START_FOLLOW",
                            "area_clear": True,
                            "estop_ready": True,
                            "initial_pose_aligned": True,
                        }
                    )
            run_script.assert_not_called()


class ActStartPoseDiagnosticTests(unittest.TestCase):
    def test_current_pose_reports_small_right_joint_1_warning(self) -> None:
        current = [
            -0.817693, -0.060082, 0.007820, -1.722782, -0.103418,
            1.597124, 0.000385, 0.806630, -0.163081, 0.058938,
            1.801137, 0.065270, -1.781376, 0.000847,
        ]
        result = describe_act_start_pose(current)
        self.assertFalse(result["within_training_start_range"])
        self.assertFalse(result["blocking"])
        self.assertTrue(result["diagnostic_only"])
        self.assertEqual(len(result["out_of_range"]), 1)
        warning = result["out_of_range"][0]
        self.assertEqual(warning["name"], "right_joint_1")
        self.assertEqual(warning["direction"], "below")
        self.assertAlmostEqual(warning["distance_to_range"], 0.0095369374)

    def test_pose_inside_reference_range_has_no_warning(self) -> None:
        inside = [
            -0.814138, -0.078300, 0.015603, -1.583656, -0.091446,
            1.459777, 0.000329, 0.860349, -0.119666, 0.034733,
            1.728223, 0.086934, -1.750825, 0.000930,
        ]
        result = describe_act_start_pose(inside)
        self.assertTrue(result["within_training_start_range"])
        self.assertIsNone(result["warning"])
        self.assertEqual(result["out_of_range"], [])


class RuntimePoseDeletionTests(unittest.TestCase):
    def test_delete_runtime_pose_requires_exact_confirmation(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.runtime_parameters = mock.Mock()
        snapshot = {"revision": 7, "poses": {"left": {}, "right": {}}}
        app.runtime_parameters.delete_pose.return_value = snapshot

        result = app.delete_runtime_pose(
            {
                "arm": "left",
                "name": "temporary_pose",
                "confirm_name": "temporary_pose",
            }
        )

        self.assertEqual(result["deleted"], {"arm": "left", "name": "temporary_pose"})
        self.assertIs(result["runtime_parameters"], snapshot)
        app.runtime_parameters.delete_pose.assert_called_once_with(
            "left",
            "temporary_pose",
            source="web_operator_delete_pose",
        )

        with self.assertRaisesRegex(ValueError, "exactly match"):
            app.delete_runtime_pose(
                {
                    "arm": "left",
                    "name": "temporary_pose",
                    "confirm_name": "another_pose",
                }
            )
        with self.assertRaisesRegex(ValueError, "request fields must be exactly"):
            app.delete_runtime_pose({"arm": "left", "name": "temporary_pose"})
        self.assertEqual(app.runtime_parameters.delete_pose.call_count, 1)


class PackagingConsoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.console = RunningConsole()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.console.close()

    def test_defaults_are_dedicated_loopback_8899(self) -> None:
        self.assertEqual(DEFAULT_BIND, "127.0.0.1")
        self.assertEqual(DEFAULT_PORT, 8899)
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["server"], {"bind": "127.0.0.1", "port": 8899})
        self.assertEqual(config["camera"]["intrinsics_resolution"], [1280, 720])
        intrinsics = np.asarray(config["camera"]["intrinsics"], dtype=np.float64)
        self.assertEqual(intrinsics.shape, (3, 3))
        self.assertAlmostEqual(intrinsics[0, 0], 907.9211813442888)
        self.assertAlmostEqual(intrinsics[1, 1], 907.4159645304343)
        self.assertAlmostEqual(intrinsics[0, 2], 638.3147247799354)
        self.assertAlmostEqual(intrinsics[1, 2], 381.14137920472274)
        self.assertEqual(
            config["camera"]["intrinsics_source"],
            "checkerboard_calibration",
        )
        self.assertIn(
            "front_realsense_420222072569_1280x720_checkerboard_20260731.json",
            config["camera"]["intrinsics_calibration"],
        )
        self.assertTrue(config["camera"]["shared_when_available"])
        self.assertEqual(config["camera"]["mode"], "shared_memory")
        self.assertEqual(
            config["camera"]["shared_memory_root"],
            "/dev/shm/ruc-video",
        )
        self.assertEqual(
            len(
                config["camera"]["depth_to_color_extrinsics"][
                    "rotation_row_major"
                ]
            ),
            9,
        )
        self.assertEqual(
            len(
                config["camera"]["depth_to_color_extrinsics"][
                    "translation_m"
                ]
            ),
            3,
        )
        self.assertTrue(config["camera"]["profile_approved"])
        self.assertIn(
            "cam_to_leftbase_handeye_left_01_1280x720_20260731.json",
            config["camera"]["cam_to_left_path"],
        )
        self.assertEqual(
            config["fixed_suction_axis"]["enabled"],
            config["fixed_suction_axis"]["calibrated"],
        )
        if config["fixed_suction_axis"]["enabled"]:
            self.assertEqual(
                len(config["fixed_suction_axis"]["axis_local_xyz"]), 3
            )
            self.assertEqual(
                len(config["fixed_suction_axis"]["approach_local_xyz"]), 3
            )
        self.assertEqual(
            config["fixed_suction_axis"]["cup_center_spacing_mm"],
            50.0,
        )
        self.assertTrue(
            config["task_profiles"]["task1"]["recognition_ready"]
        )
        self.assertTrue(
            config["task_profiles"]["task2"]["recognition_ready"]
        )
        self.assertTrue(config["task2_pick"]["enabled"])
        self.assertAlmostEqual(
            config["task2_pick"]["station_reference_left_base_xy_m"][0],
            0.25606636,
        )
        self.assertEqual(config["task2_pick"]["min_depth_valid_ratio"], 0.4)
        self.assertTrue(
            config["task_profiles"]["task3"]["recognition_ready"]
        )
        self.assertTrue(
            config["task_profiles"]["task3"]["suction_skill_ready"]
        )
        self.assertTrue(config["task3_pick"]["enabled"])
        self.assertEqual(config["task3_pick"]["table_surface_z_m"], 0.0)
        self.assertEqual(
            config["task_profiles"]["task1"]["initial_total_count"], 27
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["process_target_count"], 18
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["maximum_visible_instances"], 9
        )
        self.assertFalse(
            config["task_profiles"]["task1"][
                "surface_grid_completion_enabled"
            ]
        )
        self.assertFalse(
            config["task_profiles"]["task1"][
                "adaptive_reject_glare_matches"
            ]
        )
        self.assertEqual(
            config["task_profiles"]["task1"][
                "adaptive_homography_attempt_multiplier"
            ],
            4,
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["adaptive_sift_ratio"],
            0.86,
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["adaptive_slot_grid_shape"],
            [3, 3],
        )
        self.assertEqual(
            config["task_profiles"]["task1"][
                "adaptive_slot_polygon_norm"
            ],
            [
                [0.15, 0.325],
                [0.3525, 0.325],
                [0.338, 0.79],
                [0.11, 0.79],
            ],
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["adaptive_slot_sift_ratio"],
            0.95,
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["adaptive_slot_min_matches"],
            6,
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["adaptive_slot_min_inliers"],
            5,
        )
        self.assertEqual(
            config["task_profiles"]["task2"]["initial_total_count"], 4
        )
        task2_workflow = config["task_profiles"]["task2"]
        self.assertTrue(task2_workflow["workflow_locked"])
        self.assertEqual(task2_workflow["workflow_version"], "task2-v2.1")
        self.assertEqual(task2_workflow["workflow_cycle_target_count"], 2)
        self.assertEqual(len(task2_workflow["workflow_steps"]), 14)
        self.assertEqual(config["task2_workflow"]["joint_speed_profile"], "DEFAULT")
        self.assertEqual(config["task2_workflow"]["left_ready_pose"], "paper_init")
        self.assertEqual(config["task2_workflow"]["right_ready_pose"], "init_pose")
        self.assertEqual(config["task2_pick"]["pre_contact_clearance_m"], 0.025)
        self.assertEqual(
            config["task1_pick"]["tcp_calibration"],
            "calibration/task1_left_suction_tcp_counterclockwise_45.json",
        )
        self.assertEqual(
            config["task2_pick"]["target_offset_left_base_m"],
            [0.0, 0.01, 0.0],
        )
        self.assertEqual(
            config["task3_pick"]["target_offset_left_base_m"],
            [0.0, 0.01, 0.0],
        )
        self.assertEqual(
            config["task3_pick"]["tcp_calibration"],
            "calibration/left_suction_tcp.json",
        )
        self.assertTrue(config["task3_expand"]["enabled"])
        self.assertEqual(
            config["task3_expand"]["safe_height_m"],
            0.1712246813965305,
        )
        self.assertEqual(
            config["task3_expand"]["expand_pose_name"],
            "expand_box",
        )
        self.assertEqual(
            config["task3_expand"]["trajectory_path"],
            "/home/ubuntu/RUC-WONE/medicine_agentic/recordings/trajectories/"
            "trajectory_20260813_030953_expand_box_d343eb11",
        )
        self.assertEqual(config["task3_expand"]["replay_timeout_s"], 90.0)
        self.assertEqual(
            config["task3_expand"]["max_tracking_error_rad"],
            0.5,
        )
        self.assertEqual(
            config["trajectory_replay"]["calibration_episode_root"],
            "../recordings/trajectories",
        )
        self.assertEqual(
            config["trajectory_replay"]["max_tracking_error_rad"],
            0.5,
        )
        self.assertEqual(
            config["trajectory_replay"]["gripper_replay_arms"],
            ["right"],
        )
        self.assertEqual(
            config["trajectory_replay"]["gripper_feedback_every_n"],
            5,
        )
        self.assertEqual(
            config["trajectory_replay"]["max_gripper_tracking_error_m"],
            0.012,
        )
        self.assertEqual(
            config["trajectory_replay"]["recording_profiles"],
            {
                "act_20260815_044616_zhuangxiang_ba97f7be": {
                    "retain_frame_numbers_1_based": "odd",
                    "playback_speed_scale": 2.0,
                    "first_retained_frame_1_based": 55,
                    "suction_release_frame_1_based": 251,
                    "max_tracking_error_rad": None,
                    "max_gripper_tracking_error_m": None,
                }
            },
        )
        self.assertEqual(
            config["task1_fixed_trajectory_place"],
            {
                "enabled": True,
                "left_pose": "zhuangxiang",
                "right_pose": "zhuangxiang",
                "recording_id": "act_20260815_044616_zhuangxiang_ba97f7be",
                "joint_speed_profile": "DEFAULT",
                "replay_timeout_s": 90.0,
            },
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][5]["id"],
            "zr0_act1_insert_leaflet",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][4]["id"],
            "move_both_to_ready_pose",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][4]["implementation"],
            "ready",
        )
        self.assertIn(
            "left_watcher",
            task2_workflow["workflow_steps"][1]["label"],
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][6]["id"],
            "move_both_to_subtask2_init",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][7]["id"],
            "zr0_act2_insert_blister",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][8]["id"],
            "move_both_to_subtask3_init",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][9]["id"],
            "zr0_act3_close_carton",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][10]["id"],
            "reset_both_arms_after_close",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][11]["id"],
            "detect_shipping_box",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][12]["id"],
            "place_carton_in_shipping_box",
        )
        self.assertEqual(
            task2_workflow["workflow_steps"][12]["implementation"],
            "ready",
        )
        for index, model_slot in (
            (5, "task2.act1.insert_leaflet"),
            (7, "task2.act2.insert_blister"),
            (9, "task2.act3.close_carton"),
        ):
            step = task2_workflow["workflow_steps"][index]
            self.assertEqual(step["implementation"], "reserved_model_slot")
            self.assertEqual(step["model_slot"], model_slot)
            self.assertTrue(step["future_endpoint"].startswith("/api/skills/task2/"))
        self.assertIn(
            "subtask2_left_init",
            task2_workflow["workflow_steps"][6]["label"],
        )
        self.assertIn(
            "subtask2_right_init",
            task2_workflow["workflow_steps"][6]["label"],
        )
        self.assertIn(
            "subtask3_left_init",
            task2_workflow["workflow_steps"][8]["label"],
        )
        self.assertIn(
            "subtask3_right_init",
            task2_workflow["workflow_steps"][8]["label"],
        )
        placement = config["task2_workflow"]["shipping_box_placement"]
        self.assertNotIn("enabled", placement)
        self.assertEqual(placement["orientation_pose"], "system_home")
        self.assertEqual(placement["carton_footprint_m"], [0.13, 0.085])
        self.assertNotIn("release_tcp_clearance_above_rim_m", placement)
        self.assertNotIn("post_release_lift_flange_z_m", placement)
        shipping_detection = config["task2_workflow"]["shipping_box_detection"]
        self.assertFalse(shipping_detection["require_cavity_depth"])
        self.assertFalse(shipping_detection["task3_require_cavity_depth"])
        self.assertEqual(shipping_detection["fixed_rim_z_left_base_m"], 0.065)
        self.assertEqual(
            config["task_profiles"]["task3"]["initial_total_count"], 4
        )
        self.assertEqual(
            config["task_profiles"]["task2"][
                "surface_z_range_left_base_m"
            ],
            [0.010, 0.040],
        )
        self.assertEqual(
            config["task_profiles"]["task3"][
                "surface_z_range_left_base_m"
            ],
            None,
        )
        for task_id in ("task2", "task3"):
            profile = config["task_profiles"][task_id]
            self.assertEqual(profile["minimum_pickable_instances"], 1)
            self.assertTrue(
                profile["dual_suction_enforce_required_face_types"]
            )
            self.assertEqual(profile["required_face_types"], ["front_large"])
            self.assertTrue(profile["require_individual_front_similarity"])
        self.assertEqual(
            config["task_profiles"]["task2"]["include_roi_norm"],
            [0.42, 0.52, 0.72, 0.82],
        )
        self.assertTrue(
            config["task_profiles"]["task2"]["require_station_radius_gate"]
        )
        self.assertEqual(
            config["task_profiles"]["task2"]["front_similarity_slot_mode"],
            "template",
        )
        self.assertEqual(
            config["task_profiles"]["task2"][
                "front_similarity_template_min_score"
            ],
            0.46,
        )
        self.assertEqual(
            config["task_profiles"]["task3"]["include_roi_norm"],
            [0.03, 0.55, 0.28, 0.82],
        )
        self.assertEqual(
            config["task_profiles"]["task3"]["exclude_roi_norms"],
            [],
        )
        self.assertEqual(
            config["task_profiles"]["task3"][
                "required_long_axis_orientation_image_deg"
            ],
            90.0,
        )
        self.assertTrue(
            config["task_profiles"]["task3"]["physical_instance_gate"][
                "enabled"
            ]
        )
        self.assertEqual(
            config["task_profiles"]["task3"]["adaptive_allowed_counts"],
            [1, 2],
        )
        self.assertFalse(
            config["task_profiles"]["task3"]["adaptive_recovery_enabled"]
        )
        self.assertEqual(
            config["task_profiles"]["task3"]["adaptive_minimum_quad_fill"],
            0.7,
        )
        self.assertFalse(
            config["task_profiles"]["task3"][
                "drop_unsupported_grid_cells"
            ]
        )
        self.assertTrue(
            config["task_profiles"]["task3"][
                "require_front_panel_verification"
            ]
        )
        self.assertEqual(
            config["task_profiles"]["task3"][
                "required_consistent_detections"
            ],
            2,
        )
        self.assertEqual(
            config["task_profiles"]["task3"]["detection_attempts"],
            5,
        )
        self.assertEqual(
            config["task_profiles"]["task3"][
                "dual_suction_safety_margin_mm"
            ],
            2.0,
        )
        self.assertTrue(config["teleop_launcher"]["enabled"])
        self.assertEqual(
            config["teleop_launcher"]["arm_runtime_config"],
            "../../web_console/runtime/arm_services.json",
        )
        self.assertEqual(
            config["teleop_launcher"]["follow_stop_script"],
            "server/stop_teleop_follow.sh",
        )
        self.assertTrue(config["cartesian_jog"]["enabled"])
        self.assertFalse(config["cartesian_jog"]["dry_run"])
        self.assertEqual(
            config["cartesian_jog"]["enable_token"],
            "ENABLE_LEFT_CARTESIAN_JOG",
        )
        self.assertEqual(
            config["cartesian_jog"]["workspace_profile"],
            "task1_folded_carton_pick",
        )
        self.assertEqual(
            config["cartesian_jog"]["capture_workspace"]["z_min"],
            0.0,
        )
        self.assertEqual(
            config["cartesian_jog"]["workspace"]["z_min"],
            0.005,
        )
        self.assertEqual(
            config["cartesian_jog"][
                "calibrated_workspace_floor_z_m_by_profile"
            ]["task1_pick"],
            -0.10,
        )
        self.assertEqual(
            config["task1_pick"]["calibrated_workspace_profile"],
            "task1_pick",
        )
        self.assertFalse(
            config["cartesian_jog"]["manual_jog_workspace_enforced"]
        )
        self.assertEqual(
            config["task_profiles"]["task3"]["required_face_types"],
            ["front_large"],
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["include_roi_norm"],
            [0.05, 0.25, 0.37, 0.85],
        )
        self.assertEqual(
            config["task_profiles"]["task1"]["include_polygon_norm"],
            [[0.15, 0.325], [0.3525, 0.325], [0.338, 0.79], [0.11, 0.79]],
        )
        self.assertGreaterEqual(
            config["detector"]["provider_options"][
                "motif_max_peaks_per_variant"
            ],
            9,
        )
        self.assertEqual(
            config["cartesian_jog"]["workspace"]["z_max"],
            0.5,
        )
        self.assertEqual(
            config["cartesian_jog"]["capture_workspace"]["z_max"],
            0.5,
        )
        self.assertEqual(
            config["cartesian_jog"]["workspace"]["y_max"],
            0.25,
        )
        self.assertFalse(
            config["cartesian_jog"]["workspace"]["enforce_xy"]
        )
        self.assertFalse(
            config["cartesian_jog"]["capture_workspace"]["enforce_xy"]
        )
        self.assertLess(
            config["cartesian_jog"]["capture_workspace"]["z_min"],
            config["cartesian_jog"]["workspace"]["z_min"],
        )
        self.assertEqual(
            config["cartesian_jog"]["max_downward_step_mm"],
            5,
        )
        self.assertEqual(
            config["cartesian_jog"]["max_position_error_m"],
            0.001,
        )
        self.assertEqual(
            config["cartesian_jog"]["safe_vertical_pose"]["restore_token"],
            "RESTORE_LEFT_SAFE_VERTICAL",
        )
        self.assertAlmostEqual(
            config["cartesian_jog"]["safe_vertical_pose"]["position_m"][2],
            0.1512246813965305,
        )
        self.assertAlmostEqual(
            config["cartesian_jog"]["safe_vertical_pose"]["transit_z_m"],
            0.1712246813965305,
        )
        self.assertEqual(
            config["shared_pick"]["target_offset_left_base_m"],
            [0.0, -0.02, 0.0],
        )
        self.assertGreaterEqual(
            config["detector"]["min_edge_clearance_px"],
            config["detector"]["cup_radius_px"] + 4,
        )
        self.assertEqual(
            config["task1_pick"]["contact_flange_z_m_by_layer"],
            {
                "1": -0.085,
                "2": -0.060,
                "3": -0.035,
            },
        )
        self.assertEqual(
            config["task1_pick"]["layer_estimation"][
                "layer_minimum_height_m"
            ],
            {"2": 0.045, "3": 0.074},
        )
        self.assertEqual(config["task1_pick"]["test_lift_m"], 0.10)
        self.assertTrue(
            config["task1_pick"]["layer_estimation"]["enabled"]
        )
        self.assertEqual(
            config["task1_pick"]["layer_estimation"]["layer_height_m"],
            0.025,
        )

    def test_default_camera_profile_is_1280x720_at_30fps(self) -> None:
        camera = create_camera(
            camera_config(self.console.image_path.name),
            config_dir=self.console.root,
        )
        try:
            profile = camera.profile()
            self.assertEqual(
                profile["color"],
                {"width": 1280, "height": 720, "fps": 30, "format": "bgr8"},
            )
            self.assertEqual(
                profile["depth"],
                {"width": 1280, "height": 720, "fps": 30, "format": "z16"},
            )
            self.assertEqual(profile["aligned_to"], "color")
            self.assertFalse(profile["profile_approved"])
            camera.capture()
            self.assertFalse(camera.live_rgbd_is_fresh())
        finally:
            camera.close()

    def test_fixed_suction_axis_status_is_read_only_and_not_ready(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/calibrations/fixed-suction-axis/status",
        )
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        calibration = payload["fixed_suction_axis"]
        self.assertFalse(calibration["ready"])
        self.assertFalse(calibration["enabled"])
        self.assertIn("calibration_required", calibration["blockers"])

    def test_fixed_suction_axis_calibration_uses_xy_and_ignores_height(self) -> None:
        samples = {
            "A": {
                "flange_position_left_base_m": [0.35, -0.20, 0.18],
                "flange_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "B": {
                "flange_position_left_base_m": [0.30, -0.20, 0.03],
                "flange_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        }
        preview = compute_fixed_suction_axis_preview(samples)
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertTrue(preview["valid"])
        self.assertTrue(preview["z_ignored"])
        self.assertAlmostEqual(preview["measured_spacing_mm"], 50.0)
        self.assertAlmostEqual(preview["ignored_z_delta_mm"], 150.0)
        self.assertEqual(preview["checks"], {
            "planar_spacing": True,
            "orientation": True,
        })
        np.testing.assert_allclose(
            preview["axis_local_xyz"], [1.0, 0.0, 0.0], atol=1e-9
        )

    def test_task_profiles_are_explicit_and_isolated(self) -> None:
        status, _, body = self.console.request("GET", "/api/tasks/profiles")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["default_task"], "task1")
        self.assertTrue(payload["profiles"]["task1"]["recognition_ready"])
        self.assertFalse(payload["profiles"]["task2"]["recognition_ready"])
        self.assertFalse(payload["profiles"]["task3"]["recognition_ready"])
        self.assertEqual(
            payload["profiles"]["task3"]["height_policy"],
            "tabletop_flat_carton",
        )

    def test_configured_intrinsics_are_validated_and_exposed(self) -> None:
        config = camera_config(self.console.image_path.name)
        config.update(
            {
                "intrinsics": [
                    [912.0800170898438, 0.0, 643.6144409179688],
                    [0.0, 911.6617431640625, 366.7373962402344],
                    [0.0, 0.0, 1.0],
                ],
                "intrinsics_resolution": [1280, 720],
                "intrinsics_source": "test_factory_profile_scaled",
                "source_color_profile": {
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "format": "bgr8",
                },
                "source_intrinsics": [
                    [912.0800170898438, 0.0, 643.6144409179688],
                    [0.0, 911.6617431640625, 366.7373962402344],
                    [0.0, 0.0, 1.0],
                ],
                "distortion_model": "inverse_brown_conrady",
                "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
        )
        camera = create_camera(config, config_dir=self.console.root)
        try:
            profile = camera.profile()
            self.assertEqual(profile["intrinsics"], config["intrinsics"])
            self.assertEqual(profile["intrinsics_resolution"], [1280, 720])
            self.assertEqual(
                profile["source_color_profile"],
                config["source_color_profile"],
            )
            self.assertEqual(
                profile["distortion_model"], "inverse_brown_conrady"
            )
            self.assertEqual(
                profile["distortion_coefficients"],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            )
            self.assertFalse(profile["profile_approved"])
        finally:
            camera.close()

    def test_default_static_resources_exist_and_are_referenced(self) -> None:
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        configured = Path(config["static_dir"])
        static_dir = (
            configured
            if configured.is_absolute()
            else DEFAULT_CONFIG_PATH.parent / configured
        ).resolve()
        index_path = static_dir / "index.html"
        app_path = static_dir / "app.js"
        styles_path = static_dir / "styles.css"
        engineering_app_path = static_dir / "engineering.js"
        engineering_styles_path = static_dir / "engineering.css"
        for path in (
            index_path,
            app_path,
            styles_path,
            engineering_app_path,
            engineering_styles_path,
        ):
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file(), f"missing static resource: {path}")
        index = index_path.read_text(encoding="utf-8")
        app = app_path.read_text(encoding="utf-8")
        engineering_app = engineering_app_path.read_text(encoding="utf-8")
        self.assertIn('src="/app.js?v=20260813-zr0-multimodel-v1"', index)
        self.assertIn('id="act-model-select"', index)
        self.assertIn('actModel: "/api/act/model"', app)
        self.assertIn("function switchActModel()", app)
        self.assertIn('href="/styles.css"', index)
        self.assertIn('src="/engineering.js"', index)
        self.assertIn('href="/engineering.css"', index)
        self.assertIn('deletePose: "/api/runtime-poses/delete"', engineering_app)
        self.assertIn('data-pose-delete=', engineering_app)
        self.assertIn('confirm_name: name', engineering_app)
        self.assertIn('id="engineering-parameters"', index)
        self.assertIn('id="console-run-mode"', index)
        self.assertIn('id="console-engineering-mode"', index)
        self.assertIn('data-console-surface="run"', index)
        self.assertIn('data-console-surface="engineering"', index)
        self.assertIn('data-engineering-tool-button="parameters"', index)
        self.assertIn('data-engineering-tool-button="motion"', index)
        self.assertIn('data-engineering-tool-button="teleop"', index)
        self.assertIn('data-engineering-tool="parameters"', index)
        self.assertIn('data-engineering-tool="motion"', index)
        self.assertIn('data-engineering-tool="teleop"', index)
        self.assertIn('id="pose-read"', index)
        self.assertIn('id="pose-save"', index)
        self.assertIn('id="parameter-show-all"', index)
        self.assertIn('id="parameter-overview-body"', index)
        self.assertIn('id="start-recording"', index)
        self.assertIn('id="stop-recording"', index)
        self.assertIn('id="recording-list"', index)
        self.assertIn('id="copy-recording-path"', index)
        self.assertIn('id="start-teleop"', index)
        self.assertIn('id="stop-teleop"', index)
        self.assertIn('id="jog-capture-orientation"', index)
        self.assertIn('id="jog-enable"', index)
        self.assertIn('name="jog-step" value="5" checked', index)
        self.assertIn('name="jog-step" value="10"', index)
        self.assertIn('id="jog-restore-safe"', index)
        self.assertIn("左臂安全竖直位", index)
        self.assertIn('id="left-arm-reset-home"', index)
        self.assertIn('id="right-arm-reset-home"', index)
        self.assertIn('id="workflow-run-step"', index)
        self.assertIn('id="workflow-previous"', index)
        self.assertIn('id="workflow-next"', index)
        self.assertIn('id="workflow-reset-progress"', index)
        self.assertIn('id="workflow-auto-panel"', index)
        self.assertIn('id="workflow-auto-start"', index)
        self.assertIn('id="workflow-auto-end"', index)
        self.assertIn('id="workflow-run-range"', index)
        self.assertIn('id="workflow-stop-range"', index)
        self.assertNotIn('id="fixed-axis-lock-marker"', index)
        self.assertNotIn('id="fixed-axis-sample-a"', index)
        self.assertNotIn('id="fixed-axis-sample-b"', index)
        self.assertNotIn('id="fixed-axis-commit"', index)
        self.assertNotIn("共享门禁", index)
        self.assertNotIn("相机标准", index)
        self.assertNotIn('value="calibration_left"', index)
        self.assertNotIn('value="calibration_right"', index)
        self.assertNotIn('value="projection_validation"', index)
        self.assertIn("Task2 单层成盒药盒吸取技能已接入", app)
        self.assertIn("Task2 固定工作流 · v2.1", app)
        self.assertIn('1: "moveTask2Watcher"', app)
        self.assertIn('2: "detectTask2Carton"', app)
        self.assertIn('4: "moveBothInit"', app)
        self.assertIn('6: "moveSubtask2Init"', app)
        self.assertIn('8: "moveSubtask3Init"', app)
        self.assertIn(
            'task2ResetBothHome: "/api/skills/task2/reset-both-home"',
            app,
        )
        self.assertIn(
            'task2MoveReadyPoses: "/api/skills/task2/move-ready-poses"',
            app,
        )
        self.assertIn(
            '"/api/skills/task2/move-subtask2-init-poses"',
            app,
        )
        self.assertIn(
            '"/api/skills/task2/move-subtask3-init-poses"',
            app,
        )
        self.assertIn(
            'task2MoveWatcherPose: "/api/skills/task2/move-watcher-pose"',
            app,
        )
        self.assertIn(
            'task2DetectCarton: "/api/skills/task2/detect-carton"',
            app,
        )
        self.assertIn(
            'task2PlaceShippingBox: "/api/skills/task2/place-shipping-box"',
            app,
        )
        self.assertIn('12: "placeShippingBox"', app)
        self.assertIn("function task2PlacementBlockerLabel", app)
        self.assertIn("请先执行第12步识别纸箱", app)
        self.assertIn("API.task2PlaceShippingBoxPreflight", app)
        self.assertIn(
            "接口可用 · 自动预检第12步缓存与运动安全条件",
            app,
        )
        self.assertIn("左 paper_init / 右 init_pose", app)
        self.assertIn(
            "左 subtask2_left_init / 右 subtask2_right_init",
            app,
        )
        self.assertIn(
            "左 subtask3_left_init / 右 subtask3_right_init",
            app,
        )
        self.assertNotIn("正在识别；成功后将不间断直接吸取", app)
        self.assertIn("模型预留 · 暂未接入", app)
        self.assertIn("medicine-pack-console-mode", app)
        self.assertIn("medicine-pack-engineering-tool", app)
        self.assertIn('task2ObserveCarton: "/api/skills/task2/observe-carton"', app)
        self.assertIn(
            'task1MoveWatcherPose: "/api/skills/task1/move-watcher-pose"',
            app,
        )
        self.assertIn(
            'task1DetectCarton: "/api/skills/task1/detect-carton"',
            app,
        )
        self.assertIn(
            'task1PickStagedTop: "/api/skills/task1/pick-staged-carton-top"',
            app,
        )
        self.assertIn(
            '0: "moveTask1Watcher"',
            app,
        )
        self.assertIn(
            '1: "detectTask1Carton"',
            app,
        )
        self.assertIn(
            '2: "pickCached"',
            app,
        )
        self.assertIn(
            '4: "pickTask1StagedTop"',
            app,
        )
        self.assertIn("药盒装箱 · 单盒 7 步流程", app)
        self.assertIn("条码平行条纹", app)
        self.assertIn(
            "两臂并行启动并等待双回执",
            app,
        )
        self.assertIn(
            "第一次并行识别药盒与纸箱，生成顺时针90°的RGB-D 20槽并持久化",
            app,
        )
        self.assertIn('8: "resetBoth"', app)
        self.assertIn('13: "resetBoth"', app)
        self.assertIn('3: "expandCarton"', app)
        self.assertIn(
            'task3ExpandCarton: "/api/skills/task3/expand-carton"',
            app,
        )
        self.assertIn("安全抬升 → left.expand_box → 轨迹重放", app)
        self.assertIn("左 left_box_watcher / 右初始位姿", app)
        self.assertIn("只识别 130×85 mm 小熊正面", app)
        self.assertIn("扁平纸板高度不作硬门禁", app)
        self.assertIn("上方纸箱区域不作为目标", app)
        self.assertIn("正在前往 left_box_watcher 识别并缓存 Task3 目标", app)
        self.assertIn(
            'if (normalized?.task_id === appState.activeTask)',
            app,
        )
        self.assertIn(
            'task2PickCachedCarton: "/api/skills/task2/pick-cached-carton"',
            app,
        )
        self.assertIn("function executeWorkflowStep", app)
        self.assertIn("function assertWorkflowStepSucceeded", app)
        self.assertIn("function renderTask2WorkflowStatus", app)
        self.assertIn("右臂已校验；左臂正在前往 left_watcher", app)
        self.assertIn(
            'payload.result?.motions?.left?.executed !== true',
            app,
        )
        self.assertIn("接口可用 · 右臂成功后再执行左臂", app)
        self.assertIn("function selectedWorkflowRange", app)
        self.assertIn("async function executeWorkflowRange", app)
        self.assertIn("function requestWorkflowRangeStop", app)
        self.assertIn("每步成功后才继续", index)
        self.assertIn("任一步失败都会立即停止", app)
        self.assertIn("workflowAutoStopRequested", app)
        self.assertIn("ZR-0（ACT1）放说明书", app)
        self.assertIn("ZR-0（ACT2）放药板", app)
        self.assertIn("ZR-0（ACT3）关盒", app)
        self.assertIn(
            'el["jog-restore-safe"]?.addEventListener(',
            app,
        )
        self.assertIn("CARTESIAN_JOG_HOLD_REPEAT_DELAY_MS = 30", app)
        self.assertIn("effectiveCartesianJogStep", app)
        self.assertIn('id="jog-suction-released"', index)
        self.assertIn('id="jog-hold-disable"', index)
        self.assertIn('id="suction-on"', index)
        self.assertIn('id="suction-off"', index)
        self.assertIn('id="run-task1-pick"', index)
        self.assertIn('id="task1-pick-detail"', index)
        self.assertIn("姿态捕获范围（只读）", index)
        self.assertIn("XY 交给 IK 与关节限位判断", index)
        self.assertIn("固定姿态 XYZ 按住连续点动", index)
        self.assertIn("最近一次视觉结果", index)
        self.assertIn(
            "点击识别或开始执行任务后，才会获取并更新药盒坐标",
            index,
        )
        self.assertIn('recordingStart: "/api/recordings/start"', app)
        self.assertIn('recordingStop: "/api/recordings/stop"', app)
        self.assertIn('recordingDelete: "/api/recordings/delete"', app)
        self.assertIn("FRAME_INTERVAL_MS = 100", app)
        self.assertIn("RECORDING_FRAME_INTERVAL_MS = 200", app)
        self.assertIn("BACKGROUND_FRAME_INTERVAL_MS = 1000", app)
        self.assertIn("function runExclusivePoll", app)
        self.assertIn('runExclusivePoll("recording"', app)
        self.assertIn('runExclusivePoll("actRollout"', app)
        self.assertIn("pollInFlight", app)
        self.assertIn('document.addEventListener("visibilitychange"', app)
        self.assertIn('button.addEventListener("pointerdown"', app)
        self.assertIn('button.addEventListener("pointerup"', app)
        self.assertIn('button.addEventListener("pointercancel"', app)
        self.assertIn('button.addEventListener("lostpointercapture"', app)
        self.assertIn('window.addEventListener("pointerup"', app)
        self.assertIn('window.addEventListener("pointercancel"', app)
        self.assertIn('window.addEventListener("blur"', app)
        self.assertIn('window.addEventListener("pagehide"', app)
        self.assertIn('window.addEventListener("offline"', app)
        self.assertIn("if (document.hidden) {", app)
        self.assertIn('stopCartesianJogHold("页面进入后台")', app)
        self.assertIn("function stopCartesianJogHold", app)
        self.assertIn("await moveCartesianJog(hold.axis, hold.direction", app)
        self.assertIn("appState.cartesianJogBusy", app)
        self.assertIn("function requestCameraFrame", app)
        self.assertIn("frame_request", app)
        self.assertIn("data-copy-recording-index", app)
        self.assertIn("data-delete-recording-index", app)
        self.assertIn("function deleteRecording", app)
        self.assertIn('teleopStatus: "/api/teleop/status"', app)
        self.assertIn("cartesianJogSafetyContractValid", app)
        self.assertIn("safety.direct_cartesian_step_api === true", app)
        self.assertIn('teleopStart: "/api/teleop/start"', app)
        self.assertIn('teleopStop: "/api/teleop/stop"', app)
        self.assertIn(
            'teleopHardRestart: "/api/teleop/hard-restart"',
            app,
        )
        self.assertIn(
            'el["hard-restart-teleop"].addEventListener("click", hardRestartTeleop)',
            app,
        )
        self.assertIn('confirm: "HARD_RESTART_TELEOP"', app)
        self.assertIn("master_arms_stable: true", app)
        self.assertIn(
            'cartesianJogStatus: "/api/cartesian-jog/status"',
            app,
        )
        self.assertIn(
            'cartesianJogMove: "/api/cartesian-jog/move"',
            app,
        )
        self.assertIn(
            'cartesianJogQuickEnable: "/api/cartesian-jog/quick-enable"',
            app,
        )
        self.assertIn(
            'cartesianJogRestoreSafe: "/api/cartesian-jog/restore-safe-vertical"',
            app,
        )
        self.assertIn('suctionStatus: "/api/suction/status"', app)
        self.assertIn('suction: "/api/suction"', app)
        self.assertIn('task1PickStatus: "/api/task1/pick/status"', app)
        self.assertIn('task1Pick: "/api/task1/pick"', app)
        self.assertIn(
            'task1PickSkill: "/api/skills/task1/pick-carton"',
            app,
        )
        self.assertIn(
            'task1WatchDetectPick: "/api/skills/task1/watch-detect-pick"',
            app,
        )
        self.assertIn(
            'task2WatchDetectPick: "/api/skills/task2/watch-detect-pick"',
            app,
        )
        self.assertIn(
            'task3WatchDetectPick: "/api/skills/task3/watch-detect-pick"',
            app,
        )
        self.assertIn('task2PickStatus: "/api/task2/pick/status"', app)
        self.assertIn(
            'task2PickSkill: "/api/skills/task2/pick-carton"',
            app,
        )
        self.assertIn(
            'leftArmResetHomeSkill: "/api/skills/left-arm/reset-home"',
            app,
        )
        self.assertIn(
            'rightArmResetHomeSkill: "/api/skills/right-arm/reset-home"',
            app,
        )
        self.assertIn("API.leftArmResetHomeSkill", app)
        self.assertIn("API.rightArmResetHomeSkill", app)
        self.assertIn("task1: API.task1WatchDetectPick", app)
        self.assertIn("task2: API.task2WatchDetectPick", app)
        self.assertIn("task3: API.task3WatchDetectPick", app)
        self.assertNotIn("scheduleDetectionTracking", app)
        self.assertNotIn("detectionTrackingTimer", app)
        self.assertNotIn("detectionTrackingRequest", app)
        self.assertIn("appState.detectionBusy ||", app)
        self.assertIn("body: JSON.stringify({})", app)
        self.assertIn("      90000", app)
        self.assertNotIn('confirm: "PICK_DETECTED_CARTON"', app)
        self.assertIn("姿态已记录 · 运动未启用", app)
        self.assertIn("const executionLabel = !enabled", app)
        self.assertNotIn("/api/cartesian-jog/hold-current", app)

    def test_all_tasks_auto_range_static_contract(self) -> None:
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        configured = Path(config["static_dir"])
        static_dir = (
            configured
            if configured.is_absolute()
            else DEFAULT_CONFIG_PATH.parent / configured
        ).resolve()
        index = (static_dir / "index.html").read_text(encoding="utf-8")
        app = (static_dir / "app.js").read_text(encoding="utf-8")

        for control_id in (
            "workflow-auto-panel",
            "workflow-auto-start",
            "workflow-auto-end",
            "workflow-run-range",
            "workflow-stop-range",
        ):
            self.assertIn(f'id="{control_id}"', index)
        self.assertIn("每步成功后才继续", index)
        self.assertIn("function selectedWorkflowRange", app)
        self.assertIn("async function executeWorkflowRange", app)
        self.assertIn("function requestWorkflowRangeStop", app)
        self.assertIn(
            "if (!workflowHandler(taskId, index)) unavailable.push(index + 1)",
            app,
        )
        self.assertIn("await performWorkflowStep(taskId, index)", app)
        self.assertIn("workflowAutoStopRequested", app)
        self.assertIn("任一步失败都会立即停止", app)
        self.assertIn("const profile = TASK_PROFILES[taskId]", app)
        self.assertIn("当前任务 · 严格串行 · 失败立即停止", index)
        self.assertNotIn("仅 Task2 · 严格串行", index)

    def test_run_layout_centers_front_camera_between_wrist_cameras(
        self,
    ) -> None:
        config = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        configured = Path(config["static_dir"])
        static_dir = (
            configured
            if configured.is_absolute()
            else DEFAULT_CONFIG_PATH.parent / configured
        ).resolve()
        styles = (static_dir / "styles.css").read_text(encoding="utf-8")

        self.assertIn("minmax(480px, 2fr)", styles)
        self.assertIn(
            'body[data-console-mode="run"] .camera-panel {\n  grid-column: 2;',
            styles,
        )
        self.assertIn(
            'body[data-console-mode="run"] .wrist-camera-rail {\n  display: contents;',
            styles,
        )
        self.assertIn(
            'body[data-console-mode="run"] .wrist-camera-card:first-of-type {\n'
            "  grid-column: 1;\n  grid-row: 1;",
            styles,
        )
        self.assertIn(
            'body[data-console-mode="run"] .wrist-camera-card:last-of-type {\n'
            "  grid-column: 3;\n  grid-row: 1;",
            styles,
        )
        self.assertIn("grid-column: 1 / -1;\n  grid-row: 2;", styles)
        self.assertIn("grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));", styles)
        self.assertIn("aspect-ratio: 16 / 9", styles)
        self.assertIn("object-fit: contain;\n  background: #010503;", styles)

    def test_health_exposes_read_only_safety_contract(self) -> None:
        status, headers, body = self.console.request("GET", "/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(headers["Connection"], "close")
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        payload = json.loads(body)
        self.assertEqual(
            payload["safety"],
            {
                "mode": "read_only",
                "dry_run": True,
                "motion_api": False,
                "teleop_enable_api": False,
                "cartesian_jog_api": False,
                "cartesian_jog_dry_run": True,
                "cartesian_jog_arm": "left",
                "direct_cartesian_step_api": False,
                "autonomous_motion_api": False,
                "act_rollout_api": False,
                "act_rollout_stop_semantics": "synchronous_hold_current",
                "bounded_task_skill_api": False,
                "bounded_task_skills": [
                    "task1.pick_carton",
                    "task2.pick_carton",
                    "task3.pick_flat_carton",
                    "left_arm.reset_home",
                    "right_arm.reset_home",
                    "act.rollout",
                ],
                "direct_joint_command_api": False,
                "suction_api": False,
                "chassis_api": False,
                "navigation_api": False,
            },
        )
        self.assertEqual(payload["server"]["bind"], "127.0.0.1")
        self.assertFalse(payload["cartesian_jog"]["available"])
        self.assertEqual(payload["cartesian_jog"]["state"], "disabled")

        status, _, body = self.console.request("GET", "/api/status")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        gates = {gate["id"]: gate for gate in payload["gates"]}
        self.assertTrue(gates["operator_teleop"]["passed"])
        self.assertFalse(payload["camera"]["live_rgbd"])
        self.assertFalse(gates["live_rgbd"]["passed"])
        self.assertTrue(gates["autonomous_motion_disabled"]["passed"])
        self.assertFalse(gates["camera_profile_approved"]["passed"])
        self.assertFalse(gates["detector_backend_ready"]["passed"])
        self.assertFalse(gates["reference_face_bank"]["passed"])

    def test_runtime_parameters_update_without_service_restart(self) -> None:
        status, _, body = self.console.request("GET", "/api/runtime-parameters")
        self.assertEqual(status, 200)
        before = json.loads(body)["runtime_parameters"]
        self.assertIn("poses", before)
        self.assertIn("task2", before["tasks"])

        status, _, body = self.console.request(
            "POST",
            "/api/runtime-parameters",
            body=json.dumps(
                {
                    "task_id": "task2",
                    "values": {"contact_flange_z_m": 0.031},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        updated = json.loads(body)["runtime_parameters"]
        self.assertEqual(updated["tasks"]["task2"]["contact_flange_z_m"], 0.031)
        self.assertGreater(updated["revision"], before["revision"])

    def test_current_pose_endpoint_is_read_only_and_unavailable_when_disabled(self) -> None:
        status, _, body = self.console.request("GET", "/api/current-pose?arm=left")
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_recording_routes_are_read_only_and_disabled_by_default(self) -> None:
        status, _, body = self.console.request("GET", "/api/recordings/status")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertFalse(payload["recording"]["enabled"])
        self.assertEqual(payload["recording"]["state"], "idle")

        status, _, body = self.console.request("GET", "/api/recordings")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["recordings"], [])

        episode = self.console.app.trajectory_recorder.output_dir / "delete_test"
        episode.mkdir(parents=True)
        (episode / "meta.json").write_text(
            json.dumps(
                {
                    "recording_id": "delete_test",
                    "label": "delete test",
                    "purpose": "trajectory_both",
                    "status": "completed",
                    "created_at": time.time(),
                }
            ),
            encoding="utf-8",
        )
        status, _, body = self.console.request(
            "POST",
            "/api/recordings/delete",
            body=json.dumps({"recording_id": "delete_test"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        deletion = json.loads(body)["deletion"]
        self.assertTrue(deletion["deleted"])
        self.assertTrue(deletion["recoverable"])
        self.assertFalse(episode.exists())

        status, _, body = self.console.request(
            "POST",
            "/api/recordings/delete",
            body=json.dumps({"recording_id": "missing"}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 404)

        status, _, body = self.console.request(
            "POST",
            "/api/recordings/start",
            body=json.dumps(
                {"label": "disabled", "purpose": "trajectory_left"}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_teleop_route_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request("GET", "/api/teleop/status")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertFalse(payload["teleop"]["enabled"])
        self.assertFalse(payload["teleop"]["running"])
        self.assertEqual(payload["teleop"]["state"], "disabled")

        status, _, body = self.console.request(
            "POST",
            "/api/teleop/start",
            body=json.dumps(
                {
                    "confirm": "START_FOLLOW",
                    "area_clear": True,
                    "estop_ready": True,
                    "initial_pose_aligned": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

        status, _, body = self.console.request(
            "POST",
            "/api/teleop/hard-restart",
            body=json.dumps(
                {
                    "confirm": "HARD_RESTART_TELEOP",
                    "area_clear": True,
                    "estop_ready": True,
                    "master_arms_stable": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_teleop_mutations_require_same_origin_json(self) -> None:
        status, _, _ = self.console.request(
            "POST",
            "/api/teleop/start",
            body=b"",
        )
        self.assertEqual(status, 415)

        status, _, _ = self.console.request(
            "POST",
            "/api/teleop/start",
            body=json.dumps(
                {
                    "confirm": "START_FOLLOW",
                    "area_clear": True,
                    "estop_ready": True,
                    "initial_pose_aligned": True,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Origin": "https://malicious.example",
            },
        )
        self.assertEqual(status, 403)

    def test_cartesian_jog_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/cartesian-jog/status",
        )
        self.assertEqual(status, 200)
        jog = json.loads(body)["cartesian_jog"]
        self.assertFalse(jog["available"])
        self.assertFalse(jog["enabled"])
        self.assertEqual(jog["endpoint"]["arm"], "left")
        self.assertEqual(jog["endpoint"]["port"], 50051)

        status, _, body = self.console.request(
            "POST",
            "/api/cartesian-jog/capture-orientation",
            body=json.dumps(
                {
                    "confirm": "CAPTURE_LEFT_SUCTION_DOWN",
                    "vertical_down_confirmed": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_stopped_teleop_history_latch_does_not_block_jog(self) -> None:
        self.assertFalse(
            PackagingConsoleApp._teleop_status_blocks_cartesian_jog(
                {
                    "running": False,
                    "busy": False,
                    "desired": False,
                    "follow": {"tmux": False},
                    "safety": {
                        "latched": True,
                        "reason": "web_console_shutdown_stop",
                    },
                }
            )
        )

    def test_real_follow_ownership_still_blocks_jog(self) -> None:
        for status in (
            {"running": True},
            {"busy": True},
            {"desired": True},
            {"follow": {"tmux": True}},
        ):
            with self.subTest(status=status):
                self.assertTrue(
                    PackagingConsoleApp._teleop_status_blocks_cartesian_jog(
                        status
                    )
                )

    def test_system_follow_desired_marker_blocks_jog(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            desired = root_path / "follow-desired"
            ready_paths = (
                root_path / "follow-left-ready.json",
                root_path / "follow-right-ready.json",
            )
            with (
                mock.patch(
                    "medicine_agentic.packaging_console."
                    "SYSTEM_FOLLOW_DESIRED_PATH",
                    desired,
                ),
                mock.patch(
                    "medicine_agentic.packaging_console."
                    "SYSTEM_FOLLOW_READY_PATHS",
                    ready_paths,
                ),
            ):
                self.assertFalse(
                    PackagingConsoleApp._system_follow_ownership_active()
                )
                desired.touch()
                self.assertTrue(
                    PackagingConsoleApp._system_follow_ownership_active()
                )

    def test_fresh_system_follow_ready_file_blocks_jog(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            desired = root_path / "follow-desired"
            left_ready = root_path / "follow-left-ready.json"
            ready_paths = (left_ready, root_path / "follow-right-ready.json")
            with (
                mock.patch(
                    "medicine_agentic.packaging_console."
                    "SYSTEM_FOLLOW_DESIRED_PATH",
                    desired,
                ),
                mock.patch(
                    "medicine_agentic.packaging_console."
                    "SYSTEM_FOLLOW_READY_PATHS",
                    ready_paths,
                ),
            ):
                left_ready.write_text("{}", encoding="utf-8")
                self.assertTrue(
                    PackagingConsoleApp._system_follow_ownership_active()
                )
                old_time = time.time() - 10.0
                left_ready.touch()
                with mock.patch(
                    "medicine_agentic.packaging_console.time.time",
                    return_value=old_time + 20.0,
                ):
                    self.assertFalse(
                        PackagingConsoleApp._system_follow_ownership_active()
                    )

    def test_cartesian_jog_mutations_require_same_origin_json(self) -> None:
        status, _, _ = self.console.request(
            "POST",
            "/api/cartesian-jog/disable",
            body=b"{}",
        )
        self.assertEqual(status, 415)

        status, _, _ = self.console.request(
            "POST",
            "/api/cartesian-jog/disable",
            body=b"{}",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://malicious.example",
            },
        )
        self.assertEqual(status, 403)

        status, _, body = self.console.request(
            "POST",
            "/api/cartesian-jog/hold-current",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 404)
        self.assertFalse(json.loads(body)["ok"])

    def test_suction_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request("GET", "/api/suction/status")
        self.assertEqual(status, 200)
        suction = json.loads(body)["suction"]
        self.assertFalse(suction["enabled"])
        self.assertFalse(suction["available"])

        status, _, body = self.console.request(
            "POST",
            "/api/suction",
            body=b'{"engaged":true}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_task1_pick_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request("GET", "/api/task1/pick/status")
        self.assertEqual(status, 200)
        pick = json.loads(body)["task1_pick"]
        self.assertFalse(pick["enabled"])
        self.assertFalse(pick["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/task1/pick",
            body=b'{"confirm":"PICK_DETECTED_CARTON"}',
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

        status, _, body = self.console.request(
            "GET",
            "/api/skills/task1/pick-carton",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task1.pick_carton")
        self.assertEqual(skill["input_schema"]["properties"], {})
        self.assertFalse(skill["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/skills/task1/pick-carton",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_task2_pick_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request("GET", "/api/task2/pick/status")
        self.assertEqual(status, 200)
        pick = json.loads(body)["task2_pick"]
        self.assertFalse(pick["enabled"])
        self.assertFalse(pick["ready"])

        status, _, body = self.console.request(
            "GET",
            "/api/skills/task2/pick-carton",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task2.pick_carton")
        self.assertEqual(
            skill["selection_policy"],
            "leftmost_image_x_valid_task2_front_carton",
        )
        self.assertFalse(skill["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/skills/task2/pick-carton",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_task2_step2_post_reaches_watcher_pose_handler(self) -> None:
        expected = {
            "ok": True,
            "result": {
                "operation": "task2_move_watcher_pose",
                "motions": {
                    "right": {"executed": True},
                    "left": {"executed": True},
                },
            },
            "skill": {
                "id": "task2.move_watcher_pose",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task2_watcher_pose_skill",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task2/move-watcher-pose",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task2_step3_post_reaches_detect_carton_handler(self) -> None:
        expected = {
            "ok": True,
            "detection": {"id": "task2-unit-detection", "target_ready": True},
            "skill": {
                "id": "task2.detect_carton",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task2_detect_carton_step",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task2/detect-carton",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task2_step3_get_describes_perception_only_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task2/detect-carton",
        )

        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task2.detect_carton")
        self.assertEqual(skill["affected_arms"], [])
        self.assertEqual(skill["method"], "POST")

    def test_task1_step1_post_reaches_watcher_pose_handler(self) -> None:
        expected = {
            "ok": True,
            "result": {"operation": "task1_move_watcher_pose"},
            "skill": {
                "id": "task1.move_watcher_pose",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task1_watcher_pose_skill",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task1/move-watcher-pose",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task1_step1_get_describes_parallel_watcher_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task1/move-watcher-pose",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task1.move_watcher_pose")
        self.assertEqual(skill["execution"], "parallel")
        self.assertEqual(skill["targets"]["left"], "left_watcher")
        self.assertEqual(skill["targets"]["right"], "system_initial_pose")

    def test_task1_step4_post_reaches_fixed_trajectory_handler(self) -> None:
        expected = {
            "ok": True,
            "result": {"operation": "task1_place_carton_fixed_trajectory"},
            "skill": {
                "id": "task1.place_carton_fixed_trajectory",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task1_fixed_trajectory_place_skill",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task1/place-carton-fixed-trajectory",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task1_step4_get_describes_parallel_pose_then_replay(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task1/place-carton-fixed-trajectory",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task1.place_carton_fixed_trajectory")
        self.assertEqual(skill["execution"], "parallel_poses_then_replay")
        self.assertEqual(skill["targets"], {
            "left": "zhuangxiang",
            "right": "zhuangxiang",
        })
        self.assertIn("recording_id", skill)

    def test_task1_step2_post_reaches_detect_carton_handler(self) -> None:
        expected = {
            "ok": True,
            "detection": {"id": "task1-unit-detection", "target_ready": True},
            "skill": {"id": "task1.detect_carton", "status": "succeeded"},
        }
        with mock.patch.object(
            self.console.app,
            "run_task1_detect_carton_step",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task1/detect-carton",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task1_step2_get_describes_perception_only_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task1/detect-carton",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task1.detect_carton")
        self.assertEqual(skill["affected_arms"], [])
        self.assertEqual(skill["method"], "POST")

    def test_task1_step5_post_reaches_staged_top_pick_handler(self) -> None:
        expected = {
            "ok": True,
            "detection": {
                "target_ready": True,
                "candidate": {"barcode_evidence": {"valid": True}},
            },
            "result": {
                "flange_center_alignment": "detected_top_geometric_center"
            },
            "skill": {
                "id": "task1.pick_staged_carton_top",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task1_pick_staged_top_step",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task1/pick-staged-carton-top",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task1_step5_get_describes_barcode_top_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task1/pick-staged-carton-top",
        )

        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task1.pick_staged_carton_top")
        self.assertEqual(skill["affected_arms"], ["left"])
        self.assertEqual(skill["method"], "POST")

    def test_task1_step6_post_reaches_parallel_place_handler(self) -> None:
        expected = {
            "ok": True,
            "result": {
                "execution": "parallel_left_place_right_retreat_and_system_home",
                "motions": {
                    "left": {"executed": True},
                    "right": {"executed": True},
                },
            },
            "skill": {
                "id": "task1.place_in_box",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task1_place_in_box_step",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task1/place-in-box",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task1_step6_get_describes_parallel_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task1/place-in-box",
        )

        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task1.place_in_box")
        self.assertEqual(
            skill["execution"],
            "parallel_left_place_right_retreat_and_system_home",
        )
        self.assertEqual(skill["affected_arms"], ["left", "right"])

    def test_task3_step1_post_reaches_watcher_pose_handler(self) -> None:
        expected = {
            "ok": True,
            "result": {"operation": "task3_move_watcher_pose"},
            "skill": {
                "id": "task3.move_watcher_pose",
                "status": "succeeded",
            },
        }
        with mock.patch.object(
            self.console.app,
            "run_task3_watcher_pose_skill",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task3/move-watcher-pose",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task3_step1_get_describes_watcher_pose_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task3/move-watcher-pose",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task3.move_watcher_pose")
        self.assertEqual(skill["targets"]["left"], "left_box_watcher")
        self.assertEqual(skill["targets"]["right"], "system_initial_pose")

    def test_task3_step2_post_reaches_detect_carton_handler(self) -> None:
        expected = {
            "ok": True,
            "detection": {"id": "task3-unit-detection", "target_ready": True},
            "skill": {"id": "task3.detect_carton", "status": "succeeded"},
        }
        with mock.patch.object(
            self.console.app,
            "run_task3_detect_carton_step",
            return_value=expected,
        ) as run_step:
            status, _, body = self.console.request(
                "POST",
                "/api/skills/task3/detect-carton",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), expected)
        run_step.assert_called_once_with({})

    def test_task3_step2_get_describes_perception_only_contract(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/task3/detect-carton",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task3.detect_carton")
        self.assertEqual(skill["affected_arms"], [])
        self.assertEqual(skill["method"], "POST")

    def test_task3_pick_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request("GET", "/api/task3/pick/status")
        self.assertEqual(status, 200)
        pick = json.loads(body)["task3_pick"]
        self.assertFalse(pick["enabled"])
        self.assertFalse(pick["ready"])

        status, _, body = self.console.request(
            "GET",
            "/api/skills/task3/pick-flat-carton",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "task3.pick_flat_carton")
        self.assertEqual(
            skill["selection_policy"],
            "nearest_left_base_flat_carton_panel",
        )
        self.assertFalse(skill["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/skills/task3/pick-flat-carton",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

        status, _, body = self.console.request(
            "GET",
            "/api/skills/task3/expand-carton",
        )
        self.assertEqual(status, 200)
        expand_skill = json.loads(body)["skill"]
        self.assertEqual(expand_skill["id"], "task3.expand_carton")
        self.assertFalse(expand_skill["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/skills/task3/expand-carton",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_left_arm_reset_home_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/left-arm/reset-home",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "left_arm.reset_home")
        self.assertEqual(skill["input_schema"]["properties"], {})
        self.assertFalse(skill["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/skills/left-arm/reset-home",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_left_arm_reset_home_skill_is_one_empty_payload_call(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.reset_home.return_value = {
            "operation": "reset_home",
            "executed": True,
            "arm": "left",
            "target_joint_positions_rad": [0, 0, 0, 1.5, 0, -1.5],
        }
        app._cartesian_jog_snapshot = mock.Mock(
            return_value={
                "home_joint_pose": {"available": True},
                "busy": False,
            }
        )

        response = app.run_left_arm_reset_home_skill({})

        self.assertTrue(response["ok"])
        self.assertEqual(response["skill"]["id"], "left_arm.reset_home")
        self.assertEqual(response["skill"]["affected_arms"], ["left"])
        self.assertFalse(response["skill"]["right_arm_commanded"])
        app.cartesian_jog.reset_home.assert_called_once_with()

        with self.assertRaises(ValueError):
            app.run_left_arm_reset_home_skill({"joints": [0] * 6})

    def test_right_arm_reset_home_is_disabled_without_explicit_config(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/skills/right-arm/reset-home",
        )
        self.assertEqual(status, 200)
        skill = json.loads(body)["skill"]
        self.assertEqual(skill["id"], "right_arm.reset_home")
        self.assertEqual(skill["input_schema"]["properties"], {})
        self.assertFalse(skill["ready"])

        status, _, body = self.console.request(
            "POST",
            "/api/skills/right-arm/reset-home",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 503)
        self.assertIn("disabled", json.loads(body)["error"])

    def test_right_arm_reset_home_skill_is_one_empty_payload_call(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.right_arm_home = mock.Mock()
        app.right_arm_home.reset_home.return_value = {
            "operation": "reset_home",
            "executed": True,
            "arm": "right",
            "target_joint_positions_rad": [0, 0, 0, 1.5, 0, -1.5],
        }
        app._right_arm_home_snapshot = mock.Mock(
            return_value={
                "home_joint_pose": {"available": True},
                "busy": False,
                "teleop_running": False,
            }
        )

        response = app.run_right_arm_reset_home_skill({})

        self.assertTrue(response["ok"])
        self.assertEqual(response["skill"]["id"], "right_arm.reset_home")
        self.assertEqual(response["skill"]["affected_arms"], ["right"])
        self.assertFalse(response["skill"]["left_arm_commanded"])
        app.right_arm_home.reset_home.assert_called_once_with()

        with self.assertRaises(ValueError):
            app.run_right_arm_reset_home_skill({"joints": [0] * 6})

    def test_task1_skill_auto_initializes_and_calls_bounded_pick(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task1_pick_enabled = True
        app.task1_pick_calibration = {"ready": True}
        app.task1_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.status.return_value = {"enabled": False}
        app.cartesian_jog.capture_orientation.return_value = {
            "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
        }
        app.cartesian_jog.enable.return_value = {"enabled": True}
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": {}}
        )

        response = app.run_task1_pick_skill({})

        self.assertTrue(response["ok"])
        self.assertEqual(response["skill"]["id"], "task1.pick_carton")
        self.assertEqual(response["skill"]["status"], "succeeded")
        self.assertEqual(
            response["skill"]["stages"][0]["status"],
            "completed",
        )
        app.cartesian_jog.capture_orientation.assert_called_once_with()
        app.cartesian_jog.enable.assert_called_once_with(
            "ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
        )
        app.pick_detected_carton.assert_called_once_with(
            {"confirm": "PICK_DETECTED_CARTON"}
        )
        app.cartesian_jog.disable.assert_called_once_with()

    def test_task1_skill_disables_xyz_jog_when_pick_fails(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task1_pick_enabled = True
        app.task1_pick_calibration = {"ready": True}
        app.task1_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.status.return_value = {"enabled": True}
        app.pick_detected_carton = mock.Mock(
            side_effect=CartesianJogSafetyViolation("unit-test pick failure")
        )

        with self.assertRaisesRegex(
            CartesianJogSafetyViolation,
            "unit-test pick failure",
        ):
            app.run_task1_pick_skill({})

        app.cartesian_jog.disable.assert_called_once_with()

    def test_task3_skill_uses_dedicated_tabletop_pick(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task3_pick_enabled = True
        app.task3_pick_calibration = {"ready": True}
        app.task3_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.status.return_value = {"enabled": False}
        app.cartesian_jog.capture_orientation.return_value = {
            "quaternion_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
        }
        app.cartesian_jog.enable.return_value = {"enabled": True}
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": {}}
        )

        response = app.run_task3_pick_skill({})

        self.assertTrue(response["ok"])
        self.assertEqual(response["skill"]["id"], "task3.pick_flat_carton")
        self.assertEqual(response["skill"]["status"], "succeeded")
        app.cartesian_jog.capture_orientation.assert_called_once_with()
        app.cartesian_jog.enable.assert_called_once_with(
            "ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
        )
        app.pick_detected_carton.assert_called_once_with(
            {"confirm": "PICK_TASK3_FLAT_CARTON"},
            task_id="task3",
        )
        app.cartesian_jog.disable.assert_called_once_with()

    def test_task3_skill_disables_xyz_jog_when_pick_fails(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task3_pick_enabled = True
        app.task3_pick_calibration = {"ready": True}
        app.task3_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.status.return_value = {"enabled": True}
        app.pick_detected_carton = mock.Mock(
            side_effect=CartesianJogSafetyViolation("unit-test pick failure")
        )

        with self.assertRaisesRegex(
            CartesianJogSafetyViolation,
            "unit-test pick failure",
        ):
            app.run_task3_pick_skill({})

        app.cartesian_jog.disable.assert_called_once_with()

    def test_task3_expand_raises_moves_to_expand_box_then_replays_with_suction_held(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        trajectory_path = Path(
            "/tmp/trajectory_20260812_212904_expand_box_49f48dcc"
        )
        recording_id = trajectory_path.name
        app.task3_pick_calibration = {"ready": True}
        app.task3_pick_error = ""
        app.task3_expand_cfg = {
            "enabled": True,
            "safe_height_m": 0.1,
            "expand_pose_name": "expand_box",
            "trajectory_path": str(trajectory_path),
            "replay_timeout_s": 90.0,
            "max_tracking_error_rad": 0.5,
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.trajectory_replay = mock.Mock()
        app.trajectory_replay.status.return_value = {"active": False}
        app.trajectory_replay.preflight.return_value = {
            "ok": True,
            "preflight": {
                "path": str(trajectory_path),
                "arms": ["left"],
                "trajectory_source": "calibration_follower_action",
            },
        }
        app.trajectory_replay.start.return_value = {"state": "starting"}
        app.trajectory_replay.wait.return_value = {"state": "completed"}
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": True,
        }
        expand_joints = [0.1] * 6
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": expand_joints,
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.status.return_value = {
            "enabled": True,
            "busy": False,
            "current_position_m": [0.238, -0.041, 0.025],
        }
        app.cartesian_jog.read_current_pose.return_value = {
            "position_m": [0.238, -0.041, 0.025],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        app.cartesian_jog.move_to_fixed_orientation_entry.return_value = {
            "executed": True,
        }
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True,
        }
        app._cartesian_jog_snapshot = mock.Mock(return_value={"enabled": False})

        response = app.run_task3_expand_skill({})

        self.assertTrue(response["ok"])
        self.assertEqual(response["skill"]["id"], "task3.expand_carton")
        self.assertTrue(response["skill"]["suction_preserved"])
        self.assertEqual(response["result"]["expand_pose"], "left.expand_box")
        self.assertEqual(response["result"]["recording_id"], recording_id)
        self.assertEqual(
            [stage["name"] for stage in response["skill"]["stages"]],
            [
                "raise_to_safe_height",
                "move_to_left_expand_box",
                "replay_expand_box_trajectory",
            ],
        )
        app.trajectory_replay.preflight.assert_called_once_with(
            recording_id,
            replay_gripper=False,
        )
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            [0.238, -0.041, 0.1],
            [0.0, 0.0, 0.0, 1.0],
            transit_z_m=0.1,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task3_expand_raise_to_safe_height",
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            expand_joints,
            pose_name="expand_box",
            speed_profile="DEFAULT",
        )
        app.trajectory_replay.start.assert_called_once_with(
            recording_id=recording_id,
            confirmation=recording_id,
            replay_gripper=False,
            allow_suction_engaged=True,
            max_tracking_error_rad=0.5,
        )
        app.trajectory_replay.wait.assert_called_once_with(
            recording_id,
            timeout_s=90.0,
        )
        self.assertGreaterEqual(app.suction.status.call_count, 4)

    def test_task2_reset_both_runs_parallel_fast_profile(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "FAST"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app.right_arm_home = mock.Mock()
        barrier = threading.Barrier(2)

        def reset(arm: str, *, speed_profile: str) -> dict:
            barrier.wait(timeout=1.0)
            return {"arm": arm, "speed_profile": speed_profile}

        app.cartesian_jog.reset_home.side_effect = lambda **kwargs: reset(
            "left", **kwargs
        )
        app.right_arm_home.reset_home.side_effect = lambda **kwargs: reset(
            "right", **kwargs
        )
        app._cartesian_jog_snapshot = mock.Mock(return_value={"busy": False})
        app._right_arm_home_snapshot = mock.Mock(return_value={"busy": False})

        response = app.run_task2_reset_both_skill({})

        self.assertEqual(response["result"]["execution"], "parallel")
        self.assertEqual(response["result"]["speed_profile"], "FAST")
        self.assertEqual(set(response["result"]["arms"]), {"left", "right"})
        app.cartesian_jog.reset_home.assert_called_once_with(
            speed_profile="FAST"
        )
        app.right_arm_home.reset_home.assert_called_once_with(
            speed_profile="FAST"
        )

    def test_task2_watcher_step_verifies_right_before_left(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "FAST"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.right_arm_home = mock.Mock()
        app.cartesian_jog = mock.Mock()
        app.runtime_parameters = mock.Mock()
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints,
        }
        call_order = []

        def right_home(**kwargs: object) -> dict[str, object]:
            call_order.append("right_home")
            return {"executed": True, **kwargs}

        def left_watcher(*args: object, **kwargs: object) -> dict[str, object]:
            call_order.append("left_watcher")
            return {"executed": True}

        app.right_arm_home.reset_home.side_effect = right_home
        app.cartesian_jog.move_to_saved_joint_pose.side_effect = left_watcher
        app._cartesian_jog_snapshot = mock.Mock(return_value={"busy": False})
        app._right_arm_home_snapshot = mock.Mock(return_value={"busy": False})

        response = app.run_task2_watcher_pose_skill({})

        self.assertEqual(call_order, ["right_home", "left_watcher"])
        self.assertEqual(response["skill"]["status"], "succeeded")
        self.assertEqual(
            response["task2_workflow"]["stage"],
            "feedback_verified",
        )
        self.assertEqual(response["task2_workflow"]["state"], "succeeded")

    def test_task2_watcher_step_stops_if_right_feedback_not_executed(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "FAST"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.right_arm_home = mock.Mock()
        app.right_arm_home.reset_home.return_value = {"executed": False}
        app.cartesian_jog = mock.Mock()
        app.runtime_parameters = mock.Mock()

        with self.assertRaisesRegex(
            CartesianJogSafetyViolation,
            "right arm.*executed=true",
        ):
            app.run_task2_watcher_pose_skill({})

        app.runtime_parameters.pose.assert_not_called()
        app.cartesian_jog.move_to_saved_joint_pose.assert_not_called()
        self.assertEqual(app.task2_workflow_status()["state"], "failed")

    def test_task1_watcher_step_runs_both_arms_in_parallel(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "DEFAULT"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.right_arm_home = mock.Mock()
        app.cartesian_jog = mock.Mock()
        app.runtime_parameters = mock.Mock()
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints,
        }
        barrier = threading.Barrier(2)

        def left_watcher(*args: object, **kwargs: object) -> dict[str, object]:
            barrier.wait(timeout=2.0)
            return {"executed": True}

        def right_home(**kwargs: object) -> dict[str, object]:
            barrier.wait(timeout=2.0)
            return {"executed": True, **kwargs}

        app.cartesian_jog.move_to_saved_joint_pose.side_effect = left_watcher
        app.right_arm_home.reset_home.side_effect = right_home
        app._cartesian_jog_snapshot = mock.Mock(return_value={"busy": False})
        app._right_arm_home_snapshot = mock.Mock(return_value={"busy": False})

        response = app.run_task1_watcher_pose_skill({})

        self.assertEqual(response["result"]["execution"], "parallel")
        self.assertEqual(response["skill"]["status"], "succeeded")
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_watcher",
            speed_profile="DEFAULT",
        )
        app.right_arm_home.reset_home.assert_called_once_with(
            speed_profile="DEFAULT",
        )

    def test_task1_fixed_place_moves_both_poses_in_parallel_then_replays(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task1_fixed_place_enabled = True
        app.task1_fixed_place_left_pose = "zhuangxiang"
        app.task1_fixed_place_right_pose = "zhuangxiang"
        app.task1_fixed_place_recording_id = (
            "act_20260815_044616_zhuangxiang_ba97f7be"
        )
        app.task1_fixed_place_speed_profile = "DEFAULT"
        app.task1_fixed_place_replay_timeout_s = 90.0
        app.trajectory_replay = mock.Mock()
        app.trajectory_replay.preflight.return_value = {
            "preflight": {
                "arms": ["left", "right"],
                "max_tracking_error_rad": None,
                "max_gripper_tracking_error_m": None,
            },
        }
        app.trajectory_replay.start.return_value = {"state": "starting"}
        app.trajectory_replay.wait.return_value = {
            "state": "completed",
            "speed_scale": 2.0,
            "suction_release_state": "released",
        }
        app._trajectory_replay_blocker = mock.Mock(return_value=None)
        app.suction = mock.Mock()
        app.suction.status.side_effect = [
            {"available": True, "engaged": True},
            {"available": True, "engaged": True},
            {"available": True, "engaged": False},
        ]
        left_joints = [0.1] * 6
        right_joints = [0.2] * 6
        app.runtime_parameters = mock.Mock()

        def pose(arm: str, name: str) -> dict[str, object]:
            self.assertEqual(name, "zhuangxiang")
            if arm == "left":
                return {"joint_positions_rad": left_joints}
            return {
                "joint_positions_rad": right_joints,
                "gripper_position_m": 0.057,
            }

        app.runtime_parameters.pose.side_effect = pose
        app.cartesian_jog = mock.Mock()
        app.right_arm_home = mock.Mock()
        barrier = threading.Barrier(2)
        motion_events: list[str] = []
        motion_events_lock = threading.Lock()

        def move(arm: str, *args: object, **kwargs: object) -> dict[str, object]:
            with motion_events_lock:
                motion_events.append(f"{arm}_entered")
            barrier.wait(timeout=2.0)
            with motion_events_lock:
                motion_events.append(f"{arm}_completed")
            return {"executed": True}

        app.cartesian_jog.move_to_saved_joint_pose.side_effect = (
            lambda *args, **kwargs: move("left", *args, **kwargs)
        )
        app.right_arm_home.move_to_saved_joint_pose.side_effect = (
            lambda *args, **kwargs: move("right", *args, **kwargs)
        )
        app._cartesian_jog_snapshot = mock.Mock(return_value={"busy": False})
        app._right_arm_home_snapshot = mock.Mock(return_value={"busy": False})

        response = app.run_task1_fixed_trajectory_place_skill({})

        self.assertTrue(response["ok"])
        self.assertEqual(response["result"]["execution"], "parallel_poses_then_replay")
        self.assertEqual(
            [stage["name"] for stage in response["skill"]["stages"]],
            ["move_both_to_zhuangxiang", "replay_zhuangxiang_trajectory"],
        )
        self.assertEqual(set(motion_events[:2]), {"left_entered", "right_entered"})
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            left_joints,
            pose_name="zhuangxiang",
            speed_profile="DEFAULT",
        )
        app.right_arm_home.move_to_saved_joint_pose.assert_called_once_with(
            right_joints,
            pose_name="zhuangxiang",
            speed_profile="DEFAULT",
            gripper_position_m=0.057,
        )
        app.trajectory_replay.start.assert_called_once_with(
            recording_id=app.task1_fixed_place_recording_id,
            confirmation=app.task1_fixed_place_recording_id,
            allow_suction_engaged=True,
        )
        app.trajectory_replay.wait.assert_called_once_with(
            app.task1_fixed_place_recording_id,
            timeout_s=90.0,
        )

    def test_task3_watcher_step_verifies_right_before_left(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "DEFAULT"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.right_arm_home = mock.Mock()
        app.cartesian_jog = mock.Mock()
        app.runtime_parameters = mock.Mock()
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints,
        }
        call_order = []

        def right_home(**kwargs: object) -> dict[str, object]:
            call_order.append("right_home")
            return {"executed": True, **kwargs}

        def left_watcher(*args: object, **kwargs: object) -> dict[str, object]:
            call_order.append("left_box_watcher")
            return {"executed": True}

        app.right_arm_home.reset_home.side_effect = right_home
        app.cartesian_jog.move_to_saved_joint_pose.side_effect = left_watcher
        app._cartesian_jog_snapshot = mock.Mock(return_value={"busy": False})
        app._right_arm_home_snapshot = mock.Mock(return_value={"busy": False})

        response = app.run_task3_watcher_pose_skill({})

        self.assertEqual(call_order, ["right_home", "left_box_watcher"])
        self.assertEqual(response["skill"]["status"], "succeeded")
        app.runtime_parameters.pose.assert_called_once_with(
            "left", "left_box_watcher"
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_box_watcher",
            speed_profile="DEFAULT",
        )

    def test_task2_ready_poses_do_not_require_suction_held(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "FAST"
        app.task2_left_ready_pose = "paper_init"
        app.task2_right_ready_pose = "init_pose"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        left_joints = [0.1] * 6
        right_joints = [0.2] * 6
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.pose.side_effect = lambda arm, name: {
            "joint_positions_rad": left_joints if arm == "left" else right_joints,
            **({"gripper_position_m": 0.0} if arm == "right" else {}),
        }
        app.cartesian_jog = mock.Mock()
        app.right_arm_home = mock.Mock()
        barrier = threading.Barrier(2)

        def move(arm: str) -> dict[str, object]:
            barrier.wait(timeout=1.0)
            return {"executed": True, "arm": arm}

        app.cartesian_jog.move_to_saved_joint_pose.side_effect = (
            lambda *args, **kwargs: move("left")
        )
        app.right_arm_home.move_to_saved_joint_pose.side_effect = (
            lambda *args, **kwargs: move("right")
        )
        app._cartesian_jog_snapshot = mock.Mock(return_value={"busy": False})
        app._right_arm_home_snapshot = mock.Mock(return_value={"busy": False})

        response = app.run_task2_ready_poses_skill({})

        self.assertEqual(
            response["result"]["targets"],
            {"right": "init_pose", "left": "paper_init"},
        )
        self.assertEqual(response["result"]["execution"], "parallel")
        self.assertEqual(
            response["task2_workflow"]["stage"],
            "feedback_verified",
        )
        app.right_arm_home.move_to_saved_joint_pose.assert_called_once_with(
            right_joints,
            pose_name="init_pose",
            speed_profile="FAST",
            gripper_position_m=0.0,
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            left_joints,
            pose_name="paper_init",
            speed_profile="FAST",
        )

    def test_task2_ready_poses_requires_both_executed_feedback(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_joint_speed_profile = "FAST"
        app.task2_left_ready_pose = "paper_init"
        app.task2_right_ready_pose = "init_pose"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True, "engaged": True}
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.pose.side_effect = lambda arm, name: {
            "joint_positions_rad": [0.1] * 6,
        }
        app.cartesian_jog = mock.Mock()
        app.right_arm_home = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True,
        }
        app.right_arm_home.move_to_saved_joint_pose.return_value = {
            "executed": False,
        }

        with self.assertRaisesRegex(
            CartesianJogSafetyViolation,
            "right arm.*executed=true",
        ):
            app.run_task2_ready_poses_skill({})

        self.assertEqual(app.task2_workflow_status()["state"], "failed")

    def test_task2_subtask_init_pose_steps_use_exact_named_pairs(self) -> None:
        for subtask_id, step_index in ((2, 6), (3, 8)):
            with self.subTest(subtask_id=subtask_id):
                app = PackagingConsoleApp.__new__(PackagingConsoleApp)
                app._motion_transition_lock = threading.RLock()
                app.task2_joint_speed_profile = "DEFAULT"
                app.trajectory_recorder = mock.Mock()
                app.trajectory_recorder.status.return_value = {"active": False}
                app.suction = mock.Mock()
                app.suction.status.return_value = {
                    "available": True,
                    "engaged": True,
                }
                left_name = f"subtask{subtask_id}_left_init"
                right_name = f"subtask{subtask_id}_right_init"
                left_joints = [0.1 * subtask_id] * 6
                right_joints = [0.2 * subtask_id] * 6
                app.runtime_parameters = mock.Mock()

                def pose(arm: str, name: str) -> dict[str, object]:
                    expected = left_name if arm == "left" else right_name
                    self.assertEqual(name, expected)
                    return {
                        "joint_positions_rad": (
                            left_joints if arm == "left" else right_joints
                        )
                    }

                app.runtime_parameters.pose.side_effect = pose
                app.cartesian_jog = mock.Mock()
                app.right_arm_home = mock.Mock()
                app.cartesian_jog.move_to_saved_joint_pose.return_value = {
                    "executed": True,
                }
                app.right_arm_home.move_to_saved_joint_pose.return_value = {
                    "executed": True,
                }
                app._cartesian_jog_snapshot = mock.Mock(
                    return_value={"busy": False}
                )
                app._right_arm_home_snapshot = mock.Mock(
                    return_value={"busy": False}
                )

                response = app.run_task2_subtask_init_poses_skill(
                    {},
                    subtask_id=subtask_id,
                )

                self.assertEqual(
                    response["result"]["targets"],
                    {"left": left_name, "right": right_name},
                )
                self.assertEqual(
                    response["skill"]["id"],
                    f"task2.move_subtask{subtask_id}_init_poses",
                )
                self.assertEqual(
                    response["task2_workflow"]["step_index"],
                    step_index,
                )
                app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
                    left_joints,
                    pose_name=left_name,
                    speed_profile="DEFAULT",
                )
                app.right_arm_home.move_to_saved_joint_pose.assert_called_once_with(
                    right_joints,
                    pose_name=right_name,
                    speed_profile="DEFAULT",
                )

    def test_task2_watch_detect_pick_uses_watcher_and_exact_detection(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_pick_enabled = True
        app.task2_pick_calibration = {"ready": True}
        app.task2_pick_error = ""
        app.task2_joint_speed_profile = "FAST"
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.runtime_parameters = mock.Mock()
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints,
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True,
        }
        cached = {
            "id": "watch-frame-1",
            "task_id": "task2",
            "target_ready": True,
            "point_left_base_m": [0.2, 0.1, 0.0],
        }
        app.detect = mock.Mock(return_value={"detection": cached})
        app._prepare_task2_direct_pick_entry = mock.Mock(
            return_value={
                "pre_contact_position_m": [0.2, 0.1, 0.0],
                "saved_orientation_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
                "motion": {"executed": True},
            }
        )
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": cached}
        )

        response = app.run_watch_detect_pick_skill({}, task_id="task2")

        self.assertEqual(response["skill"]["id"], "task2.watch_detect_pick")
        self.assertEqual(
            response["skill"]["display_name"],
            "药盒识别 + 药盒吸取",
        )
        self.assertEqual(
            [step["name"] for step in response["skill"]["steps"]],
            ["药盒识别", "药盒吸取"],
        )
        app.runtime_parameters.pose.assert_called_once_with(
            "left",
            "left_watcher",
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_watcher",
            speed_profile="FAST",
        )
        app.detect.assert_called_once_with(task_id="task2")
        app._prepare_task2_direct_pick_entry.assert_called_once_with(cached)
        app.pick_detected_carton.assert_called_once_with(
            {"confirm": "PICK_TASK2_SINGLE_CARTON"},
            task_id="task2",
            detection_override=cached,
            prepared_pre_contact_position_m=[0.2, 0.1, 0.0],
        )

    def test_task1_observe_moves_both_arms_then_uses_task1_cache(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app._workflow_detection_cache = {}
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.task2_joint_speed_profile = "DEFAULT"
        app.runtime_parameters = mock.Mock()
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints,
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True,
        }
        app.right_arm_home = mock.Mock()
        app.right_arm_home.reset_home.return_value = {"executed": True}
        detection = {
            "id": "task1-front-frame",
            "task_id": "task1",
            "target_ready": True,
            "blockers": [],
        }
        app.detect = mock.Mock(return_value={"detection": detection})

        task1 = app.run_observe_carton_step({}, task_id="task1")

        app.runtime_parameters.pose.assert_called_once_with(
            "left", "left_watcher"
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_watcher",
            speed_profile="DEFAULT",
        )
        app.right_arm_home.reset_home.assert_called_once_with(
            speed_profile="DEFAULT",
        )
        self.assertEqual(
            [stage["name"] for stage in task1["skill"]["stages"]],
            [
                "move_left_watcher_and_right_home_parallel",
                "rgbd_detect_and_cache",
            ],
        )
        self.assertEqual(
            app._workflow_detection_cache["task1"]["detection"]["id"],
            "task1-front-frame",
        )

    def test_task3_watch_detect_pick_starts_from_box_watcher(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task3_pick_enabled = True
        locked = [0.0, 0.70710678, 0.0, 0.70710678]
        app.config = {
            "shared_pick": {"target_offset_left_base_m": [0.0, 0.0, 0.0]},
        }
        app.task3_pick_cfg = {
            "enabled": True,
            "table_surface_z_m": 0.0,
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.025,
            "max_locked_orientation_error_deg": 1.0,
        }
        app.task3_pick_calibration = {
            "locked_flange_quaternion_xyzw": locked,
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": [
                    -0.008,
                    0.003,
                    0.018,
                ],
            },
        }
        app.task3_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.task.return_value = {}
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints,
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True
        }
        app.cartesian_jog.reset_home.return_value = {"executed": True}
        app.cartesian_jog.move_to_fixed_orientation_entry.return_value = {
            "executed": True
        }
        detection = {
            "id": "task3-watch-frame",
            "task_id": "task3",
            "target_ready": True,
            "blockers": [],
            "point_left_base_m": [0.236, 0.143, 0.005],
        }
        app.detect = mock.Mock(return_value={"detection": detection})
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": detection}
        )

        response = app.run_watch_detect_pick_skill({}, task_id="task3")

        self.assertEqual(
            app.runtime_parameters.pose.call_args_list,
            [mock.call("left", "left_box_watcher")],
        )
        self.assertEqual(
            app.cartesian_jog.move_to_saved_joint_pose.call_args_list,
            [
                mock.call(
                    watcher_joints,
                    pose_name="left_box_watcher",
                    speed_profile="DEFAULT",
                ),
            ],
        )
        app.cartesian_jog.reset_home.assert_called_once_with()
        app.detect.assert_called_once_with(task_id="task3")
        app.pick_detected_carton.assert_called_once_with(
            {"confirm": "PICK_TASK3_FLAT_CARTON"},
            task_id="task3",
            detection_override=detection,
            prepared_pre_contact_position_m=[
                0.22799999999999998,
                0.146,
                0.043,
            ],
        )
        self.assertEqual(
            [stage["name"] for stage in response["skill"]["stages"][:2]],
            ["move_left_box_watcher", "rgbd_detect_and_cache"],
        )

    def test_task3_observe_moves_to_box_watcher_before_detection(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app._workflow_detection_cache = {}
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True
        }
        detection = {
            "id": "task3-box-watcher-frame",
            "task_id": "task3",
            "target_ready": True,
            "blockers": [],
        }
        app.detect = mock.Mock(return_value={"detection": detection})

        response = app.run_observe_carton_step({}, task_id="task3")

        app.runtime_parameters.pose.assert_called_once_with(
            "left", "left_box_watcher"
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_box_watcher",
            speed_profile="DEFAULT",
        )
        app.detect.assert_called_once_with(task_id="task3")
        self.assertEqual(
            [stage["name"] for stage in response["skill"]["stages"]],
            ["move_left_box_watcher", "rgbd_detect_and_cache"],
        )
        self.assertEqual(
            app._workflow_detection_cache["task3"]["detection"]["id"],
            "task3-box-watcher-frame",
        )

    def test_task2_observe_alone_moves_to_watcher_and_uses_task2_cache(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app._workflow_detection_cache = {}
        app.task2_joint_speed_profile = "FAST"
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True
        }
        detection = {
            "id": "task2-watcher-frame",
            "task_id": "task2",
            "target_ready": True,
            "blockers": [],
        }
        app.detect = mock.Mock(return_value={"detection": detection})

        response = app.run_observe_carton_step({}, task_id="task2")

        app.runtime_parameters.pose.assert_called_once_with(
            "left", "left_watcher"
        )
        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_watcher",
            speed_profile="FAST",
        )
        app.detect.assert_called_once_with(task_id="task2")
        self.assertEqual(
            [stage["name"] for stage in response["skill"]["stages"]],
            ["move_left_watcher", "rgbd_detect_and_cache"],
        )
        self.assertEqual(
            app._workflow_detection_cache["task2"]["detection"]["id"],
            "task2-watcher-frame",
        )

    def test_task2_cached_pick_uses_direct_approach_entry(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        cached = {
            "id": "task2-cached-1",
            "task_id": "task2",
            "target_ready": True,
            "point_left_base_m": [0.25, -0.18, 0.02],
        }
        app._workflow_detection_cache = {
            "task2": {"cached_at": time.time(), "detection": cached}
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app._prepare_task2_direct_pick_entry = mock.Mock(
            return_value={
                "pre_contact_position_m": [0.24, -0.18, 0.05],
                "motion": {"executed": True},
            }
        )
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": cached}
        )

        response = app.run_pick_cached_carton_step({}, task_id="task2")

        self.assertTrue(response["ok"])
        app._prepare_task2_direct_pick_entry.assert_called_once_with(cached)
        app.pick_detected_carton.assert_called_once_with(
            {"confirm": "PICK_TASK2_SINGLE_CARTON"},
            task_id="task2",
            detection_override=cached,
            prepared_pre_contact_position_m=[0.24, -0.18, 0.05],
        )
        self.assertEqual(
            response["skill"]["stages"][0]["name"],
            "move_directly_to_pick_approach",
        )
        self.assertNotIn("task2", app._workflow_detection_cache)

    def test_task1_cached_pick_goes_directly_from_watcher_to_approach(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        cached = {
            "id": "task1-cached-1",
            "task_id": "task1",
            "target_ready": True,
            "point_left_base_m": [0.25, -0.18, 0.075],
            "layer_estimate": {"valid": True, "layer": 3},
        }
        app._workflow_detection_cache = {
            "task1": {"cached_at": time.time(), "detection": cached}
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app._prepare_task1_direct_pick_entry = mock.Mock(
            return_value={
                "pre_contact_position_m": [0.24, -0.18, 0.10],
                "motion": {"executed": True},
                "orientation_source": "task1_pick_calibration",
            }
        )
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": cached}
        )

        response = app.run_pick_cached_carton_step({}, task_id="task1")

        self.assertTrue(response["ok"])
        app._prepare_task1_direct_pick_entry.assert_called_once_with(cached)
        app.pick_detected_carton.assert_called_once_with(
            {"confirm": "PICK_DETECTED_CARTON"},
            task_id="task1",
            detection_override=cached,
            prepared_pre_contact_position_m=[0.24, -0.18, 0.10],
        )
        self.assertEqual(
            response["skill"]["stages"][0]["name"],
            "move_directly_from_watcher_to_pick_approach",
        )
        app.cartesian_jog.disable.assert_called_once_with()
        self.assertNotIn("task1", app._workflow_detection_cache)

    def test_task1_cached_pick_disables_xyz_jog_when_pick_fails(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        cached = {
            "id": "task1-cached-failure",
            "task_id": "task1",
            "target_ready": True,
            "point_left_base_m": [0.25, -0.18, 0.075],
            "layer_estimate": {"valid": True, "layer": 3},
        }
        app._workflow_detection_cache = {
            "task1": {"cached_at": time.time(), "detection": cached}
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app._prepare_task1_direct_pick_entry = mock.Mock(
            return_value={
                "pre_contact_position_m": [0.24, -0.18, 0.10],
                "motion": {"executed": True},
                "orientation_source": "task1_pick_calibration",
            }
        )
        app.pick_detected_carton = mock.Mock(
            side_effect=CartesianJogSafetyViolation("unit-test cached failure")
        )

        with self.assertRaisesRegex(
            CartesianJogSafetyViolation,
            "unit-test cached failure",
        ):
            app.run_pick_cached_carton_step({}, task_id="task1")

        app.cartesian_jog.disable.assert_called_once_with()

    def test_task3_cached_pick_disables_xyz_jog(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        cached = {
            "id": "task3-cached-1",
            "task_id": "task3",
            "target_ready": True,
            "point_left_base_m": [0.25, 0.12, 0.005],
        }
        app._workflow_detection_cache = {
            "task3": {"cached_at": time.time(), "detection": cached}
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app._prepare_task3_direct_pick_entry = mock.Mock(
            return_value={
                "pre_contact_position_m": [0.24, 0.13, 0.03],
                "motion": {"executed": True},
            }
        )
        app.pick_detected_carton = mock.Mock(
            return_value={"ok": True, "result": {}, "detection": cached}
        )

        response = app.run_pick_cached_carton_step({}, task_id="task3")

        self.assertTrue(response["ok"])
        app.cartesian_jog.disable.assert_called_once_with()
        self.assertNotIn("task3", app._workflow_detection_cache)

    def test_task3_cached_pick_disables_xyz_jog_when_pick_fails(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        cached = {
            "id": "task3-cached-failure",
            "task_id": "task3",
            "target_ready": True,
            "point_left_base_m": [0.25, 0.12, 0.005],
        }
        app._workflow_detection_cache = {
            "task3": {"cached_at": time.time(), "detection": cached}
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.cartesian_jog = mock.Mock()
        app._prepare_task3_direct_pick_entry = mock.Mock(
            return_value={
                "pre_contact_position_m": [0.24, 0.13, 0.03],
                "motion": {"executed": True},
            }
        )
        app.pick_detected_carton = mock.Mock(
            side_effect=CartesianJogSafetyViolation("unit-test cached failure")
        )

        with self.assertRaisesRegex(
            CartesianJogSafetyViolation,
            "unit-test cached failure",
        ):
            app.run_pick_cached_carton_step({}, task_id="task3")

        app.cartesian_jog.disable.assert_called_once_with()

    def test_task2_direct_entry_uses_saved_vertical_orientation(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.config = {
            "shared_pick": {
                "target_offset_left_base_m": [0.0, -0.02, 0.0],
            },
        }
        app.task2_pick_cfg = {
            "enabled": True,
            "target_offset_left_base_m": [0.0, 0.01, 0.0],
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.0,
            "contact_flange_z_m": 0.025,
            "max_locked_orientation_error_deg": 1.0,
        }
        locked = [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]
        app.task2_pick_calibration = {
            "locked_flange_quaternion_xyzw": locked,
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": [
                    -0.008,
                    0.003,
                    0.018,
                ],
            },
        }
        app.task2_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.task.return_value = {}
        app.runtime_parameters.pose.return_value = {
            "quaternion_xyzw": locked,
            "joint_positions_rad": [0.0] * 6,
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_fixed_orientation_entry.return_value = {
            "executed": True,
        }
        detection = {
            "task_id": "task2",
            "target_ready": True,
            "point_left_base_m": [0.25, -0.18, 0.02],
        }

        result = app._prepare_task2_direct_pick_entry(detection)

        for actual, expected in zip(
            result["pre_contact_position_m"], [0.242, -0.187, 0.025]
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(
            result["combined_target_offset_left_base_m"],
            [0.0, -0.01, 0.0],
        )
        app.runtime_parameters.pose.assert_called_once_with(
            "left", "left_pick_ready"
        )
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            result["pre_contact_position_m"],
            locked,
            transit_z_m=0.10,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task2_direct_pick_entry",
        )

    def test_task1_direct_entry_uses_counterclockwise45_calibration_without_pick_ready(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.config = {
            "shared_pick": {
                "target_offset_left_base_m": [0.0, -0.02, 0.0],
            },
        }
        app.task1_pick_cfg = {
            "enabled": True,
            "calibrated_workspace_profile": "task1_pick",
            "target_offset_left_base_m": [0.0, 0.015, 0.0],
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.025,
            "contact_flange_z_m_by_layer": {
                "1": 0.025,
                "2": 0.050,
                "3": 0.075,
            },
            "max_locked_orientation_error_deg": 1.0,
        }
        counterclockwise45 = [
            -0.27234661719867775,
            0.6532804283477525,
            0.2691979926310088,
            0.6531343221739686,
        ]
        app.task1_pick_calibration = {
            "locked_flange_quaternion_xyzw": counterclockwise45,
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": [
                    -0.00796631857643053,
                    -0.0036557995770675742,
                    0.018018696574719266,
                ],
            },
        }
        app.task1_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.task.return_value = {}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_fixed_orientation_entry.return_value = {
            "executed": True,
        }
        detection = {
            "task_id": "task1",
            "target_ready": True,
            "point_left_base_m": [0.25, -0.18, 0.075],
            "layer_estimate": {"valid": True, "layer": 3},
        }

        result = app._prepare_task1_direct_pick_entry(detection)

        self.assertEqual(result["entry_pose"], None)
        self.assertEqual(result["orientation_source"], "task1_pick_calibration")
        self.assertEqual(result["detected_layer"], 3)
        self.assertEqual(
            result["combined_target_offset_left_base_m"],
            [0.0, -0.005000000000000001, 0.0],
        )
        app.runtime_parameters.pose.assert_not_called()
        app.cartesian_jog.reset_home.assert_not_called()
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            result["pre_contact_position_m"],
            counterclockwise45,
            transit_z_m=0.10,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task1_direct_pick_entry",
            calibrated_workspace_profile="task1_pick",
            use_configured_safe_transit=False,
        )

    def test_task3_entry_returns_to_system_home_before_vertical_approach(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.config = {
            "shared_pick": {"target_offset_left_base_m": [0.0, -0.02, 0.0]},
        }
        app.task3_pick_cfg = {
            "enabled": True,
            "target_offset_left_base_m": [0.0, -0.005, 0.0],
            "table_surface_z_m": 0.0,
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.025,
            "max_locked_orientation_error_deg": 1.0,
        }
        locked = [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]
        app.task3_pick_calibration = {
            "locked_flange_quaternion_xyzw": locked,
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": [
                    -0.008,
                    0.003,
                    0.018,
                ],
            },
        }
        app.task3_pick_error = ""
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        app.runtime_parameters = mock.Mock()
        app.runtime_parameters.task.return_value = {}
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.reset_home.return_value = {"executed": True}
        app.cartesian_jog.move_to_fixed_orientation_entry.return_value = {
            "executed": True,
        }
        detection = {
            "task_id": "task3",
            "target_ready": True,
            "point_left_base_m": [0.236, 0.143, 0.005],
        }

        result = app._prepare_task3_direct_pick_entry(detection)

        self.assertEqual(result["entry_pose"], "left.system_home")
        self.assertEqual(result["orientation_source"], "task3_pick_calibration")
        self.assertEqual(
            result["combined_target_offset_left_base_m"],
            [0.0, -0.025, 0.0],
        )
        for actual, expected in zip(
            result["pre_contact_position_m"], [0.228, 0.121, 0.043]
        ):
            self.assertAlmostEqual(actual, expected)
        app.runtime_parameters.pose.assert_not_called()
        app.cartesian_jog.reset_home.assert_called_once_with()
        app.cartesian_jog.move_to_fixed_orientation_entry.assert_called_once_with(
            result["pre_contact_position_m"],
            locked,
            transit_z_m=0.10,
            enable_token="ENABLE_LEFT_CARTESIAN_JOG",
            area_clear=True,
            estop_ready=True,
            operation="task3_direct_pick_entry",
        )

    def test_watch_detect_pick_fails_closed_when_direct_entry_is_unavailable(
        self,
    ) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app._motion_transition_lock = threading.RLock()
        app.task2_pick_enabled = True
        app.task2_pick_calibration = {"ready": True}
        app.task2_pick_error = ""
        app.task2_joint_speed_profile = "FAST"
        app.suction = mock.Mock()
        app.suction.status.return_value = {
            "available": True,
            "engaged": False,
        }
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        app.runtime_parameters = mock.Mock()
        watcher_joints = [0.1, -0.2, 0.3, 1.4, -0.1, -1.3]
        app.runtime_parameters.pose.return_value = {
            "joint_positions_rad": watcher_joints
        }
        app.cartesian_jog = mock.Mock()
        app.cartesian_jog.move_to_saved_joint_pose.return_value = {
            "executed": True,
        }
        cached = {
            "id": "missing-direct-entry-1",
            "task_id": "task2",
            "target_ready": True,
            "point_left_base_m": [0.2, -0.2, 0.03],
        }
        app.detect = mock.Mock(return_value={"detection": cached})
        app._prepare_task2_direct_pick_entry = mock.Mock(
            side_effect=ValueError(
                "saved pose does not exist: left.left_pick_ready"
            )
        )
        app.pick_detected_carton = mock.Mock()

        with self.assertRaisesRegex(ValueError, "left_pick_ready"):
            app.run_watch_detect_pick_skill({}, task_id="task2")

        app.cartesian_jog.move_to_saved_joint_pose.assert_called_once_with(
            watcher_joints,
            pose_name="left_watcher",
            speed_profile="FAST",
        )
        app._prepare_task2_direct_pick_entry.assert_called_once_with(cached)
        app.pick_detected_carton.assert_not_called()

    def test_task3_contact_height_is_table_plus_tcp_offset(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task3_pick_enabled = True
        app.task3_pick_error = ""
        app.task3_pick_cfg = {
            "table_surface_z_m": 0.0,
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.025,
            "test_lift_m": 0.020,
        }
        app.task3_pick_calibration = {
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": [
                    -0.0082,
                    0.0030,
                    0.018018696574719266,
                ],
            },
        }
        app._cartesian_jog_snapshot = mock.Mock(
            return_value={"available": True, "enabled": False}
        )
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True}

        status = app.task3_pick_status()

        self.assertTrue(status["ready"])
        self.assertEqual(status["height_mode"], "table_surface_plus_tcp_offset")
        self.assertAlmostEqual(
            status["contact_flange_z_m"],
            0.018018696574719266,
        )

    def test_task2_contact_height_prefers_operator_confirmed_override(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task2_pick_enabled = True
        app.task2_pick_error = ""
        app.task2_pick_cfg = {
            "contact_flange_z_m": 0.027334022389815374,
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.025,
            "test_lift_m": 0.020,
        }
        app.task2_pick_calibration = {
            "contact_sample": {
                "absolute_contact_flange_z_m": 0.03929911657471927,
                "carton_surface_center_in_base_m": [0.256, -0.185, 0.021],
            },
        }
        app._cartesian_jog_snapshot = mock.Mock(
            return_value={"available": True, "enabled": True}
        )
        app.suction = mock.Mock()
        app.suction.status.return_value = {"available": True}

        status = app.task2_pick_status()

        self.assertTrue(status["ready"])
        self.assertAlmostEqual(
            status["contact_flange_z_m"],
            0.027334022389815374,
        )

    def test_layer_estimation_uses_height_above_table_not_world_z(self) -> None:
        height, width = 180, 240
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hsv[:, :] = [85, 180, 120]
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        depth = np.full((height, width), 1000, dtype=np.uint16)
        intrinsics = np.asarray(
            [[220.0, 0.0, 120.0], [0.0, 220.0, 90.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        config = {
            "enabled": True,
            "layer_height_m": 0.025,
            "allowed_layers": [1, 2, 3],
            "maximum_layer_error_m": 0.006,
            "table_hsv_lower": [70, 45, 25],
            "table_hsv_upper": [100, 255, 255],
            "minimum_depth_m": 0.3,
            "maximum_depth_m": 1.5,
            "sample_limit": 4000,
            "ransac_iterations": 80,
            "ransac_inlier_threshold_m": 0.002,
            "minimum_inliers": 500,
            "minimum_inlier_ratio": 0.5,
            "random_seed": 7,
        }
        for layer in (1, 2, 3):
            with self.subTest(layer=layer):
                result = estimate_carton_layer(
                    bgr,
                    depth,
                    0.001,
                    intrinsics,
                    [0.0, 0.0, 1.0 - layer * 0.025],
                    config,
                )
                self.assertTrue(result["valid"])
                self.assertEqual(result["layer"], layer)
                self.assertAlmostEqual(
                    result["measured_height_m"],
                    layer * 0.025,
                    places=6,
                )

    def test_task1_layer_boundary_keeps_72_9mm_on_second_layer(self) -> None:
        config = {
            "layer_height_m": 0.025,
            "allowed_layers": [1, 2, 3],
            "layer_minimum_height_m": {"2": 0.045, "3": 0.074},
            "maximum_layer_error_m": 0.012,
        }

        second = classify_carton_height(0.07292164274205304, config)
        third = classify_carton_height(0.075, config)

        self.assertEqual(second["layer"], 2)
        self.assertEqual(third["layer"], 3)

    def test_task1_pick_plans_contact_from_fresh_detection_and_lifts(self) -> None:
        app = PackagingConsoleApp.__new__(PackagingConsoleApp)
        app.task1_pick_enabled = True
        app.task1_pick_error = ""
        app.config = {
            "shared_pick": {
                "target_offset_left_base_m": [0.0, -0.005, 0.0],
            },
        }
        app.task1_pick_cfg = {
            "enabled": True,
            "confirm_token": "PICK_DETECTED_CARTON",
            "calibrated_workspace_profile": "task1_pick",
            "transit_z_m": 0.10,
            "pre_contact_clearance_m": 0.025,
            "test_lift_m": 0.100,
            "max_locked_orientation_error_deg": 1.0,
            "contact_flange_z_m_by_layer": {
                "1": 0.0393,
                "2": 0.0643,
                "3": 0.0893,
            },
        }
        locked = [0.0, 0.7071067811865476, 0.0, 0.7071067811865476]
        contact_offset = [-0.008, 0.003, 0.018]
        app.task1_pick_calibration = {
            "usable_for_motion": True,
            "locked_flange_quaternion_xyzw": locked,
            "contact_sample": {
                "surface_to_target_flange_offset_in_base_m": contact_offset,
            },
        }
        app._motion_transition_lock = threading.RLock()
        app.trajectory_recorder = mock.Mock()
        app.trajectory_recorder.status.return_value = {"active": False}
        jog = mock.Mock()
        jog.status.return_value = {
            "available": True,
            "enabled": True,
            "busy": False,
            "current_position_m": [0.24, -0.08, 0.11],
            "locked_quaternion_xyzw": locked,
        }
        jog.move_fixed_orientation_path.return_value = {
            "executed": True,
        }
        app.cartesian_jog = jog
        app.teleop_launcher = mock.Mock()
        app.teleop_launcher.status.return_value = {
            "running": False,
            "busy": False,
            "desired": False,
        }

        suction_state = {"engaged": False}
        suction = mock.Mock()
        suction.settle_s = 0.0
        suction.status.side_effect = lambda: {
            "available": True,
            "engaged": suction_state["engaged"],
        }

        def engage(value: bool) -> dict:
            suction_state["engaged"] = value
            return {"engaged": value, "write_confirmed": True}

        suction.set_engaged.side_effect = engage
        app.suction = suction
        surface = [0.256, -0.185, 0.021]
        app.detect = mock.Mock(
            return_value={
                "ok": True,
                "detection": {
                    "id": "fresh-detection",
                    "target_ready": True,
                    "point_left_base_m": surface,
                    "layer_estimate": {
                        "valid": True,
                        "layer": 2,
                        "measured_height_m": 0.051,
                    },
                    "blockers": [],
                },
            }
        )

        result = app.pick_detected_carton(
            {"confirm": "PICK_DETECTED_CARTON"}
        )

        app.detect.assert_called_once_with()
        self.assertTrue(result["ok"])
        self.assertTrue(result["suction"]["engaged"])
        calls = jog.move_fixed_orientation_path.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            calls[0].kwargs["calibrated_workspace_profile"],
            "task1_pick",
        )
        self.assertEqual(
            calls[1].kwargs["calibrated_workspace_profile"],
            "task1_pick",
        )
        approach = calls[0].args[0]
        contact = [
            surface[0] + contact_offset[0],
            surface[1] + contact_offset[1] - 0.005,
            0.0643,
        ]
        self.assertEqual(approach[0], [0.24, -0.08, 0.11])
        self.assertEqual(approach[1], [contact[0], contact[1], 0.11])
        self.assertEqual(approach[2], [contact[0], contact[1], contact[2] + 0.025])
        self.assertEqual(approach[3], contact)
        self.assertEqual(
            calls[1].args[0],
            [[contact[0], contact[1], contact[2] + 0.100]],
        )
        suction.set_engaged.assert_called_once_with(True)
        self.assertEqual(result["result"]["detected_layer"], 2)
        self.assertEqual(
            result["result"]["shared_target_offset_left_base_m"],
            [0.0, -0.005, 0.0],
        )

    def test_offline_frame_is_resized_and_served_as_jpeg(self) -> None:
        frame = self.console.app.camera.capture()
        self.assertEqual(frame.bgr.shape, (720, 1280, 3))
        self.assertIsNone(frame.depth_z16)

        status, headers, body = self.console.request("GET", "/api/camera/frame.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        decoded = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (720, 1280, 3))

    def test_wrist_camera_frame_is_relayed_on_same_origin(self) -> None:
        frame_body = b"\xff\xd8fake-jpeg\xff\xd9"

        class FakeFrame:
            def __init__(self) -> None:
                self.headers = {"Content-Type": "image/jpeg"}
                self._stream = io.BytesIO(frame_body)

            def __enter__(self):
                return self

            def __exit__(self, *_args) -> None:
                return None

            def read(self, size: int = -1) -> bytes:
                return self._stream.read(size)

        with mock.patch(
            "medicine_agentic.packaging_console.urllib.request.urlopen",
            return_value=FakeFrame(),
        ) as urlopen:
            status, headers, body = self.console.request(
                "GET",
                "/api/wrist-cameras/left/frame.jpg",
            )

        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertEqual(body, frame_body)
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:8877/api/cameras/left/frame.jpg",
        )

    def test_unknown_wrist_camera_is_not_proxied(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/wrist-cameras/overhead/frame.jpg",
        )
        self.assertEqual(status, 404)
        self.assertIn(b"wrist camera must be left or right", body)

    def test_frame_preview_reuses_recorder_frame_while_recording(self) -> None:
        cached_preview = CameraFrame(
            bgr=np.full((48, 80, 3), (20, 120, 220), dtype=np.uint8),
            depth_z16=None,
            captured_at=time.time(),
            frame_number=42,
        )
        with mock.patch.object(
            self.console.app.trajectory_recorder,
            "active_preview_frame",
            return_value=(True, cached_preview),
        ), mock.patch.object(
            self.console.app.camera,
            "capture",
        ) as capture:
            body = self.console.app.frame_jpeg()
            decoded = cv2.imdecode(
                np.frombuffer(body, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape, (48, 80, 3))
        capture.assert_not_called()

    def test_frame_preview_reuses_recent_encoded_jpeg(self) -> None:
        self.console.app._preview_jpeg = None
        self.console.app._preview_jpeg_cached_at = 0.0
        with mock.patch.object(
            self.console.app.camera,
            "capture",
            wraps=self.console.app.camera.capture,
        ) as capture:
            first = self.console.app.frame_jpeg()
            second = self.console.app.frame_jpeg()
        self.assertEqual(first, second)
        capture.assert_called_once()

    def test_frame_preview_does_not_compete_before_first_recorded_frame(
        self,
    ) -> None:
        with mock.patch.object(
            self.console.app.trajectory_recorder,
            "active_preview_frame",
            return_value=(True, None),
        ), mock.patch.object(
            self.console.app.camera,
            "capture",
        ) as capture:
            with self.assertRaisesRegex(
                CameraUnavailable,
                "not ready",
            ):
                self.console.app.frame_jpeg()
        capture.assert_not_called()

    def test_detect_endpoint_is_2d_preview_only(self) -> None:
        status, _, body = self.console.request(
            "POST",
            "/api/detect",
            body=b"{}",
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertTrue(payload["ok"])
        detection = payload["detection"]
        self.assertNotIn("found", detection)
        self.assertIsInstance(detection["detected_2d"], bool)
        self.assertFalse(detection["graspable_2d"])
        self.assertFalse(detection["target_ready"])
        self.assertIsInstance(detection["count"], int)
        self.assertEqual(detection["recognized_count"], detection["count"])
        self.assertEqual(
            detection["recognized_count"],
            len(detection["candidates"]),
        )
        self.assertEqual(
            detection["pickable_instance_count"],
            detection["instance_count"],
        )
        self.assertEqual(
            detection["detector"]["name"],
            "task1_3x3_adaptive_rgbd",
        )
        self.assertFalse(detection["detector"]["ok"])
        self.assertEqual(
            detection["detector"]["allowed_counts"],
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
        )
        self.assertIsNone(detection["candidate"])
        self.assertTrue(detection["overlay_url"].startswith("/api/camera/frame.jpg"))
        self.assertTrue(
            any("no depth" in blocker for blocker in detection["blockers"])
        )

        status, headers, overlay = self.console.request(
            "GET",
            detection["overlay_url"],
        )
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "image/jpeg")
        self.assertGreater(len(overlay), 100)

        status, _, body = self.console.request(
            "GET",
            "/api/camera/frame.jpg?overlay=unknown",
        )
        self.assertEqual(status, 404)
        self.assertFalse(json.loads(body)["ok"])

    def test_detect_requires_two_consistent_3d_targets(self) -> None:
        profile = dict(self.console.app.task_profiles_cfg.get("task2", {}))
        self.console.app.task_profiles_cfg["task2"] = {
            **profile,
            "detection_attempts": 3,
            "required_consistent_detections": 2,
            "consensus_tolerance_m": 0.030,
        }
        detections = [
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.150, -0.350, 0.029],
                    "instance_count": 1,
                    "blockers": [],
                },
            },
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.152, -0.351, 0.028],
                    "instance_count": 1,
                    "blockers": [],
                },
            },
        ]
        try:
            with mock.patch.object(
                self.console.app,
                "task2_detector",
                mock.Mock(),
            ), mock.patch.object(
                self.console.app,
                "_detect_once",
                side_effect=detections,
            ) as detect_once:
                response = self.console.app.detect(task_id="task2")
        finally:
            self.console.app.task_profiles_cfg["task2"] = profile

        detect_once.assert_has_calls(
            [mock.call(task_id="task2"), mock.call(task_id="task2")]
        )
        self.assertEqual(detect_once.call_count, 2)
        detection = response["detection"]
        self.assertTrue(detection["target_ready"])
        self.assertTrue(detection["temporal_consensus"]["valid"])
        self.assertEqual(detection["temporal_consensus"]["attempts_used"], 2)
        self.assertLess(
            detection["temporal_consensus"]["matched_distance_m"],
            0.003,
        )

    def test_task1_consensus_prefers_most_complete_stable_frame(self) -> None:
        profile = dict(self.console.app.task_profiles_cfg.get("task1", {}))
        self.console.app.task_profiles_cfg["task1"] = {
            **profile,
            "detection_attempts": 5,
            "required_consistent_detections": 2,
            "consensus_tolerance_m": 0.030,
        }
        detections = [
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.210, 0.112, -0.039],
                    "instance_count": 3,
                    "recognized_count": 3,
                    "safe_grasp_candidate_count": 3,
                    "blockers": [],
                },
            },
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.211, 0.112, -0.039],
                    "instance_count": 3,
                    "recognized_count": 3,
                    "safe_grasp_candidate_count": 3,
                    "blockers": [],
                },
            },
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.212, 0.112, -0.039],
                    "instance_count": 6,
                    "recognized_count": 6,
                    "safe_grasp_candidate_count": 6,
                    "blockers": [],
                },
            },
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.211, 0.113, -0.039],
                    "instance_count": 4,
                    "recognized_count": 4,
                    "safe_grasp_candidate_count": 4,
                    "blockers": [],
                },
            },
            {
                "ok": True,
                "detection": {
                    "target_ready": True,
                    "point_left_base_m": [0.210, 0.113, -0.039],
                    "instance_count": 5,
                    "recognized_count": 5,
                    "safe_grasp_candidate_count": 5,
                    "blockers": [],
                },
            },
        ]
        try:
            with mock.patch.object(
                self.console.app,
                "_detect_once",
                side_effect=detections,
            ) as detect_once:
                response = self.console.app.detect(task_id="task1")
        finally:
            self.console.app.task_profiles_cfg["task1"] = profile

        self.assertEqual(detect_once.call_count, 5)
        detection = response["detection"]
        self.assertEqual(detection["recognized_count"], 6)
        self.assertTrue(detection["target_ready"])
        self.assertEqual(
            detection["temporal_consensus"]["policy"],
            "prefer_highest_stable_layer_then_most_complete_task1_frame",
        )
        self.assertEqual(detection["temporal_consensus"]["carton_count"], 6)

    def test_graspable_candidate_is_selected_before_higher_unverified_score(
        self,
    ) -> None:
        original = self.console.app.detector
        self.console.app.detector = FakeSelectionProvider()
        try:
            # This test covers candidate selection, not Task1 layout
            # completion.  Keep the independent grid proposal out of scope.
            with mock.patch(
                "medicine_agentic.packaging_console.propose_task1_surface_grid",
                return_value=[],
            ) as grid_completion:
                detection = self.console.app.detect()
        finally:
            self.console.app.detector = original

        grid_completion.assert_not_called()
        payload = detection["detection"]
        self.assertTrue(payload["detected_2d"])
        self.assertTrue(payload["graspable_2d"])
        self.assertFalse(payload["target_ready"])
        self.assertAlmostEqual(payload["candidate"]["score"], 0.88)
        self.assertTrue(payload["candidate"]["graspable"])
        self.assertTrue(payload["dual_suction_ready_2d"])
        target = payload["dual_suction_target"]
        self.assertTrue(target["valid_2d"])
        self.assertEqual(target["midpoint_px"], [210.0, 120.0])
        self.assertEqual(len(target["cup_centers_px"]), 2)
        self.assertFalse(target["depth_support"]["available"])

    def test_candidate_with_two_valid_depth_patches_is_selected_first(self) -> None:
        original = self.console.app.detector
        self.console.app.detector = FakeDepthSelectionProvider()

        def depth_support(_depth, _scale, target, _config):
            valid = bool(target and target.midpoint_px[0] > 300.0)
            return {
                "available": True,
                "valid": valid,
                "cups": [
                    {"valid": valid, "median_depth_m": 0.9 if valid else None},
                    {"valid": valid, "median_depth_m": 0.9 if valid else None},
                ],
            }

        try:
            with mock.patch(
                "medicine_agentic.packaging_console.evaluate_dual_suction_depth",
                side_effect=depth_support,
            ), mock.patch(
                "medicine_agentic.packaging_console.propose_task1_surface_grid",
                return_value=[],
            ):
                detection = self.console.app.detect()
        finally:
            self.console.app.detector = original

        payload = detection["detection"]
        self.assertEqual(payload["candidate"]["center_px"], [380.0, 120.0])
        self.assertTrue(payload["dual_suction_target"]["depth_support"]["valid"])
        candidate_support = {
            tuple(item["center_px"]): item["dual_suction"]["depth_support"]
            for item in payload["candidates"]
        }
        self.assertFalse(candidate_support[(210.0, 120.0)]["valid"])
        self.assertTrue(candidate_support[(380.0, 120.0)]["valid"])

    def test_no_motion_or_suction_api_exists(self) -> None:
        forbidden_actions = (
            "/api/move",
            "/api/execute",
            "/api/arms/home",
            "/api/suction/on",
            "/api/vacuum/off",
            "/api/chassis/move",
            "/api/chassis/velocity",
            "/api/navigation/goal",
            "/api/navigation/map",
        )
        for path in forbidden_actions:
            with self.subTest(path=path):
                status, _, body = self.console.request(
                    "POST",
                    path,
                    body=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                self.assertEqual(status, 404)
                self.assertFalse(json.loads(body)["ok"])

        status, _, _ = self.console.request("PUT", "/api/status", body=b"{}")
        self.assertEqual(status, 405)
        status, _, _ = self.console.request("DELETE", "/api/status")
        self.assertEqual(status, 405)

    def test_static_path_traversal_is_rejected(self) -> None:
        for path in (
            "/%2e%2e/outside-secret.txt",
            "/%2e%2e%2foutside-secret.txt",
            "/subdir/%2e%2e/%2e%2e/outside-secret.txt",
        ):
            with self.subTest(path=path):
                status, _, body = self.console.request("GET", path)
                self.assertEqual(status, 403)
                payload = json.loads(body)
                self.assertFalse(payload["ok"])
                self.assertIn("traversal", payload["error"])

    def test_non_loopback_host_header_is_rejected(self) -> None:
        status, _, body = self.console.request(
            "GET",
            "/api/health",
            headers={"Host": "attacker.example"},
        )
        self.assertEqual(status, 421)
        self.assertFalse(json.loads(body)["ok"])

    def test_non_loopback_bind_is_rejected_before_camera_start(self) -> None:
        self.assertTrue(_is_loopback("127.0.0.1"))
        self.assertTrue(_is_loopback("::1"))
        self.assertTrue(_is_loopback("localhost"))
        for bind in ("0.0.0.0", "::", "192.168.1.80", "dosw1"):
            with self.subTest(bind=bind):
                self.assertFalse(_is_loopback(bind))
                with self.assertRaisesRegex(ValueError, "non-loopback"):
                    run_server(
                        self.console.config_path,
                        bind_override=bind,
                        port_override=8899,
                    )

    def test_reserved_existing_service_ports_are_rejected(self) -> None:
        self.assertEqual(RESERVED_PORTS, frozenset({8765, 8766, 8888, 9999}))
        for port in RESERVED_PORTS:
            with self.subTest(port=port):
                with self.assertRaisesRegex(ValueError, "reserved TCP port"):
                    run_server(
                        self.console.config_path,
                        bind_override="127.0.0.1",
                        port_override=port,
                    )


if __name__ == "__main__":
    unittest.main()

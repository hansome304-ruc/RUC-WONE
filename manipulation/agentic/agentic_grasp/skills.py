"""High-level skill layer: this is what the agent calls.

Every action that moves the arm honours ``dry_run`` for safety.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from agentic_grasp.config import Settings, settings
from agentic_grasp.grasp_profiles import GraspProfile, StrategyName, get_profile
from agentic_grasp.perception_client import Grasp, PerceptionClient, Segmentation
from agentic_grasp.sensors import CalibBundle, Frame, RealSenseRGBD, load_calib
from agentic_grasp.transforms import (
    apply_tcp_offset,
    offset_along_pose_z,
    pose_for_airbot,
)

log = logging.getLogger(__name__)
ArmName = Literal["left", "right"]


@dataclass
class GraspAttempt:
    grasp_in_cam: Grasp
    T_grasp_in_base: np.ndarray
    T_eef_in_base: np.ndarray
    T_pre_in_base: np.ndarray
    T_lift_in_base: np.ndarray
    arm: ArmName
    strategy: StrategyName = "default"
    workspace_z_min: float | None = None
    image_v_ratio: float | None = None
    side_entry: bool = False
    side_overdrive: float = 0.0


class WorkspaceViolation(RuntimeError):
    pass


class GraspSkills:
    """Wraps cameras + arms + perception client into agent-facing skills."""

    def __init__(
        self,
        cfg: Settings = settings,
        cam_serial: str | None = None,
        connect_arms: bool = True,
        speed: str = "SLOW",
        require_calib: bool = True,
    ):
        self.cfg = cfg
        self._speed_name = speed
        self.cam: RealSenseRGBD | None = None
        self.left = self.right = None
        self._recording_stop: threading.Event | None = None
        self._recording_thread: threading.Thread | None = None
        # Track which side-effects __init__ has already taken so __del__ can
        # release them even if a later step raises (RealSense busy, calib
        # missing, etc). Without this, a failed __init__ leaves the AIRBOT
        # gRPC servo stream open on the server side and subsequent runs hit
        # "Another request is in progress" until airbot_server is restarted.
        try:
            self.cam = RealSenseRGBD(cam_serial or cfg.cam_front_serial)
            try:
                self.calib: CalibBundle | None = load_calib(
                    cfg.left_calib_path, cfg.right_calib_path, cfg.intrinsics_path
                )
            except FileNotFoundError:
                if require_calib:
                    raise
                log.warning(
                    "Running WITHOUT calibration. /grasp endpoints can be queried, "
                    "but cam→base transforms will be identity and any motion is unsafe."
                )
                self.calib = None
            self.client = PerceptionClient()

            if connect_arms:
                from airbot_py.arm import AIRBOTPlay, RobotMode, SpeedProfile  # imported here so dry-run scripts can run without arm

                self._RobotMode = RobotMode
                self._SpeedProfile = SpeedProfile

                self.left = AIRBOTPlay(url="localhost", port=cfg.left_arm_port)
                self.right = AIRBOTPlay(url="localhost", port=cfg.right_arm_port)
                assert self.left.connect(), "failed to connect to left arm gRPC"
                assert self.right.connect(), "failed to connect to right arm gRPC"
                for arm in (self.left, self.right):
                    arm.switch_mode(RobotMode.PLANNING_POS)
                    arm.set_speed_profile(getattr(SpeedProfile, speed))
        except Exception:
            # Make sure we don't leak streams/devices on a partial init.
            self.close()
            raise

    def close(self) -> None:
        """Release RealSense + AIRBOT gRPC streams. Safe to call multiple times."""
        self.stop_video_recording()
        for arm_name in ("left", "right"):
            arm = getattr(self, arm_name, None)
            if arm is None:
                continue
            for method in ("disconnect", "close", "shutdown"):
                fn = getattr(arm, method, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception as exc:
                        log.debug("%s.%s() raised: %s", arm_name, method, exc)
                    break
            setattr(self, arm_name, None)
        if self.cam is not None:
            try:
                self.cam.close()
            except Exception as exc:
                log.debug("cam.close() raised: %s", exc)
            self.cam = None

    def start_video_recording(self, out_path: str | Path, fps: float = 10.0) -> None:
        """Record the front RGB-D camera's color stream while grasping."""
        if self.cam is None:
            raise RuntimeError("Camera is not initialized")
        if self._recording_thread is not None:
            raise RuntimeError("Video recording is already running")

        import cv2

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        stop_event = threading.Event()
        self._recording_stop = stop_event

        def _worker() -> None:
            writer = None
            frame_period = 1.0 / max(float(fps), 1.0)
            try:
                while not stop_event.is_set():
                    started = time.time()
                    frame = self.cam.capture(warmup_frames=0)
                    bgr = frame.rgb[..., ::-1].copy()
                    if writer is None:
                        h, w = bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
                        if not writer.isOpened():
                            raise RuntimeError(f"Failed to open video writer: {out_path}")
                    writer.write(bgr)
                    time.sleep(max(0.0, frame_period - (time.time() - started)))
            except Exception as exc:
                log.warning("video recording stopped after error: %s", exc)
            finally:
                if writer is not None:
                    writer.release()

        self._recording_thread = threading.Thread(target=_worker, name="grasp-video-recording", daemon=True)
        self._recording_thread.start()
        log.info("recording grasp video → %s", out_path)

    def stop_video_recording(self) -> None:
        if self._recording_stop is None or self._recording_thread is None:
            return
        self._recording_stop.set()
        self._recording_thread.join(timeout=5.0)
        self._recording_stop = None
        self._recording_thread = None
        log.info("stopped grasp video recording")

    def __del__(self):
        # Best-effort cleanup at GC time. Don't raise from here.
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ===================== sensing =====================

    def capture(self) -> Frame:
        return self.cam.capture()

    def find_object(
        self,
        prompt: str | None = None,
        bbox: list[int] | None = None,
        save_overlay: str | None = None,
    ) -> Segmentation:
        frame = self.capture()
        seg = self.client.segment(frame.rgb, prompt=prompt, bbox=bbox)
        if save_overlay:
            self._save_segmentation_overlay(frame.rgb, seg, save_overlay)
        return seg

    def propose_grasps(
        self,
        mask: np.ndarray | None = None,
        prompt: str | None = None,
        endpoint: str = "/grasp",
        top_k: int = 10,
    ) -> list[Grasp]:
        frame = self.capture()
        if mask is None:
            if prompt is None:
                raise ValueError("Need a mask or a prompt")
            seg = self.client.segment(frame.rgb, prompt=prompt)
            mask = seg.mask
        return self.client.grasps(
            frame.rgb, frame.depth_mm, self._K_for_perception(frame), mask,
            endpoint=endpoint, top_k=top_k,
        )

    # ===================== motion =====================

    def home(self, dry_run: bool = False):
        joints = [0.0, 0.0, 0.0, 1.5, 0.0, -1.5]
        if dry_run:
            log.info("[DRY] home() would move both arms to %s", joints)
            return
        for arm in (self.left, self.right):
            assert arm is not None
            arm.switch_mode(self._RobotMode.PLANNING_POS)
            arm.move_to_joint_pos(joints)
            # IMPORTANT: use `move_eef_pos` (one-shot, planning-mode) instead of
            # `servo_eef_pos` (opens a persistent SetState stream that AIRBOT's
            # fsm_node serialises against `move_to_cart_pose`). Mixing them
            # causes `FAILED_PRECONDITION: Another request is in progress`
            # on every subsequent cart move.
            arm.move_eef_pos(0.07)
        
        # Wait for arms to fully home and stabilize so they don't block or vibrate the camera
        log.info("Waiting 3.0 seconds for arms to fully home and stabilize...")
        time.sleep(3.0)

    def open_gripper(self, arm: ArmName, width: float = 0.07, dry_run: bool = False):
        if dry_run:
            log.info("[DRY] open_gripper(%s, %.3fm)", arm, width); return
        # See `home()` for why this is `move_eef_pos`, not `servo_eef_pos`.
        self._arm(arm).move_eef_pos(float(width))

    def close_gripper(self, arm: ArmName, dry_run: bool = False):
        if dry_run:
            log.info("[DRY] close_gripper(%s)", arm); return
        self._arm(arm).move_eef_pos(0.0)

    def move_to_cart_pose_in_base(
        self,
        T_eef_in_base: np.ndarray,
        arm: ArmName,
        z_min: float | None = None,
        dry_run: bool = False,
    ) -> bool:
        self._check_in_workspace(T_eef_in_base[:3, 3], z_min=z_min)
        target = pose_for_airbot(T_eef_in_base)
        if dry_run:
            log.info("[DRY] move_to_cart_pose(%s) → xyz=%s quat=%s",
                     arm, target[0], target[1])
            return True
        ok = self._arm(arm).move_to_cart_pose(target, blocking=True)
        if not ok:
            log.warning("planning failed for arm=%s target=%s", arm, target)
        return bool(ok)

    def estop(self):
        """Make both arms limp. Call this from signal handlers."""
        if self.left is not None and self.right is not None:
            for arm in (self.left, self.right):
                try:
                    arm.switch_mode(self._RobotMode.GRAVITY_COMP)
                except Exception:
                    pass

    # ===================== composite =====================

    def grasp_object(
        self,
        prompt: str,
        arm: ArmName = "right",
        endpoint: str = "/grasp",
        pre_grasp_back: float = 0.08,
        lift_height: float = 0.10,
        max_attempts: int = 5,
        execute_single_attempt: bool | None = None,
        strategy: StrategyName = "default",
        side_overdrive: float | None = None,
        dry_run: bool = False,
    ) -> tuple[bool, GraspAttempt | None]:
        """Full pipeline: segment → grasp → pre → grasp → close → lift."""
        profile = self._profile_for_strategy(strategy)
        if profile.pre_grasp_back is not None:
            pre_grasp_back = min(float(pre_grasp_back), profile.pre_grasp_back)
        effective_side_overdrive = profile.side_overdrive if side_overdrive is None else float(side_overdrive)
        if strategy != "default":
            log.info(
                "using %s strategy: z=[%.3f,%.3f], pre_z>=%.3f, ws_z>=%.3f, pre_grasp_back=%.3f, side_overdrive=%.3f",
                strategy,
                profile.grasp_z_min, profile.grasp_z_max, profile.grasp_pre_z_min,
                profile.workspace_z_min, pre_grasp_back,
                effective_side_overdrive if strategy == "side-bottle" else 0.0,
            )
        frame = self.capture()
        self._save_sam_input(frame.rgb, "debug_out/sam_input.png")
        seg = self.client.segment(frame.rgb, prompt=prompt)
        log.info("found %r bbox=%s score=%.2f mask_px=%d",
                 prompt, seg.bbox, seg.score, int(seg.mask.sum()))
        self._save_segmentation_overlay(frame.rgb, seg, "debug_out/grasp_segment_overlay.png")
        if int(seg.mask.sum()) < 100:
            log.warning("mask too small, abort")
            return False, None

        grasps = self.client.grasps(
            frame.rgb,
            frame.depth_mm,
            self._K_for_perception(frame),
            seg.mask,
            endpoint=endpoint,
            top_k=max_attempts * 3,
        )
        if not grasps:
            log.warning("perception server returned 0 grasps")
            return False, None

        if self.calib is None:
            log.warning("no calibration loaded; using identity cam→base (DRY-RUN ONLY)")
            T_cam_to_base = np.eye(4)
        else:
            T_cam_to_base = (
                self.calib.T_cam_to_left if arm == "left" else self.calib.T_cam_to_right
            )

        selected_attempts: list[tuple[float, GraspAttempt]] = []
        for i, g in enumerate(grasps):
            if len(selected_attempts) >= max_attempts:
                break

            T_in_base = T_cam_to_base @ g.T_cam
            T_eef = apply_tcp_offset(T_in_base, self.cfg.gripper_tcp_offset)
            if profile.force_side_pose:
                T_eef = self._side_entry_pose(T_eef, arm)
                T_eef = self._overdrive_pose(T_eef, strategy, effective_side_overdrive)
            approach_y = float(T_eef[1, 2])
            min_side_y = float(self.cfg.side_approach_min_y)
            if strategy != "side-bottle" and self._is_far_side_approach(arm, approach_y, min_side_y):
                # A parallel-jaw grasp is symmetric if we rotate 180 degrees
                # around the closing axis: the contact line stays the same, but
                # the pre-grasp comes from the active arm's side.
                T_flipped = T_eef.copy()
                T_flipped[:3, :3] = T_eef[:3, :3] @ np.diag([-1.0, 1.0, -1.0])
                flipped_approach_y = float(T_flipped[1, 2])
                if self._is_far_side_approach(arm, flipped_approach_y, min_side_y):
                    log.info(
                        "skip candidate %d: far-side approach remains bad after flip "
                        "(approach_y %.3f -> %.3f)",
                        i + 1, approach_y, flipped_approach_y,
                    )
                    continue
                log.info(
                    "candidate %d: flipped far-side approach_y %.3f -> %.3f",
                    i + 1, approach_y, flipped_approach_y,
                )
                T_eef = T_flipped

            T_pre = self._pre_grasp_pose(T_eef, strategy, -float(pre_grasp_back))
            T_lift = T_eef.copy()
            T_lift[2, 3] += float(lift_height)
            attempt = GraspAttempt(
                g,
                T_in_base,
                T_eef,
                T_pre,
                T_lift,
                arm,
                strategy=strategy,
                workspace_z_min=profile.workspace_z_min,
                image_v_ratio=self._attempt_image_v_ratio(frame, seg, T_cam_to_base, T_eef),
                side_entry=(strategy == "side-bottle"),
                side_overdrive=effective_side_overdrive if strategy == "side-bottle" else 0.0,
            )
            reason = self._reject_grasp_attempt(attempt, profile)
            if reason is not None:
                log.info(
                    "skip candidate %d: %s score=%.2f xyz=%s entry=%s closing=%s",
                    i + 1, reason, g.score, T_eef[:3, 3].round(3).tolist(),
                    self._entry_axis(T_eef, strategy).round(3).tolist(),
                    T_eef[:3, 1].round(3).tolist(),
                )
                continue

            rank = self._rank_grasp_attempt(attempt, profile)
            selected_attempts.append((rank, attempt))
            log.info(
                "candidate %d accepted rank=%.3f score=%.2f xyz=%s entry=%s closing=%s pre_xyz=%s image_v=%s",
                i + 1, rank, g.score, T_eef[:3, 3].round(3).tolist(),
                self._entry_axis(T_eef, strategy).round(3).tolist(),
                T_eef[:3, 1].round(3).tolist(),
                T_pre[:3, 3].round(3).tolist(),
                None if attempt.image_v_ratio is None else round(attempt.image_v_ratio, 3),
            )

        if not selected_attempts:
            log.warning("no grasp candidates survived conservative filters")
            return False, None

        selected_attempts.sort(key=lambda item: item[0], reverse=True)
        if execute_single_attempt is None:
            execute_single_attempt = self.cfg.execute_single_attempt
        attempts_to_execute = selected_attempts
        if execute_single_attempt and not dry_run:
            # In real runs we still allow trying the next candidate if the
            # pre-grasp pose itself is unreachable. That is safe because the arm
            # has not moved near the object yet. Once pre-grasp succeeds, any
            # later failure stops the run instead of sweeping around the bottle.
            attempts_to_execute = selected_attempts[:max_attempts]

        for motion_idx, (_, attempt) in enumerate(attempts_to_execute, start=1):
            g = attempt.grasp_in_cam
            self._save_selected_grasp_debug(
                frame=frame,
                seg=seg,
                attempt=attempt,
                T_cam_to_base=T_cam_to_base,
                out_image="debug_out/selected_grasp.png",
                out_json="debug_out/selected_grasp.json",
            )
            log.info(
                "execute attempt %d/%d score=%.2f xyz=%s entry=%s closing=%s",
                motion_idx, len(attempts_to_execute), g.score,
                attempt.T_eef_in_base[:3, 3].round(3).tolist(),
                self._entry_axis(attempt.T_eef_in_base, attempt.strategy).round(3).tolist(),
                attempt.T_eef_in_base[:3, 1].round(3).tolist(),
            )
            status = self._execute_grasp_attempt(
                attempt=attempt,
                pre_grasp_back=pre_grasp_back,
                lift_height=lift_height,
                dry_run=dry_run,
            )
            if status == "success":
                return True, attempt
            if not dry_run and status == "pre_failed":
                log.warning("pre-grasp planning failed before moving near object; trying next candidate")
                continue
            if not dry_run:
                log.warning("real execution failed after reaching near-object phase; stopping")
                return False, attempt

        return False, None

    # ===================== internals =====================

    def _profile_for_strategy(self, strategy: StrategyName) -> GraspProfile:
        profile = get_profile(strategy)
        if strategy != "default":
            return profile
        return GraspProfile(
            name=profile.name,
            description=profile.description,
            grasp_z_min=self.cfg.grasp_z_min,
            grasp_z_max=self.cfg.grasp_z_max,
            grasp_pre_z_min=self.cfg.grasp_pre_z_min,
            workspace_z_min=self.cfg.ws_z_min,
            max_abs_entry_z=self.cfg.grasp_max_abs_approach_z,
            max_abs_closing_z=self.cfg.grasp_max_abs_closing_z,
            pre_grasp_back=profile.pre_grasp_back,
            side_overdrive=profile.side_overdrive,
            min_image_v_ratio=profile.min_image_v_ratio,
            target_image_v_ratio=profile.target_image_v_ratio,
            force_side_pose=profile.force_side_pose,
        )

    def _reject_grasp_attempt(self, attempt: GraspAttempt, profile: GraspProfile) -> str | None:
        xyz = attempt.T_eef_in_base[:3, 3]
        pre_xyz = attempt.T_pre_in_base[:3, 3]
        approach = self._entry_axis(attempt.T_eef_in_base, attempt.strategy)
        closing = attempt.T_eef_in_base[:3, 1]
        if xyz[2] < profile.grasp_z_min or xyz[2] > profile.grasp_z_max:
            return f"grasp z={xyz[2]:.3f} outside [{profile.grasp_z_min:.3f},{profile.grasp_z_max:.3f}]"
        if pre_xyz[2] < profile.grasp_pre_z_min:
            return f"pre-grasp z={pre_xyz[2]:.3f} below {profile.grasp_pre_z_min:.3f}"
        if abs(float(approach[2])) > profile.max_abs_entry_z:
            return f"entry direction too vertical, abs(z)={abs(float(approach[2])):.3f}"
        if profile.max_abs_closing_z is not None and abs(float(closing[2])) > profile.max_abs_closing_z:
            return f"closing direction too vertical, abs(z)={abs(float(closing[2])):.3f}"
        if profile.min_image_v_ratio is not None and attempt.image_v_ratio is not None:
            if attempt.image_v_ratio < profile.min_image_v_ratio:
                return f"grasp projects too high in bbox, image_v={attempt.image_v_ratio:.3f}"
        for label, pose in [
            ("pre", attempt.T_pre_in_base),
            ("grasp", attempt.T_eef_in_base),
            ("lift", attempt.T_lift_in_base),
        ]:
            try:
                self._check_in_workspace(pose[:3, 3], z_min=attempt.workspace_z_min)
            except WorkspaceViolation as exc:
                return f"{label} outside workspace: {exc}"
        return None

    def _rank_grasp_attempt(self, attempt: GraspAttempt, profile: GraspProfile) -> float:
        xyz = attempt.T_eef_in_base[:3, 3]
        approach = self._entry_axis(attempt.T_eef_in_base, attempt.strategy)
        closing = attempt.T_eef_in_base[:3, 1]
        z_mid = 0.5 * (profile.grasp_z_min + profile.grasp_z_max)
        z_span = max(profile.grasp_z_max - profile.grasp_z_min, 1e-6)
        z_score = 1.0 - min(abs(float(xyz[2]) - z_mid) / (0.5 * z_span), 1.0)
        horizontal_score = 1.0 - min(abs(float(approach[2])) / profile.max_abs_entry_z, 1.0)
        if profile.max_abs_closing_z is None:
            closing_horizontal_score = 0.0
        else:
            closing_horizontal_score = 1.0 - min(abs(float(closing[2])) / profile.max_abs_closing_z, 1.0)
        side_score = min(abs(float(approach[1])), 1.0)
        score = float(
            attempt.grasp_in_cam.score
            + 0.4 * z_score
            + 0.3 * horizontal_score
            + 0.3 * closing_horizontal_score
            + 0.2 * side_score
        )
        if profile.target_image_v_ratio is not None and attempt.image_v_ratio is not None:
            # In the image, larger v means lower in the object bbox. For small
            # upright bottles we prefer the body over the cap/neck.
            body_score = 1.0 - min(abs(float(attempt.image_v_ratio) - profile.target_image_v_ratio) / 0.38, 1.0)
            score += 0.8 * body_score
        if profile.force_side_pose:
            score += 0.4 * side_score
        return score

    def _execute_grasp_attempt(
        self,
        attempt: GraspAttempt,
        pre_grasp_back: float,
        lift_height: float,
        dry_run: bool = False,
    ) -> str:
        del pre_grasp_back, lift_height  # poses are already baked into the attempt
        try:
            self.open_gripper(
                attempt.arm,
                width=max(attempt.grasp_in_cam.width + 0.01, 0.05),
                dry_run=dry_run,
            )
            if not self.move_to_cart_pose_in_base(
                attempt.T_pre_in_base,
                attempt.arm,
                z_min=attempt.workspace_z_min,
                dry_run=dry_run,
            ):
                return "pre_failed"
            if not self.move_to_cart_pose_in_base(
                attempt.T_eef_in_base,
                attempt.arm,
                z_min=attempt.workspace_z_min,
                dry_run=dry_run,
            ):
                return "grasp_failed"
            self.close_gripper(attempt.arm, dry_run=dry_run)
            if not dry_run:
                time.sleep(0.5)
            if not self.move_to_cart_pose_in_base(
                attempt.T_lift_in_base,
                attempt.arm,
                z_min=attempt.workspace_z_min,
                dry_run=dry_run,
            ):
                return "lift_failed"
            return "success"
        except WorkspaceViolation as exc:
            log.warning("skip execution: %s", exc)
            return "pre_failed"

    def _arm(self, name: ArmName):
        arm = self.left if name == "left" else self.right
        if arm is None:
            raise RuntimeError("Arms not connected (constructed with connect_arms=False)")
        return arm

    @staticmethod
    def _is_far_side_approach(arm: ArmName, approach_y: float, min_side_y: float) -> bool:
        if arm == "left":
            return approach_y > -min_side_y
        return approach_y < min_side_y

    @staticmethod
    def _entry_axis(T_eef: np.ndarray, strategy: StrategyName) -> np.ndarray:
        if strategy == "side-bottle":
            return T_eef[:3, 0]
        return T_eef[:3, 2]

    def _pre_grasp_pose(self, T_eef: np.ndarray, strategy: StrategyName, dist: float) -> np.ndarray:
        if strategy == "side-bottle":
            out = T_eef.copy()
            out[:3, 3] += float(dist) * self._entry_axis(T_eef, strategy)
            return out
        return offset_along_pose_z(T_eef, dist)

    def _overdrive_pose(self, T_eef: np.ndarray, strategy: StrategyName, dist: float) -> np.ndarray:
        if strategy != "side-bottle" or dist == 0.0:
            return T_eef
        out = T_eef.copy()
        out[:3, 3] += float(dist) * self._entry_axis(T_eef, strategy)
        return out

    @staticmethod
    def _side_entry_pose(T_eef: np.ndarray, arm: ArmName) -> np.ndarray:
        del arm  # side-bottle uses a fixed table-parallel hand frame for now.
        out = T_eef.copy()
        # Keep both the hand's forward direction and the gripper closing axis
        # in the tabletop plane. The previous version left `closing` vertical,
        # which rolled the gripper 90 degrees and made side-grasping a bottle
        # impossible.
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        closing = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        z_axis = np.cross(x_axis, closing)
        z_axis /= np.linalg.norm(z_axis)
        out[:3, :3] = np.column_stack([x_axis, closing, z_axis])
        return out

    def _K_for_perception(self, frame: Frame) -> np.ndarray:
        """Prefer the calibrated intrinsics over RealSense's factory ones."""
        if self.calib is None:
            return frame.K
        K = self.calib.K_calib
        rs_w = float(frame.K[0, 2] * 2)
        cal_w = float(K[0, 2] * 2)
        if abs(rs_w - cal_w) > 50:
            log.warning(
                "calibration intrinsics seem to be at a different resolution "
                "(cal cx*2=%.0f vs rs cx*2=%.0f); falling back to RealSense intrinsics",
                cal_w, rs_w,
            )
            return frame.K
        return K

    def _attempt_image_v_ratio(
        self,
        frame: Frame,
        seg: Segmentation,
        T_cam_to_base: np.ndarray,
        T_eef_in_base: np.ndarray,
    ) -> float | None:
        T_base_to_cam = np.linalg.inv(T_cam_to_base)
        p_cam = (T_base_to_cam @ np.r_[T_eef_in_base[:3, 3], 1.0])[:3]
        pt = self._project_point(self._K_for_perception(frame), p_cam)
        if pt is None:
            return None
        _, v = pt
        _, y1, _, y2 = seg.bbox
        denom = max(float(y2 - y1), 1.0)
        return float(np.clip((float(v) - float(y1)) / denom, 0.0, 1.0))

    def _check_in_workspace(self, xyz: np.ndarray, z_min: float | None = None) -> None:
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        c = self.cfg
        min_z = c.ws_z_min if z_min is None else float(z_min)
        if not (c.ws_x_min <= x <= c.ws_x_max
                and c.ws_y_min <= y <= c.ws_y_max
                and min_z <= z <= c.ws_z_max):
            raise WorkspaceViolation(
                f"target xyz=({x:.3f},{y:.3f},{z:.3f}) outside workspace box "
                f"x∈[{c.ws_x_min},{c.ws_x_max}] y∈[{c.ws_y_min},{c.ws_y_max}] "
                f"z∈[{min_z},{c.ws_z_max}]"
            )

    @staticmethod
    def _save_sam_input(rgb: np.ndarray, out_path: str) -> None:
        import cv2

        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), rgb[..., ::-1])
        log.info("saved SAM input image → %s", path)

    def _save_selected_grasp_debug(
        self,
        frame: Frame,
        seg: Segmentation,
        attempt: GraspAttempt,
        T_cam_to_base: np.ndarray,
        out_image: str,
        out_json: str,
    ) -> None:
        import cv2
        import json

        bgr = frame.rgb[..., ::-1].copy()
        overlay = bgr.copy()
        overlay[seg.mask.astype(bool)] = [0, 0, 120]
        bgr = cv2.addWeighted(bgr, 0.75, overlay, 0.25, 0)
        x1, y1, x2, y2 = seg.bbox
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        K = self._K_for_perception(frame)
        T_base_to_cam = np.linalg.inv(T_cam_to_base)
        T_eef_cam = T_base_to_cam @ attempt.T_eef_in_base
        T_pre_cam = T_base_to_cam @ attempt.T_pre_in_base
        T_lift_cam = T_base_to_cam @ attempt.T_lift_in_base
        self._draw_pose_overlay(bgr, K, T_eef_cam, attempt.grasp_in_cam.width, "EXEC", (0, 0, 255))
        self._draw_point_overlay(bgr, K, T_pre_cam[:3, 3], "PRE", (255, 255, 0))
        self._draw_point_overlay(bgr, K, T_lift_cam[:3, 3], "LIFT", (0, 255, 255))

        data = {
            "arm": attempt.arm,
            "strategy": attempt.strategy,
            "side_entry": attempt.side_entry,
            "side_overdrive": attempt.side_overdrive,
            "score": float(attempt.grasp_in_cam.score),
            "width": float(attempt.grasp_in_cam.width),
            "image_v_ratio": attempt.image_v_ratio,
            "xyz_base": attempt.T_eef_in_base[:3, 3].round(6).tolist(),
            "pre_xyz_base": attempt.T_pre_in_base[:3, 3].round(6).tolist(),
            "lift_xyz_base": attempt.T_lift_in_base[:3, 3].round(6).tolist(),
            "tool_forward_axis_base": attempt.T_eef_in_base[:3, 0].round(6).tolist(),
            "entry_axis_base": self._entry_axis(attempt.T_eef_in_base, attempt.strategy).round(6).tolist(),
            "approach_axis_base": attempt.T_eef_in_base[:3, 2].round(6).tolist(),
            "closing_axis_base": attempt.T_eef_in_base[:3, 1].round(6).tolist(),
            "T_eef_in_base": attempt.T_eef_in_base.round(8).tolist(),
            "T_grasp_in_cam_raw": attempt.grasp_in_cam.T_cam.round(8).tolist(),
        }

        image_path = Path(out_image)
        json_path = Path(out_json)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), bgr)
        json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.info("saved selected grasp visualization → %s", image_path)
        log.info("saved selected grasp metadata → %s", json_path)

    @staticmethod
    def _project_point(K: np.ndarray, p: np.ndarray) -> tuple[int, int] | None:
        if p[2] <= 0:
            return None
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        return int(round(u)), int(round(v))

    @classmethod
    def _draw_point_overlay(
        cls,
        bgr: np.ndarray,
        K: np.ndarray,
        p_cam: np.ndarray,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        import cv2

        pt = cls._project_point(K, p_cam)
        if pt is None:
            return
        cv2.circle(bgr, pt, 6, color, 2)
        cv2.putText(bgr, label, (pt[0] + 6, pt[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    @classmethod
    def _draw_pose_overlay(
        cls,
        bgr: np.ndarray,
        K: np.ndarray,
        T_cam: np.ndarray,
        width: float,
        label: str,
        color: tuple[int, int, int],
    ) -> None:
        import cv2

        center = cls._project_point(K, T_cam[:3, 3])
        if center is None:
            return
        cv2.circle(bgr, center, 8, color, 2)
        cv2.putText(bgr, label, (center[0] + 8, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        axis_len = 0.05
        for axis_idx, axis_color, axis_label in [
            (0, (0, 0, 255), "x"),
            (1, (0, 255, 0), "closing"),
            (2, (255, 0, 0), "approach"),
        ]:
            tip = np.array([0.0, 0.0, 0.0, 1.0])
            tip[axis_idx] = axis_len
            tip_px = cls._project_point(K, (T_cam @ tip)[:3])
            if tip_px is None:
                continue
            cv2.arrowedLine(bgr, center, tip_px, axis_color, 2, tipLength=0.2)
            cv2.putText(bgr, axis_label, tip_px, cv2.FONT_HERSHEY_SIMPLEX, 0.4, axis_color, 1)

        half_w = max(float(width), 0.04) / 2.0
        left = (T_cam @ np.array([0.0, -half_w, 0.0, 1.0]))[:3]
        right = (T_cam @ np.array([0.0, half_w, 0.0, 1.0]))[:3]
        left_px = cls._project_point(K, left)
        right_px = cls._project_point(K, right)
        if left_px is not None and right_px is not None:
            cv2.line(bgr, left_px, right_px, (255, 0, 255), 3)
            for p3, p2 in [(left, left_px), (right, right_px)]:
                tip_px = cls._project_point(K, p3 + T_cam[:3, 2] * 0.025)
                if tip_px is not None:
                    cv2.line(bgr, p2, tip_px, (255, 0, 255), 2)

    @staticmethod
    def _save_segmentation_overlay(rgb: np.ndarray, seg: Segmentation, out_path: str) -> None:
        import cv2  # opencv-python-headless is in dos-w1 env

        bgr = rgb[..., ::-1].copy()
        overlay = bgr.copy()
        overlay[seg.mask.astype(bool)] = [0, 0, 255]
        blended = cv2.addWeighted(bgr, 0.6, overlay, 0.4, 0)
        x1, y1, x2, y2 = seg.bbox
        cv2.rectangle(blended, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(blended, f"score={seg.score:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        from pathlib import Path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, blended)
        log.info("saved overlay → %s", out_path)

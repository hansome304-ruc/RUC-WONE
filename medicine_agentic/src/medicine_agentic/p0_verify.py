from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np


AcousticDecision = Literal["attached", "not_attached", "unavailable", "uncertain"]
AcousticPolicy = Literal["supporting", "required"]


@dataclass(frozen=True)
class VisualProbeEvidence:
    passed: bool
    commanded_delta_m: tuple[float, float, float]
    observed_delta_m: tuple[float, float, float] | None
    tracking_error_mm: float | None
    observed_vertical_lift_mm: float | None
    observed_lateral_motion_mm: float | None
    source_vacated: bool | None
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class HoldDecision:
    passed: bool
    visual: VisualProbeEvidence
    acoustic: AcousticDecision
    acoustic_policy: AcousticPolicy
    reason: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["visual"] = self.visual.to_dict()
        return payload


def assess_visual_probe_lift(
    before_point_left_base_m: list[float] | tuple[float, ...],
    after_point_left_base_m: list[float] | tuple[float, ...] | None,
    *,
    commanded_delta_left_base_m: tuple[float, float, float] = (0.0, 0.0, 0.02),
    source_vacated: bool | None = None,
    maximum_tracking_error_mm: float = 8.0,
    minimum_vertical_lift_mm: float = 12.0,
    maximum_lateral_motion_mm: float = 8.0,
) -> VisualProbeEvidence:
    """Check whether the detected carton followed the 20 mm TCP probe lift.

    The comparison happens in the calibrated left-base frame, so it does not
    rely on a particular camera orientation.  A missing/occluded after-lift
    detection fails closed; acoustic evidence is never allowed to replace it.
    """

    command = np.asarray(commanded_delta_left_base_m, dtype=np.float64)
    before = np.asarray(before_point_left_base_m, dtype=np.float64)
    if command.shape != (3,) or before.shape != (3,):
        raise ValueError("before point and commanded delta must contain 3 values")
    if not np.all(np.isfinite(command)) or not np.all(np.isfinite(before)):
        raise ValueError("probe inputs contain non-finite values")
    if after_point_left_base_m is None:
        return VisualProbeEvidence(
            passed=False,
            commanded_delta_m=tuple(float(value) for value in command),
            observed_delta_m=None,
            tracking_error_mm=None,
            observed_vertical_lift_mm=None,
            observed_lateral_motion_mm=None,
            source_vacated=source_vacated,
            reason="carton was not localized after the probe lift",
        )

    after = np.asarray(after_point_left_base_m, dtype=np.float64)
    if after.shape != (3,) or not np.all(np.isfinite(after)):
        raise ValueError("after point must contain 3 finite values")
    observed = after - before
    error_mm = float(np.linalg.norm(observed - command) * 1000.0)
    vertical_mm = float(observed[2] * 1000.0)
    lateral_mm = float(np.linalg.norm(observed[:2]) * 1000.0)
    reasons: list[str] = []
    if error_mm > maximum_tracking_error_mm:
        reasons.append(
            f"carton/TCP tracking error {error_mm:.1f}mm exceeds "
            f"{maximum_tracking_error_mm:.1f}mm"
        )
    if vertical_mm < minimum_vertical_lift_mm:
        reasons.append(
            f"observed vertical lift {vertical_mm:.1f}mm is below "
            f"{minimum_vertical_lift_mm:.1f}mm"
        )
    if lateral_mm > maximum_lateral_motion_mm:
        reasons.append(
            f"observed lateral motion {lateral_mm:.1f}mm exceeds "
            f"{maximum_lateral_motion_mm:.1f}mm"
        )
    if source_vacated is False:
        reasons.append("the original tabletop location still appears occupied")
    passed = not reasons
    return VisualProbeEvidence(
        passed=passed,
        commanded_delta_m=tuple(float(value) for value in command),
        observed_delta_m=tuple(float(value) for value in observed),
        tracking_error_mm=error_mm,
        observed_vertical_lift_mm=vertical_mm,
        observed_lateral_motion_mm=lateral_mm,
        source_vacated=source_vacated,
        reason="carton followed the probe lift" if passed else "; ".join(reasons),
    )


def fuse_hold_evidence(
    visual: VisualProbeEvidence,
    *,
    acoustic: AcousticDecision = "unavailable",
    acoustic_policy: AcousticPolicy = "supporting",
) -> HoldDecision:
    """Fuse evidence conservatively.

    Visual following during the physical 20 mm probe is always mandatory.
    In ``supporting`` mode a calibrated sound detector adds evidence but cannot
    veto a successful visual check.  ``required`` mode is available only after
    a microphone calibration set has demonstrated acceptable error rates.
    """

    if acoustic not in {"attached", "not_attached", "unavailable", "uncertain"}:
        raise ValueError(f"invalid acoustic decision: {acoustic}")
    if acoustic_policy not in {"supporting", "required"}:
        raise ValueError(f"invalid acoustic policy: {acoustic_policy}")
    if not visual.passed:
        return HoldDecision(
            passed=False,
            visual=visual,
            acoustic=acoustic,
            acoustic_policy=acoustic_policy,
            reason=f"visual probe verification failed: {visual.reason}",
        )
    if acoustic_policy == "required" and acoustic != "attached":
        return HoldDecision(
            passed=False,
            visual=visual,
            acoustic=acoustic,
            acoustic_policy=acoustic_policy,
            reason=f"required acoustic verifier returned {acoustic}",
        )
    suffix = (
        " with matching acoustic evidence"
        if acoustic == "attached"
        else f"; acoustic evidence is {acoustic} and remains non-blocking"
    )
    return HoldDecision(
        passed=True,
        visual=visual,
        acoustic=acoustic,
        acoustic_policy=acoustic_policy,
        reason="visual probe verification passed" + suffix,
    )

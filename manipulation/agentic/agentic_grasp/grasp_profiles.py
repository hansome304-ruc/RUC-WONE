"""Named grasp strategy profiles.

Tune these values first when adapting the pipeline to a new object class or
table setup. Distances are metres, axes are in the active arm base frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


StrategyName = Literal["default", "low-bottle", "side-bottle"]


@dataclass(frozen=True)
class GraspProfile:
    name: StrategyName
    description: str
    grasp_z_min: float
    grasp_z_max: float
    grasp_pre_z_min: float
    workspace_z_min: float
    max_abs_entry_z: float
    max_abs_closing_z: float | None
    pre_grasp_back: float | None = None
    side_overdrive: float = 0.0
    min_image_v_ratio: float | None = None
    target_image_v_ratio: float | None = None
    force_side_pose: bool = False


PROFILES: dict[StrategyName, GraspProfile] = {
    "default": GraspProfile(
        name="default",
        description="General ZeroGrasp mode for cluttered scenes and top/angled grasps.",
        grasp_z_min=0.12,
        grasp_z_max=0.24,
        grasp_pre_z_min=0.12,
        workspace_z_min=0.10,
        max_abs_entry_z=0.35,
        max_abs_closing_z=0.30,
    ),
    "low-bottle": GraspProfile(
        name="low-bottle",
        description="Low upright bottle mode; keeps ZeroGrasp orientation but accepts lower grasp points.",
        grasp_z_min=0.02,
        grasp_z_max=0.12,
        grasp_pre_z_min=0.02,
        workspace_z_min=0.02,
        max_abs_entry_z=0.35,
        max_abs_closing_z=0.30,
        pre_grasp_back=0.06,
        min_image_v_ratio=0.35,
        target_image_v_ratio=0.62,
    ),
    "side-bottle": GraspProfile(
        name="side-bottle",
        description="Clean-table bottle mode; forces a table-parallel side grasp.",
        grasp_z_min=0.02,
        grasp_z_max=0.12,
        grasp_pre_z_min=0.02,
        workspace_z_min=0.02,
        max_abs_entry_z=0.35,
        max_abs_closing_z=None,
        pre_grasp_back=0.06,
        side_overdrive=0.015,
        min_image_v_ratio=0.35,
        target_image_v_ratio=0.62,
        force_side_pose=True,
    ),
    "template": GraspProfile(
        name="template",
        description="Template object mode; prefers stable top/angled grasps with normal object height.",
        grasp_z_min=0.05,
        grasp_z_max=0.20,
        grasp_pre_z_min=0.05,
        workspace_z_min=0.05,
        max_abs_entry_z=0.80,
        max_abs_closing_z=0.50,
        pre_grasp_back=0.08,
        min_image_v_ratio=None,
        target_image_v_ratio=0.50,
        force_side_pose=False,
    ),
}


def get_profile(name: StrategyName) -> GraspProfile:
    return PROFILES[name]

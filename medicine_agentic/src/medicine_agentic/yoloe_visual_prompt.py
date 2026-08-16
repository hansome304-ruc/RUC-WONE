"""YOLOE visual-prompt detector for Task 1 medicine cartons.

The detector is intentionally an optional plugin.  Importing the main
``medicine_agentic`` package does not import Ultralytics or Torch.  A fixed
reference image supplies one or more labelled carton faces, so adding this
backend does not require a training run or a labelled dataset.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from medicine_agentic.reference_faces import ReferenceFaceBank
from medicine_agentic.task1_box import BoxCandidate


@dataclass(frozen=True)
class FacePrompt:
    class_id: int
    face_type: str
    reference_face_id: str
    bbox_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class MotifTemplate:
    image_path: Path
    gray: np.ndarray
    face_type: str
    reference_face_id: str
    box_center_offset_px: tuple[float, float]
    box_size_px: tuple[float, float]
    box_angle_deg: float
    min_score: float
    angle_min_deg: float
    angle_max_deg: float
    angle_step_deg: float
    scale_min: float
    scale_max: float
    scale_steps: int


def _resolve_path(value: Any, *, config_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _normalized_roi(
    shape: tuple[int, int],
    values: Any,
) -> tuple[int, int, int, int]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("roi_norm must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in values)
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"invalid normalized ROI: {values}")
    height, width = shape
    return (
        int(round(x1 * width)),
        int(round(y1 * height)),
        int(round(x2 * width)),
        int(round(y2 * height)),
    )


def _long_axis_angle(rect: tuple[Any, Any, float]) -> float:
    (_, _), (width, height), angle = rect
    result = float(angle if width >= height else angle + 90.0)
    while result >= 90.0:
        result -= 180.0
    while result < -90.0:
        result += 180.0
    return result


def _bbox_from_polygon(
    polygon: tuple[tuple[float, float], ...],
) -> tuple[float, float, float, float]:
    points = np.asarray(polygon, dtype=np.float32)
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    return intersection / max(first_area + second_area - intersection, 1e-6)


def _candidate_from_polygon(
    polygon: np.ndarray,
    *,
    confidence: float,
    prompt: FacePrompt,
    frame_shape: tuple[int, int],
    roi_px: tuple[int, int, int, int],
    options: dict[str, Any],
    pink_mask: np.ndarray | None = None,
) -> BoxCandidate | None:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if len(contour) < 3 or not np.isfinite(contour).all():
        return None

    rect = cv2.minAreaRect(contour)
    center = (float(rect[0][0]), float(rect[0][1]))
    raw_width, raw_height = (float(rect[1][0]), float(rect[1][1]))
    long_side = max(raw_width, raw_height)
    short_side = min(raw_width, raw_height)
    rect_area = long_side * short_side
    if short_side <= 0.0 or not math.isfinite(rect_area):
        return None

    x1, y1, x2, y2 = roi_px
    if not (x1 <= center[0] <= x2 and y1 <= center[1] <= y2):
        return None

    # Use image-normalized sanity limits. Fixed pixel areas made a correct
    # visual-prompt result disappear whenever capture resolution or apparent
    # scale changed (for example, 848x480 -> 1280x720).
    frame_area = float(frame_shape[0] * frame_shape[1])
    area_fraction = rect_area / max(frame_area, 1.0)
    min_area_fraction = float(options.get("min_box_area_fraction", 0.0002))
    max_area_fraction = float(options.get("max_box_area_fraction", 0.25))
    if not (min_area_fraction <= area_fraction <= max_area_fraction):
        return None

    aspect = long_side / short_side
    min_aspect = float(options.get("min_box_aspect_ratio", 1.15))
    max_aspect = float(options.get("max_box_aspect_ratio", 2.0))
    if not (min_aspect <= aspect <= max_aspect):
        return None

    contour_area = abs(float(cv2.contourArea(contour)))
    rectangularity = min(1.0, contour_area / max(rect_area, 1e-6))
    if rectangularity < float(options.get("min_mask_rectangularity", 0.78)):
        return None

    contour_int = np.round(contour).astype(np.int32)
    pink_fraction = 1.0
    if pink_mask is not None:
        if pink_mask.shape != frame_shape:
            raise ValueError(
                f"pink mask shape {pink_mask.shape} does not match {frame_shape}"
            )
        instance_mask = np.zeros(frame_shape, dtype=np.uint8)
        cv2.fillPoly(instance_mask, [contour_int], 255)
        instance_pixels = instance_mask > 0
        instance_area = int(np.count_nonzero(instance_pixels))
        if instance_area <= 0:
            return None
        pink_fraction = float(
            np.count_nonzero((pink_mask > 0) & instance_pixels) / instance_area
        )
        # Colour is diagnostic by default, not an identity gate. The visual
        # prompt supplies carton identity; lighting/white balance should not
        # silently delete an otherwise valid instance mask.
        if bool(options.get("verify_pink_color", False)) and pink_fraction < float(
            options.get("min_pink_fraction", 0.50)
        ):
            return None

    bx, by, bw, bh = cv2.boundingRect(contour_int)
    if bw <= 0 or bh <= 0:
        return None
    component = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
    local_contour = contour_int - np.asarray([bx - 1, by - 1], dtype=np.int32)
    cv2.fillPoly(component, [local_contour], 255)
    distance = cv2.distanceTransform(component, cv2.DIST_L2, 5)
    _, clearance, _, suction_local = cv2.minMaxLoc(distance)
    suction = (int(suction_local[0] + bx - 1), int(suction_local[1] + by - 1))

    rectangle = cv2.boxPoints(rect)
    rectangle_polygon = tuple(
        (float(point[0]), float(point[1])) for point in rectangle
    )
    return BoxCandidate(
        center_px=center,
        suction_px=suction,
        polygon_px=rectangle_polygon,
        long_side_px=long_side,
        short_side_px=short_side,
        angle_deg=_long_axis_angle(rect),
        rectangularity=rectangularity,
        bright_fill=pink_fraction,
        edge_clearance_px=float(clearance),
        score=float(confidence),
        provider="yoloe_visual_prompt",
        face_type=prompt.face_type,
        face_score=float(confidence),
        reference_face_id=prompt.reference_face_id,
        graspable=False,
        grasp_blockers=(),
    )


def _deduplicate(
    candidates: list[BoxCandidate],
    *,
    iou_threshold: float,
) -> list[BoxCandidate]:
    kept: list[BoxCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        candidate_bbox = _bbox_from_polygon(candidate.polygon_px)
        if any(
            _bbox_iou(candidate_bbox, _bbox_from_polygon(other.polygon_px))
            >= iou_threshold
            for other in kept
        ):
            continue
        kept.append(candidate)
    return kept


def _surface_evidence(
    polygon: np.ndarray,
    *,
    pink_mask: np.ndarray,
) -> tuple[float, float, tuple[int, int], float] | None:
    """Return visible fraction, pink fraction, safest pixel, and clearance."""

    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    expected_area = abs(float(cv2.contourArea(contour)))
    if expected_area <= 1.0:
        return None
    contour_int = np.round(contour).astype(np.int32)
    x, y, width, height = cv2.boundingRect(contour_int)
    frame_height, frame_width = pink_mask.shape
    crop_x1 = max(0, x)
    crop_y1 = max(0, y)
    crop_x2 = min(frame_width, x + width)
    crop_y2 = min(frame_height, y + height)
    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
        return None

    local = contour_int - np.asarray([crop_x1, crop_y1], dtype=np.int32)
    instance = np.zeros(
        (crop_y2 - crop_y1, crop_x2 - crop_x1),
        dtype=np.uint8,
    )
    cv2.fillPoly(instance, [local], 255)
    visible_area = int(np.count_nonzero(instance))
    if visible_area <= 0:
        return None
    visible_fraction = min(1.0, visible_area / expected_area)
    pink_crop = pink_mask[crop_y1:crop_y2, crop_x1:crop_x2]
    surface = np.where((instance > 0) & (pink_crop > 0), 255, 0).astype(
        np.uint8
    )
    # A cropped mask may be non-zero at its border.  Distance transform does
    # not know about pixels outside the array, so explicitly add a zero rim;
    # otherwise it can incorrectly choose a suction point on the box edge.
    surface[0, :] = 0
    surface[-1, :] = 0
    surface[:, 0] = 0
    surface[:, -1] = 0
    pink_fraction = float(np.count_nonzero(surface) / visible_area)
    distance = cv2.distanceTransform(surface, cv2.DIST_L2, 5)
    _, clearance, _, safest_local = cv2.minMaxLoc(distance)
    safest = (
        int(safest_local[0] + crop_x1),
        int(safest_local[1] + crop_y1),
    )
    return visible_fraction, pink_fraction, safest, float(clearance)


def _normalise_angle(angle_deg: float) -> float:
    result = float(angle_deg)
    while result >= 90.0:
        result -= 180.0
    while result < -90.0:
        result += 180.0
    return result


class YOLOEVisualPromptDetector:
    """Detect front/back carton faces from a fixed visual-prompt scene."""

    name = "yoloe_visual_prompt"

    def __init__(
        self,
        *,
        options: dict[str, Any],
        config_dir: Path,
        face_bank: ReferenceFaceBank | None,
    ) -> None:
        self._options = dict(options)
        self._model_path = _resolve_path(
            self._options.get("model_path"),
            config_dir=config_dir,
            label="model_path",
        )
        self._reference_image_path = _resolve_path(
            self._options.get("reference_image"),
            config_dir=config_dir,
            label="reference_image",
        )
        self._reference_bgr = cv2.imread(
            str(self._reference_image_path),
            cv2.IMREAD_COLOR,
        )
        if self._reference_bgr is None:
            raise ValueError(
                f"reference image cannot be decoded: {self._reference_image_path}"
            )

        prompt_settings = self._options.get("face_prompts")
        if not isinstance(prompt_settings, list) or not prompt_settings:
            raise ValueError("face_prompts must be a non-empty list")

        class_by_face: dict[str, int] = {}
        prompts: list[FacePrompt] = []
        for index, setting in enumerate(prompt_settings):
            if not isinstance(setting, dict):
                raise ValueError(f"face_prompts[{index}] must be an object")
            face_type = str(setting.get("face_type", "")).strip()
            reference_face_id = str(
                setting.get("reference_face_id", "")
            ).strip()
            bbox_setting = setting.get("bbox_xyxy")
            if not face_type or not reference_face_id:
                raise ValueError(
                    f"face_prompts[{index}] requires face_type and "
                    "reference_face_id"
                )
            if not isinstance(bbox_setting, (list, tuple)) or len(bbox_setting) != 4:
                raise ValueError(
                    f"face_prompts[{index}].bbox_xyxy must contain four values"
                )
            bbox = tuple(float(value) for value in bbox_setting)
            if not (
                0.0 <= bbox[0] < bbox[2] <= self._reference_bgr.shape[1]
                and 0.0 <= bbox[1] < bbox[3] <= self._reference_bgr.shape[0]
            ):
                raise ValueError(
                    f"face_prompts[{index}].bbox_xyxy is outside the "
                    "reference image"
                )
            class_id = class_by_face.setdefault(face_type, len(class_by_face))
            prompt = FacePrompt(
                class_id=class_id,
                face_type=face_type,
                reference_face_id=reference_face_id,
                bbox_xyxy=bbox,
            )
            if face_bank is not None:
                reference = face_bank.face_by_id(reference_face_id)
                if str(reference.face_type) != face_type:
                    raise ValueError(
                        f"{reference_face_id} belongs to {reference.face_type}, "
                        f"not {face_type}"
                    )
            prompts.append(prompt)

        from ultralytics import YOLOE
        from ultralytics.models.yolo.yoloe.predict import YOLOEVPSegPredictor

        self._model = YOLOE(str(self._model_path))
        self._seg_predictor = YOLOEVPSegPredictor
        self._prompts = tuple(prompts)
        self._prompt_by_class = {
            prompt.class_id: prompt for prompt in reversed(self._prompts)
        }
        self._visual_prompts = {
            "bboxes": [list(prompt.bbox_xyxy) for prompt in self._prompts],
            "cls": [prompt.class_id for prompt in self._prompts],
        }
        self._motifs = self._load_motifs(
            self._options.get("motif_templates", []),
            config_dir=config_dir,
            face_bank=face_bank,
        )
        self._embedding_ready = False
        self._inference_count = 0
        self._last_latency_ms: float | None = None

    def _load_motifs(
        self,
        settings: Any,
        *,
        config_dir: Path,
        face_bank: ReferenceFaceBank | None,
    ) -> tuple[MotifTemplate, ...]:
        if settings is None:
            return ()
        if not isinstance(settings, list):
            raise ValueError("motif_templates must be a list")
        motifs: list[MotifTemplate] = []
        for index, setting in enumerate(settings):
            if not isinstance(setting, dict):
                raise ValueError(f"motif_templates[{index}] must be an object")
            image_path = _resolve_path(
                setting.get("image"),
                config_dir=config_dir,
                label=f"motif_templates[{index}].image",
            )
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None or min(gray.shape) < 8:
                raise ValueError(f"invalid motif image: {image_path}")
            face_type = str(setting.get("face_type", "")).strip()
            reference_face_id = str(
                setting.get("reference_face_id", "")
            ).strip()
            if not face_type or not reference_face_id:
                raise ValueError(
                    f"motif_templates[{index}] requires face_type and "
                    "reference_face_id"
                )
            if face_bank is not None:
                reference = face_bank.face_by_id(reference_face_id)
                if str(reference.face_type) != face_type:
                    raise ValueError(
                        f"{reference_face_id} belongs to {reference.face_type}, "
                        f"not {face_type}"
                    )

            offset = setting.get("box_center_offset_px")
            size = setting.get("box_size_px")
            if not isinstance(offset, (list, tuple)) or len(offset) != 2:
                raise ValueError(
                    f"motif_templates[{index}].box_center_offset_px "
                    "must contain two values"
                )
            if not isinstance(size, (list, tuple)) or len(size) != 2:
                raise ValueError(
                    f"motif_templates[{index}].box_size_px must contain two values"
                )
            box_size = (float(size[0]), float(size[1]))
            if min(box_size) <= 0.0:
                raise ValueError("motif box dimensions must be positive")
            angle_step = float(setting.get("angle_step_deg", 4.0))
            scale_steps = int(setting.get("scale_steps", 11))
            if angle_step <= 0.0 or scale_steps < 1:
                raise ValueError("motif angle/scale sampling is invalid")
            motifs.append(
                MotifTemplate(
                    image_path=image_path,
                    gray=gray,
                    face_type=face_type,
                    reference_face_id=reference_face_id,
                    box_center_offset_px=(
                        float(offset[0]),
                        float(offset[1]),
                    ),
                    box_size_px=box_size,
                    box_angle_deg=float(setting.get("box_angle_deg", 0.0)),
                    min_score=float(setting.get("min_score", 0.36)),
                    angle_min_deg=float(setting.get("angle_min_deg", -20.0)),
                    angle_max_deg=float(setting.get("angle_max_deg", 20.0)),
                    angle_step_deg=angle_step,
                    scale_min=float(setting.get("scale_min", 0.75)),
                    scale_max=float(setting.get("scale_max", 1.25)),
                    scale_steps=scale_steps,
                )
            )
        return tuple(motifs)

    @staticmethod
    def _edge_map(gray: np.ndarray) -> np.ndarray:
        return cv2.Canny(
            cv2.GaussianBlur(gray, (3, 3), 0),
            30,
            100,
        )

    def _detect_motifs(
        self,
        rgb: np.ndarray,
        *,
        pink_mask: np.ndarray,
    ) -> list[BoxCandidate]:
        if not self._motifs:
            return []
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        scene_edges_full = self._edge_map(gray)
        roi_x0, roi_y0, roi_x1, roi_y1 = _normalized_roi(
            rgb.shape[:2],
            self._options.get("roi_norm", [0.0, 0.0, 1.0, 1.0]),
        )
        scene_edges = scene_edges_full[roi_y0:roi_y1, roi_x0:roi_x1]
        raw: list[
            tuple[
                float,
                MotifTemplate,
                tuple[float, float],
                tuple[float, float],
                float,
                np.ndarray,
            ]
        ] = []
        peaks_per_variant = int(
            self._options.get("motif_max_peaks_per_variant", 4)
        )
        for motif in self._motifs:
            angles = np.arange(
                motif.angle_min_deg,
                motif.angle_max_deg + motif.angle_step_deg * 0.5,
                motif.angle_step_deg,
            )
            scales = np.linspace(
                motif.scale_min,
                motif.scale_max,
                motif.scale_steps,
            )
            motif_height, motif_width = motif.gray.shape
            motif_center = np.asarray(
                [(motif_width - 1) / 2.0, (motif_height - 1) / 2.0],
                dtype=np.float32,
            )
            offset = np.asarray(motif.box_center_offset_px, dtype=np.float32)
            for angle in angles:
                matrix = cv2.getRotationMatrix2D(
                    tuple(float(value) for value in motif_center),
                    float(angle),
                    1.0,
                )
                rotated = cv2.warpAffine(
                    motif.gray,
                    matrix,
                    (motif_width, motif_height),
                    borderMode=cv2.BORDER_REPLICATE,
                )
                rotated_offset = matrix[:, :2] @ offset
                for scale in scales:
                    interpolation = (
                        cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                    )
                    template = cv2.resize(
                        rotated,
                        None,
                        fx=float(scale),
                        fy=float(scale),
                        interpolation=interpolation,
                    )
                    template_edges = self._edge_map(template)
                    height, width = template_edges.shape
                    if (
                        height >= scene_edges.shape[0]
                        or width >= scene_edges.shape[1]
                    ):
                        continue
                    response = cv2.matchTemplate(
                        scene_edges,
                        template_edges,
                        cv2.TM_CCOEFF_NORMED,
                    )
                    work = response.copy()
                    for _ in range(max(1, peaks_per_variant)):
                        _, score, _, location = cv2.minMaxLoc(work)
                        if float(score) < motif.min_score:
                            break
                        x, y = location
                        match_center = np.asarray(
                            [
                                roi_x0 + x + width / 2.0,
                                roi_y0 + y + height / 2.0,
                            ],
                            dtype=np.float32,
                        )
                        box_center = match_center + rotated_offset * float(scale)
                        box_size = (
                            motif.box_size_px[0] * float(scale),
                            motif.box_size_px[1] * float(scale),
                        )
                        box_angle = motif.box_angle_deg + float(angle)
                        rectangle = (
                            (float(box_center[0]), float(box_center[1])),
                            (float(box_size[0]), float(box_size[1])),
                            float(box_angle),
                        )
                        polygon = cv2.boxPoints(rectangle)
                        raw.append(
                            (
                                float(score),
                                motif,
                                (
                                    float(box_center[0]),
                                    float(box_center[1]),
                                ),
                                box_size,
                                box_angle,
                                polygon,
                            )
                        )
                        suppression = max(4, int(round(min(width, height) * 0.5)))
                        cv2.circle(work, location, suppression, -1.0, -1)

        raw.sort(key=lambda item: item[0], reverse=True)
        geometry_kept: list[
            tuple[
                float,
                MotifTemplate,
                tuple[float, float],
                tuple[float, float],
                float,
                np.ndarray,
            ]
        ] = []
        nms_threshold = float(self._options.get("motif_deduplicate_iou", 0.40))
        for item in raw:
            polygon = tuple(
                (float(point[0]), float(point[1])) for point in item[5]
            )
            bbox = _bbox_from_polygon(polygon)
            if any(
                _bbox_iou(
                    bbox,
                    _bbox_from_polygon(
                        tuple(
                            (float(point[0]), float(point[1]))
                            for point in kept[5]
                        )
                    ),
                )
                >= nms_threshold
                for kept in geometry_kept
            ):
                continue
            geometry_kept.append(item)
            if len(geometry_kept) >= int(
                self._options.get("motif_max_raw_candidates", 40)
            ):
                break

        candidates: list[BoxCandidate] = []
        minimum_visible = float(
            self._options.get("motif_min_visible_fraction", 0.85)
        )
        minimum_pink = float(
            self._options.get("motif_min_pink_fraction", 0.50)
        )
        for score, motif, center, box_size, angle, polygon in geometry_kept:
            evidence = _surface_evidence(polygon, pink_mask=pink_mask)
            if evidence is None:
                continue
            visible, pink_fraction, suction, clearance = evidence
            if visible < minimum_visible or pink_fraction < minimum_pink:
                continue
            long_side = max(box_size)
            short_side = min(box_size)
            candidates.append(
                BoxCandidate(
                    center_px=center,
                    suction_px=suction,
                    polygon_px=tuple(
                        (float(point[0]), float(point[1])) for point in polygon
                    ),
                    long_side_px=float(long_side),
                    short_side_px=float(short_side),
                    angle_deg=_normalise_angle(
                        angle if box_size[0] >= box_size[1] else angle + 90.0
                    ),
                    rectangularity=1.0,
                    bright_fill=pink_fraction,
                    edge_clearance_px=clearance,
                    score=score,
                    provider="front_motif_template",
                    face_type=motif.face_type,
                    face_score=score,
                    reference_face_id=motif.reference_face_id,
                    graspable=False,
                    grasp_blockers=(),
                )
            )
        return candidates

    def detect(self, rgb: np.ndarray) -> list[BoxCandidate]:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be uint8 HxWx3, got {rgb.dtype} {rgb.shape}")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        predict_options = {
            "conf": float(self._options.get("model_confidence", 0.03)),
            "iou": float(self._options.get("model_iou", 0.35)),
            "imgsz": int(self._options.get("image_size", 640)),
            "device": str(self._options.get("device", "cpu")),
            "verbose": False,
        }
        started = time.perf_counter()
        if not self._embedding_ready:
            results = self._model.predict(
                bgr,
                visual_prompts=self._visual_prompts,
                refer_image=self._reference_bgr,
                predictor=self._seg_predictor,
                **predict_options,
            )
            self._embedding_ready = True
        else:
            results = self._model.predict(bgr, **predict_options)
        self._inference_count += 1

        roi_px = _normalized_roi(
            rgb.shape[:2],
            self._options.get("roi_norm", [0.0, 0.0, 1.0, 1.0]),
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hue, saturation, value = cv2.split(hsv)
        pink_mask = np.where(
            (
                (hue >= int(self._options.get("pink_hue_min", 130)))
                & (hue <= int(self._options.get("pink_hue_max", 175)))
                & (
                    saturation
                    >= int(self._options.get("pink_saturation_min", 8))
                )
                & (
                    saturation
                    <= int(self._options.get("pink_saturation_max", 130))
                )
                & (value >= int(self._options.get("pink_value_min", 130)))
            ),
            255,
            0,
        ).astype(np.uint8)
        proposal_minimum = float(
            self._options.get("proposal_min_confidence", 0.10)
        )
        candidates: list[BoxCandidate] = []
        if results:
            result = results[0]
            if result.boxes is not None and result.masks is not None:
                boxes = result.boxes
                for confidence, class_value, polygon in zip(
                    boxes.conf.cpu().tolist(),
                    boxes.cls.cpu().tolist(),
                    result.masks.xy,
                ):
                    if float(confidence) < proposal_minimum:
                        continue
                    prompt = self._prompt_by_class.get(int(class_value))
                    if prompt is None:
                        continue
                    candidate = _candidate_from_polygon(
                        np.asarray(polygon, dtype=np.float32),
                        confidence=float(confidence),
                        prompt=prompt,
                        frame_shape=rgb.shape[:2],
                        roi_px=roi_px,
                        options=self._options,
                        pink_mask=pink_mask,
                    )
                    if candidate is not None:
                        candidates.append(candidate)

        candidates.extend(self._detect_motifs(rgb, pink_mask=pink_mask))
        final_candidates = _deduplicate(
            candidates,
            iou_threshold=float(self._options.get("deduplicate_iou", 0.45)),
        )[: int(self._options.get("max_candidates", 12))]
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        return final_candidates

    def status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ok": True,
            "mode": "visual_prompt_instance_segmentation",
            "model": self._model_path.name,
            "reference_image": self._reference_image_path.name,
            "face_types": sorted(
                {prompt.face_type for prompt in self._prompts}
            ),
            "motif_templates": [
                motif.image_path.name for motif in self._motifs
            ],
            "embedding_ready": self._embedding_ready,
            "inference_count": self._inference_count,
            "last_latency_ms": self._last_latency_ms,
            "device": str(self._options.get("device", "cpu")),
        }


def create_detector(
    *,
    options: dict[str, Any],
    config_dir: Path,
    face_bank: ReferenceFaceBank | None,
) -> YOLOEVisualPromptDetector:
    """Plugin factory used by ``create_detector_provider``."""

    return YOLOEVisualPromptDetector(
        options=options,
        config_dir=config_dir,
        face_bank=face_bank,
    )

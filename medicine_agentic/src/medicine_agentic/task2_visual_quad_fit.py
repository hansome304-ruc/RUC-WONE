import json
import itertools
import math
import os
from pathlib import Path

import cv2
import numpy as np


CAPTURE = Path(os.environ.get("TASK2_CAPTURE", "/tmp/task2_boundary_capture"))
SIFT_REPORT = Path(os.environ.get("TASK2_SIFT_REPORT", "/tmp/task2_multi_sift/report.json"))
OUT = Path(os.environ.get("TASK2_VISUAL_OUT", "/tmp/task2_visual_quad"))
OUT.mkdir(parents=True, exist_ok=True)

color = cv2.imread(str(CAPTURE / "color.png"), cv2.IMREAD_COLOR)
depth = cv2.imread(str(CAPTURE / "depth.png"), cv2.IMREAD_UNCHANGED)
meta = json.loads((CAPTURE / "meta.json").read_text())
report = json.loads(SIFT_REPORT.read_text())
anchors = sorted([np.asarray(item["polygon_px"], np.float32) for item in report["instances"]], key=lambda q: q.mean(0)[0])
if color is None or depth is None or len(anchors) < 2:
    raise RuntimeError("Need a captured RGB-D frame and at least two SIFT anchors")
if os.environ.get("TASK2_FORCE_LEFT_RECOVERY") == "1":
    if len(anchors) < 3:
        raise RuntimeError("Forced left recovery needs the three right-hand anchors")
    anchors = anchors[-3:]
h, w = color.shape[:2]


def rect_axes(polygon):
    (_, _), (a, b), angle = cv2.minAreaRect(polygon)
    if a > b:
        short, long = b, a
        angle += 90.0
    else:
        short, long = a, b
    theta = math.radians(angle)
    u = np.asarray([math.cos(theta), math.sin(theta)], np.float32)
    v = np.asarray([-math.sin(theta), math.cos(theta)], np.float32)
    # u must be the short, mostly horizontal axis; v the long, downward axis.
    if abs(u[0]) < abs(v[0]):
        u, v = v, u
    if u[0] < 0:
        u = -u
    if v[1] < 0:
        v = -v
    return float(short), float(long), u, v


geometry = [rect_axes(q) for q in anchors]
expected_short = float(np.median([g[0] for g in geometry]))
expected_long = float(np.median([g[1] for g in geometry]))
u = np.mean([g[2] for g in geometry], axis=0); u /= np.linalg.norm(u)
v = np.mean([g[3] for g in geometry], axis=0); v /= np.linalg.norm(v)
centers = np.asarray([q.mean(0) for q in anchors], np.float32)
target_total = int(os.environ.get("TASK2_TARGET_TOTAL", "4"))
if target_total not in (3, 4):
    raise RuntimeError("Target total must be three or four cartons")

# Establish only a search centre. The generic wrapper supplies one from a scan
# of all unoccupied pink-face regions, so no equal-spacing slot is assumed.
# A caller may explicitly test one topology. The generic wrapper tests every
# topology and lets actual boundary evidence decide which carton is missing.
best = None
requested_prior = os.environ.get("TASK2_PRIOR_CENTER")
if requested_prior:
    prior_center = np.asarray([float(value) for value in requested_prior.split(",")], np.float32)
    if prior_center.shape != (2,):
        raise RuntimeError("TASK2_PRIOR_CENTER must be x,y")
    positions = tuple()
    missing_position = int(np.sum(centers[:, 0] < prior_center[0]))
    assignment_residual = 0.0
requested_positions = os.environ.get("TASK2_ANCHOR_POSITIONS")
if requested_prior:
    position_candidates = []
elif requested_positions:
    position_candidates = [tuple(int(value) for value in requested_positions.split(","))]
else:
    position_candidates = list(itertools.combinations(range(4), len(anchors)))
for positions in position_candidates:
    if len(positions) != len(anchors) or tuple(sorted(positions)) != positions or any(value < 0 or value >= target_total for value in positions):
        continue
    design = np.c_[np.ones(len(positions)), np.asarray(positions, np.float32)]
    cx, _, _, _ = np.linalg.lstsq(design, centers[:, 0], rcond=None)
    cy, _, _, _ = np.linalg.lstsq(design, centers[:, 1], rcond=None)
    predicted_x = cx[0] + cx[1] * np.arange(target_total, dtype=np.float32)
    residual = float(np.mean((design @ cx - centers[:, 0]) ** 2 + (design @ cy - centers[:, 1]) ** 2))
    step = float(cx[1])
    task_center_min = 0.42 * w
    task_center_max = 0.71 * w
    if not (task_center_min <= predicted_x[0] and predicted_x[-1] <= task_center_max):
        continue
    if 0.82 * expected_short <= step <= 1.45 * expected_short and (best is None or residual < best[0]):
        best = (residual, positions, cx, cy)
if not requested_prior:
    if best is None:
        raise RuntimeError("Cannot assign anchors")
    assignment_residual, positions, coeff_x, coeff_y = best
    missing = [index for index in range(target_total) if index not in positions]
    if len(missing) != 1:
        raise RuntimeError("This visual-line fitter expects exactly one missing box")
    missing_position = missing[0]
    prior_center = np.asarray(
        [coeff_x[0] + coeff_x[1] * missing_position, coeff_y[0] + coeff_y[1] * missing_position],
        np.float32,
    )

# Learn the face appearance from inset areas of the three reliable boxes.
lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)
positive_mask = np.zeros((h, w), np.uint8)
flap_mask = np.zeros((h, w), np.uint8)
outside_mask = np.zeros((h, w), np.uint8)
for polygon, geometry_item in zip(anchors, geometry):
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    short_value, long_value, local_u, local_v = geometry_item
    center_value = polygon.mean(axis=0)

    def local_quad(x0, x1, y0, y1):
        return np.asarray(
            [
                center_value + local_u * (x0 * short_value) + local_v * (y0 * long_value),
                center_value + local_u * (x1 * short_value) + local_v * (y0 * long_value),
                center_value + local_u * (x1 * short_value) + local_v * (y1 * long_value),
                center_value + local_u * (x0 * short_value) + local_v * (y1 * long_value),
            ],
            np.float32,
        )

    # Learn the printed pink face from the body, not from pale top regions.
    face_core = local_quad(-0.38, 0.38, -0.28, 0.40)
    cv2.fillConvexPoly(positive_mask, np.rint(face_core).astype(np.int32), 1)
    # The opened white flap is immediately above the true top edge. Make it a
    # named negative class so its outer edge cannot masquerade as the face top.
    flap_strip = local_quad(-0.43, 0.43, -0.69, -0.53)
    cv2.fillConvexPoly(flap_mask, np.rint(flap_strip).astype(np.int32), 1)
    outside_mask |= cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))) - mask
flap_mask[positive_mask > 0] = 0
outside_mask[(positive_mask > 0) | (flap_mask > 0)] = 0
rng = np.random.default_rng(7103)


def clusters(mask, count):
    values = lab[mask > 0].reshape(-1, 3)
    if len(values) > 8000:
        values = values[rng.choice(len(values), 8000, False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.25)
    _, _, result = cv2.kmeans(values.astype(np.float32), count, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
    return result


positive_centers = clusters(positive_mask, 8)
flap_centers = clusters(flap_mask, 5)
outside_centers = clusters(outside_mask, 8)
# Edge scoring and the final presence gate only sample near this candidate.
# Computing cluster distances over the full 1280x720 frame dominated the
# glare fallback, especially when several candidate priors ran concurrently.
probability_radius_x = int(expected_short * 1.05) + 24
probability_radius_y = int(expected_long * 0.85) + 24
px0 = max(0, int(prior_center[0] - probability_radius_x))
px1 = min(w, int(prior_center[0] + probability_radius_x) + 1)
py0 = max(0, int(prior_center[1] - probability_radius_y))
py1 = min(h, int(prior_center[1] + probability_radius_y) + 1)
local_lab = lab[py0:py1, px0:px1]
pd = np.min(np.sum((local_lab[:, :, None, :] - positive_centers[None, None, :, :]) ** 2, axis=3), axis=2)
fd = np.min(np.sum((local_lab[:, :, None, :] - flap_centers[None, None, :, :]) ** 2, axis=3), axis=2)
od = np.min(np.sum((local_lab[:, :, None, :] - outside_centers[None, None, :, :]) ** 2, axis=3), axis=2)
# Equal-priority negative classes: a large amount of green/yellow background
# must not drown out the smaller but crucial white-flap class.
nd = np.minimum(fd, od)
face_probability = np.zeros((h, w), dtype=np.float32)
face_probability[py0:py1, px0:px1] = 1.0 / (
    1.0 + np.exp(-np.clip((nd - pd) / 260.0, -10.0, 10.0))
)

gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
gradient = np.sqrt(gx * gx + gy * gy)
roi_radius_x = int(expected_short * 0.85)
roi_radius_y = int(expected_long * 0.72)
rx0 = max(0, int(prior_center[0] - roi_radius_x)); rx1 = min(w, int(prior_center[0] + roi_radius_x))
ry0 = max(0, int(prior_center[1] - roi_radius_y)); ry1 = min(h, int(prior_center[1] + roi_radius_y))
gradient_scale = max(float(np.percentile(gradient[ry0:ry1, rx0:rx1], 92)), 1.0)
gradient_norm = np.clip(gradient / gradient_scale, 0.0, 1.0)
depth_m = depth.astype(np.float32) * float(meta.get("depth_scale_m", 0.001))


def sample_bilinear(array, points):
    map_x = points[:, 0].astype(np.float32).reshape(-1, 1)
    map_y = points[:, 1].astype(np.float32).reshape(-1, 1)
    return cv2.remap(array, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).reshape(-1, *array.shape[2:])


def line_candidate(side, shift, delta_deg):
    if side in ("left", "right"):
        tangent0, normal0, length = v, u, expected_long
        base = -0.5 * expected_short if side == "left" else 0.5 * expected_short
        inward_sign = 1.0 if side == "left" else -1.0
    else:
        tangent0, normal0, length = u, v, expected_short
        base = -0.5 * expected_long if side == "top" else 0.5 * expected_long
        inward_sign = 1.0 if side == "top" else -1.0
    theta = math.radians(delta_deg)
    tangent = tangent0 * math.cos(theta) + np.asarray([-tangent0[1], tangent0[0]], np.float32) * math.sin(theta)
    tangent /= np.linalg.norm(tangent)
    normal = np.asarray([-tangent[1], tangent[0]], np.float32)
    if np.dot(normal, normal0) < 0:
        normal = -normal
    point = prior_center + normal0 * (base + shift)
    fractions = np.linspace(-0.42, 0.42, max(36, int(length * 0.50))).astype(np.float32)
    samples = point[None, :] + fractions[:, None] * length * tangent[None, :]
    inward = normal * inward_sign
    inside = samples + 5.0 * inward
    outside = samples - 5.0 * inward
    inner2 = samples + 9.0 * inward
    outer2 = samples - 9.0 * inward

    g = np.maximum.reduce([sample_bilinear(gradient_norm, samples + offset * normal).ravel() for offset in (-1.5, 0.0, 1.5)])
    li = sample_bilinear(lab, inside); lo = sample_bilinear(lab, outside)
    lab_delta = np.linalg.norm(li - lo, axis=1)
    pi = sample_bilinear(face_probability, inside).ravel(); po = sample_bilinear(face_probability, outside).ravel()
    pi2 = sample_bilinear(face_probability, inner2).ravel(); po2 = sample_bilinear(face_probability, outer2).ravel()
    zi = sample_bilinear(depth_m, inside).ravel(); zo = sample_bilinear(depth_m, outside).ravel()
    valid_depth = (zi > 0.2) & (zo > 0.2)
    depth_jump = np.zeros_like(zi)
    depth_jump[valid_depth] = np.abs(zo[valid_depth] - zi[valid_depth])

    gradient_continuity = float(np.mean(g >= 0.23))
    strong_gradient = float(np.mean(g >= 0.42))
    lab_continuity = float(np.mean(lab_delta >= 12.0))
    face_direction = float(np.mean(((pi + pi2) * 0.5 - (po + po2) * 0.5) >= 0.08))
    depth_continuity = float(np.mean(depth_jump >= 0.006))
    # Every side must be a long coherent image line. Colour and depth determine
    # which side of it is the carton face; either sensor may be weak locally.
    score = (
        4.6 * gradient_continuity
        + 2.4 * strong_gradient
        + 2.0 * lab_continuity
        + 6.5 * face_direction
        + 1.7 * depth_continuity
        + 0.7 * float(np.mean(g))
        - 0.0030 * shift * shift
        - 0.010 * delta_deg * delta_deg
    )
    if face_direction < 0.28:
        score -= 8.0 * (0.28 - face_direction)
    return {
        "side": side,
        "point": point,
        "tangent": tangent,
        "normal": normal,
        "shift": float(shift),
        "delta_deg": float(delta_deg),
        "score": score,
        "gradient_continuity": gradient_continuity,
        "strong_gradient": strong_gradient,
        "lab_continuity": lab_continuity,
        "face_direction": face_direction,
        "depth_continuity": depth_continuity,
    }


chosen = {}
all_candidates = {}


def search_side(side):
    # Each carton may be independently rotated. Neighbour angles establish a
    # broad coordinate frame only; the actual line evidence may rotate a side
    # up to 14 degrees away from their mean. Search coarsely first, then refine
    # the strongest neighbourhoods at one-pixel/one-degree resolution. This
    # preserves the final grid precision while avoiding a full dense sweep.
    coarse = [
        line_candidate(side, shift, angle)
        for shift in np.arange(-18, 18.1, 3.0)
        for angle in np.arange(-14, 14.1, 3.0)
    ]
    coarse.sort(key=lambda item: item["score"], reverse=True)
    refine_points = set()
    for item in coarse[:24]:
        for shift in np.arange(item["shift"] - 2.0, item["shift"] + 2.1, 1.0):
            for angle in np.arange(item["delta_deg"] - 2.0, item["delta_deg"] + 2.1, 1.0):
                if -18.0 <= shift <= 18.0 and -14.0 <= angle <= 14.0:
                    refine_points.add((float(shift), float(angle)))
    candidates = coarse + [
        line_candidate(side, shift, angle)
        for shift, angle in sorted(refine_points)
    ]
    candidates.sort(key=lambda item: item["score"], reverse=True)
    # Keep a broad pool. The opened flap can produce many near-identical,
    # high-scoring top-line hypotheses; the true body seam may rank lower on
    # its own and is selected only after opposite-edge size validation.
    return candidates[:120]


for side in ("left", "right", "top", "bottom"):
    all_candidates[side] = search_side(side)


def choose_opposite(first_name, second_name, expected_separation, reference_axis):
    best_pair = None
    for first in all_candidates[first_name]:
        for second in all_candidates[second_name]:
            separation = abs(float(np.dot(second["point"] - first["point"], reference_axis)))
            parallel = abs(first["delta_deg"] - second["delta_deg"])
            # The four cartons are the same physical product. Perspective can
            # move and rotate them, but cannot make one body 12% taller than
            # its three neighbours. Reject flap-to-bottom pairs outright.
            if parallel > 7.0 or not (0.96 * expected_separation <= separation <= 1.04 * expected_separation):
                continue
            relative_error = (separation - expected_separation) / max(expected_separation, 1.0)
            pair_score = (
                first["score"]
                + second["score"]
                - 180.0 * relative_error * relative_error
                - 0.035 * parallel * parallel
            )
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, first, second, separation, parallel)
    if best_pair is None:
        raise RuntimeError(f"No compatible {first_name}/{second_name} pair")
    return best_pair


horizontal_pair = choose_opposite("left", "right", expected_short, u)
vertical_pair = choose_opposite("top", "bottom", expected_long, v)
chosen["left"], chosen["right"] = horizontal_pair[1], horizontal_pair[2]
chosen["top"], chosen["bottom"] = vertical_pair[1], vertical_pair[2]


def intersection(first, second):
    p, r = first["point"], first["tangent"]
    q, s = second["point"], second["tangent"]
    cross = r[0] * s[1] - r[1] * s[0]
    if abs(cross) < 1e-6:
        raise RuntimeError("Parallel adjacent boundary lines")
    qp = q - p
    t = (qp[0] * s[1] - qp[1] * s[0]) / cross
    return p + t * r


quad = np.asarray(
    [
        intersection(chosen["top"], chosen["left"]),
        intersection(chosen["top"], chosen["right"]),
        intersection(chosen["bottom"], chosen["right"]),
        intersection(chosen["bottom"], chosen["left"]),
    ],
    np.float32,
)
area = abs(cv2.contourArea(quad))
widths = [float(np.linalg.norm(quad[1] - quad[0])), float(np.linalg.norm(quad[2] - quad[3]))]
heights = [float(np.linalg.norm(quad[3] - quad[0])), float(np.linalg.norm(quad[2] - quad[1]))]

# Presence evidence is separate from edge fitting. An empty green/tape region
# can contain four strong lines, but it cannot contain a large coherent patch
# classified as the learned pink carton face. Reflection may create holes, so
# use an eroded interior and tolerate a minority of low-colour pixels.
quad_mask = np.zeros((h, w), np.uint8)
cv2.fillConvexPoly(quad_mask, np.rint(quad).astype(np.int32), 1)
quad_inner = cv2.erode(quad_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
interior_values = face_probability[quad_inner]
interior_face_mean = float(np.mean(interior_values)) if len(interior_values) else 0.0
interior_face_fraction = float(np.mean(interior_values >= 0.55)) if len(interior_values) else 0.0
mean_depth_boundary = float(np.mean([item["depth_continuity"] for item in chosen.values()]))
strong_rgb_glare_boundary = bool(
    interior_face_mean >= 0.33
    and interior_face_fraction >= 0.15
    and min(item["gradient_continuity"] for item in chosen.values()) >= 0.55
    and min(item["lab_continuity"] for item in chosen.values()) >= 0.70
)
passed = bool(
    cv2.isContourConvex(quad)
    and 0.78 * expected_short <= np.mean(widths) <= 1.22 * expected_short
    and 0.88 * expected_long <= np.mean(heights) <= 1.12 * expected_long
    and all(
        (
            item["gradient_continuity"] >= 0.30
            and max(item["face_direction"], item["lab_continuity"]) >= 0.22
        )
        or (
            item["face_direction"] >= 0.65
            and item["lab_continuity"] >= 0.65
            and item["depth_continuity"] >= 0.50
        )
        or (
            item["depth_continuity"] >= 0.80
            and max(
                item["gradient_continuity"],
                item["lab_continuity"],
                item["face_direction"],
            ) >= 0.25
        )
        or (
            item["face_direction"] >= 0.85
            and item["lab_continuity"] >= 0.85
            and max(item["gradient_continuity"], item["depth_continuity"]) >= 0.15
        )
        for item in chosen.values()
    )
    and (interior_face_mean >= 0.40 or strong_rgb_glare_boundary)
    # Strong glare can cover most of the printed pink surface.  In that case
    # accept a slightly smaller learned-colour fraction only when all four
    # independently fitted sides are strongly supported by RGB/depth and the
    # mean interior colour probability still identifies a carton body.
    and (
        interior_face_fraction >= 0.18
        or (
            interior_face_fraction >= 0.15
            and interior_face_mean >= 0.43
            and mean_depth_boundary >= 0.60
            and min(
                max(item["gradient_continuity"], item["depth_continuity"])
                for item in chosen.values()
            ) >= 0.40
            and min(item["face_direction"] for item in chosen.values()) >= 0.90
        )
    )
    and (mean_depth_boundary >= 0.30 or strong_rgb_glare_boundary)
)

# The recovered body must occupy the hypothesized sequence position and must
# not overlap any already reliable box. This prevents a false topology from
# relabelling one of the three anchors as the missing carton.
known_overlaps = []
for anchor in anchors:
    intersection, _ = cv2.intersectConvexConvex(quad.astype(np.float32), anchor.astype(np.float32))
    union = abs(cv2.contourArea(quad)) + abs(cv2.contourArea(anchor)) - intersection
    known_overlaps.append(float(intersection / max(union, 1.0)))
prior_shift = float(np.linalg.norm(quad.mean(axis=0) - prior_center))
passed = bool(passed and max(known_overlaps, default=0.0) <= 0.12 and prior_shift <= 22.0)
hypothesis_score = float(
    sum(item["score"] for item in chosen.values())
    - 0.12 * assignment_residual
    - 0.06 * prior_shift * prior_shift
    - 25.0 * max(known_overlaps, default=0.0)
)

overlay = color.copy()
for side, item in chosen.items():
    p0 = item["point"] - item["tangent"] * 100
    p1 = item["point"] + item["tangent"] * 100
    cv2.line(overlay, tuple(np.rint(p0).astype(int)), tuple(np.rint(p1).astype(int)), (0, 200, 255), 1, cv2.LINE_AA)
cv2.polylines(overlay, [np.rint(quad).astype(np.int32)], True, (0, 255, 0) if passed else (0, 0, 255), 3, cv2.LINE_AA)
cv2.circle(overlay, tuple(np.rint(quad.mean(0)).astype(int)), 5, (0, 0, 255), -1, cv2.LINE_AA)
for polygon in anchors:
    cv2.polylines(overlay, [np.rint(polygon).astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / "overlay.png"), overlay)
cv2.imwrite(str(OUT / "crop.png"), overlay[max(0, ry0 - 15):min(h, ry1 + 15), max(0, rx0 - 15):min(w, rx1 + 15)])
result = {
    "frame_id": meta["frame_id"],
    "passed": passed,
    "anchor_positions": list(positions),
    "missing_position": missing_position,
    "assignment_residual": assignment_residual,
    "hypothesis_score": hypothesis_score,
    "prior_shift_px": prior_shift,
    "known_box_overlaps": known_overlaps,
    "prior_center_px": prior_center.tolist(),
    "polygon_px": quad.tolist(),
    "center_px": quad.mean(0).tolist(),
    "expected_size_px": [expected_short, expected_long],
    "measured_widths_px": widths,
    "measured_heights_px": heights,
    "area_px": area,
    "presence": {
        "interior_face_mean": interior_face_mean,
        "interior_face_fraction": interior_face_fraction,
        "mean_depth_boundary": mean_depth_boundary,
        "strong_rgb_glare_boundary": strong_rgb_glare_boundary,
    },
    "opposite_pairs": {
        "left_right": {"separation_px": horizontal_pair[3], "parallel_error_deg": horizontal_pair[4]},
        "top_bottom": {"separation_px": vertical_pair[3], "parallel_error_deg": vertical_pair[4]},
    },
    "edges": {
        side: {key: value for key, value in item.items() if key not in ("point", "tangent", "normal")}
        for side, item in chosen.items()
    },
}
(OUT / "report.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

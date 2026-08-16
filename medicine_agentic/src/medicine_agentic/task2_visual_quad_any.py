import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


sift_path = Path(os.environ.get("TASK2_SIFT_REPORT", "/tmp/task2_multi_sift/report.json"))
capture = Path(os.environ.get("TASK2_CAPTURE", "/tmp/task2_boundary_capture"))
sift = json.loads(sift_path.read_text())
anchors = sorted(
    [np.asarray(item["polygon_px"], np.float32) for item in sift.get("instances", [])],
    key=lambda polygon: polygon.mean(axis=0)[0],
)
anchor_count = len(anchors)
target_total = int(os.environ.get("TASK2_TARGET_TOTAL", "4"))
if target_total not in (3, 4) or anchor_count != target_total - 1:
    raise RuntimeError("Recovery requires exactly one missing carton for a three- or four-carton model")

color = cv2.imread(str(capture / "color.png"), cv2.IMREAD_COLOR)
if color is None:
    raise RuntimeError("Missing captured colour frame")
h, w = color.shape[:2]
lab = cv2.cvtColor(color, cv2.COLOR_BGR2LAB).astype(np.float32)


def rect_geometry(polygon):
    (_, _), (a, b), angle = cv2.minAreaRect(polygon)
    if a > b:
        a, b = b, a
        angle += 90.0
    theta = math.radians(angle)
    u = np.asarray([math.cos(theta), math.sin(theta)], np.float32)
    v = np.asarray([-math.sin(theta), math.cos(theta)], np.float32)
    if abs(u[0]) < abs(v[0]):
        u, v = v, u
    if u[0] < 0:
        u = -u
    if v[1] < 0:
        v = -v
    return float(a), float(b), u, v


geometry = [rect_geometry(polygon) for polygon in anchors]
expected_short = float(np.median([item[0] for item in geometry]))
expected_long = float(np.median([item[1] for item in geometry]))
positive_mask = np.zeros((h, w), np.uint8)
flap_mask = np.zeros((h, w), np.uint8)
outside_mask = np.zeros((h, w), np.uint8)
known_mask = np.zeros((h, w), np.uint8)

for polygon, (short, long, u, v) in zip(anchors, geometry):
    center = polygon.mean(axis=0)

    def local_quad(x0, x1, y0, y1):
        return np.asarray(
            [
                center + u * (x0 * short) + v * (y0 * long),
                center + u * (x1 * short) + v * (y0 * long),
                center + u * (x1 * short) + v * (y1 * long),
                center + u * (x0 * short) + v * (y1 * long),
            ],
            np.float32,
        )

    cv2.fillConvexPoly(positive_mask, np.rint(local_quad(-0.38, 0.38, -0.28, 0.40)).astype(np.int32), 1)
    cv2.fillConvexPoly(flap_mask, np.rint(local_quad(-0.43, 0.43, -0.69, -0.53)).astype(np.int32), 1)
    pmask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(pmask, np.rint(polygon).astype(np.int32), 1)
    known_mask |= pmask
    outside_mask |= cv2.dilate(pmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))) - pmask

flap_mask[positive_mask > 0] = 0
outside_mask[(positive_mask > 0) | (flap_mask > 0)] = 0
rng = np.random.default_rng(8841)


def clusters(mask, count):
    values = lab[mask > 0].reshape(-1, 3)
    if len(values) > 7000:
        values = values[rng.choice(len(values), 7000, False)]
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 45, 0.3)
    _, _, centers = cv2.kmeans(values.astype(np.float32), min(count, len(values)), None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    return centers


positive = clusters(positive_mask, 7)
flap = clusters(flap_mask, 5)
outside = clusters(outside_mask, 7)
pd = np.min(np.sum((lab[:, :, None, :] - positive[None, None, :, :]) ** 2, axis=3), axis=2)
fd = np.min(np.sum((lab[:, :, None, :] - flap[None, None, :, :]) ** 2, axis=3), axis=2)
od = np.min(np.sum((lab[:, :, None, :] - outside[None, None, :, :]) ** 2, axis=3), axis=2)
probability = 1.0 / (1.0 + np.exp(-np.clip((np.minimum(fd, od) - pd) / 260.0, -10.0, 10.0)))

# Search the whole usable task area. Known cartons are removed; no regular
# spacing or fixed slot is used. A box-filter heat map proposes coherent pink
# body interiors, including ones with reflection holes.
residual = probability.copy()
known_exclusion = cv2.dilate(known_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
residual[known_exclusion] = 0.0
kernel_w = max(15, int(round(0.62 * expected_short)) | 1)
kernel_h = max(21, int(round(0.60 * expected_long)) | 1)
heat = cv2.boxFilter(residual, cv2.CV_32F, (kernel_w, kernel_h), normalize=True)
allowed = np.zeros((h, w), np.uint8)
row_center = float(np.median([polygon.mean(axis=0)[1] for polygon in anchors]))
x0, x1 = int(0.41 * w), int(0.72 * w)
y0 = max(0, int(row_center - 0.42 * expected_long))
y1 = min(h, int(row_center + 0.42 * expected_long))
allowed[y0:y1 + 1, x0:x1 + 1] = 1
heat[allowed == 0] = -1.0
for polygon in anchors:
    center = tuple(np.rint(polygon.mean(axis=0)).astype(int))
    cv2.circle(heat, center, int(0.72 * expected_short), -1.0, -1)

maximum = cv2.dilate(heat, np.ones((17, 17), np.uint8))
peak_y, peak_x = np.nonzero((heat == maximum) & (heat > 0.05))
ranked = sorted([(float(heat[y, x]), float(x), float(y)) for y, x in zip(peak_y, peak_x)], reverse=True)
prior_centers = []
try:
    configured_priors = json.loads(os.environ.get("TASK2_RECOVERY_PRIORS", "[]"))
except json.JSONDecodeError:
    configured_priors = []
for value in configured_priors:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        continue
    x, y = float(value[0]), float(value[1])
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        continue
    point = np.asarray([x, y], np.float32)
    if any(
        np.linalg.norm(point - polygon.mean(axis=0)) < 0.48 * expected_short
        for polygon in anchors
    ):
        continue
    prior_centers.append((float(max(heat[int(round(y)), int(round(x))], 0.0)), x, y))

priors = list(prior_centers)
for heat_score, x, y in ranked:
    # The face heat peak may sit on the printed lower half or on an unreflected
    # upper patch rather than at the geometric centre. Cartons may be slightly
    # staggered, but remain in the same work row, so retain the heat-derived x
    # and limit only y to a band around the confirmed cartons. The subsequent
    # four-line fit still determines the final top/bottom edges independently.
    # Heat provides x/presence only. Printed graphics and reflection can move
    # its y maximum by tens of pixels, so start every candidate on the robust
    # row centre. The four-line fitter may still translate up/down by 18 px and
    # determines the true top/bottom edges from evidence.
    y = row_center
    point = np.asarray([x, y], np.float32)
    if any(np.linalg.norm(point - np.asarray(previous[1:], np.float32)) < 0.48 * expected_short for previous in priors):
        continue
    priors.append((heat_score, x, y))
    # More peaks are usually duplicate fragments of the same reflected face.
    # Four separated x candidates cover the usable work area while bounding
    # worst-case latency of the expensive four-edge fit.
    if len(priors) >= 3:
        break

root = Path(os.environ.get("TASK2_VISUAL_OUT", "/tmp/task2_visual_any"))
root.mkdir(parents=True, exist_ok=True)
fit_script = Path(
    os.environ.get(
        "TASK2_VISUAL_FIT_SCRIPT",
        Path(__file__).with_name("task2_visual_quad_fit.py"),
    )
)


def fit_prior(index_and_prior):
    index, (heat_score, x, y) = index_and_prior
    candidate_dir = root / f"candidate_{index + 1}"
    environment = os.environ.copy()
    environment["TASK2_PRIOR_CENTER"] = f"{x},{y}"
    environment["TASK2_TARGET_TOTAL"] = str(target_total)
    environment["TASK2_VISUAL_OUT"] = str(candidate_dir)
    process = subprocess.run(
        [sys.executable, str(fit_script)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=35,
        env=environment,
    )
    if process.returncode != 0 or not (candidate_dir / "report.json").exists():
        return {"prior_center_px": [x, y], "heat_score": heat_score, "passed": False}
    result = json.loads((candidate_dir / "report.json").read_text())
    result["heat_score"] = heat_score
    result["output_dir"] = str(candidate_dir)
    # Coherent face heat is presence evidence, but four fitted edges dominate.
    result["hypothesis_score"] = float(result["hypothesis_score"] + 5.0 * heat_score)
    return result


# Each fit reads the same immutable frame and writes to its own directory.
# Bound parallelism to three processes to reduce worst-case latency without
# competing heavily with the live console for CPU and memory.
priority_count = len(prior_centers)
priority_hypotheses = [
    fit_prior((index, prior))
    for index, prior in enumerate(priors[:priority_count])
]
if any(item.get("passed") for item in priority_hypotheses):
    hypotheses = priority_hypotheses
else:
    remaining = list(enumerate(priors[priority_count:], start=priority_count))
    with ThreadPoolExecutor(max_workers=min(3, max(1, len(remaining)))) as executor:
        hypotheses = priority_hypotheses + list(executor.map(fit_prior, remaining))

passed = sorted([item for item in hypotheses if item.get("passed")], key=lambda item: item["hypothesis_score"], reverse=True)
if not passed:
    final = {
        "passed": False,
        "reason": "no unoccupied region has a complete supported carton body",
        "candidate_priors": [[item[1], item[2], item[0]] for item in priors],
        "hypotheses": hypotheses,
    }
elif len(passed) > 1 and passed[0]["hypothesis_score"] - passed[1]["hypothesis_score"] < 2.0:
    final = {
        "passed": False,
        "reason": "ambiguous carton body candidates",
        "score_margin": passed[0]["hypothesis_score"] - passed[1]["hypothesis_score"],
        "hypotheses": hypotheses,
    }
else:
    winner = passed[0]
    final = dict(winner)
    final["passed"] = True
    final["target_total"] = target_total
    final["selection"] = "whole task-area pink-body scan plus independent four-edge RGB-D validation"
    final["score_margin"] = None if len(passed) == 1 else winner["hypothesis_score"] - passed[1]["hypothesis_score"]
    final["hypotheses"] = [
        {
            "prior_center_px": item.get("prior_center_px"),
            "heat_score": item.get("heat_score"),
            "missing_position": item.get("missing_position"),
            "passed": item.get("passed", False),
            "hypothesis_score": item.get("hypothesis_score"),
            "center_px": item.get("center_px"),
        }
        for item in hypotheses
    ]
    source = Path(winner["output_dir"])
    shutil.copy2(source / "overlay.png", root / "overlay.png")
    shutil.copy2(source / "crop.png", root / "crop.png")

(root / "report.json").write_text(json.dumps(final, indent=2))
print(json.dumps(final, indent=2))

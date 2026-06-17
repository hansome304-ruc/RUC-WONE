"""E2E smoke test from inside the perception-server container.

Drives /segment then /grasp using demo/000.* and demo/000.meta.yml as the
real K source. Prints timing and grasp poses for visual sanity check.
"""
import base64
import json
import time
import urllib.request

import cv2
import numpy as np
import yaml


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        f"http://localhost:9100{path}",
        method="POST",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


rgb = cv2.cvtColor(cv2.imread("demo/000.rgb.png"), cv2.COLOR_BGR2RGB)
depth = cv2.imread("demo/000.depth.png", cv2.IMREAD_UNCHANGED).astype(np.uint16)
H, W = rgb.shape[:2]

with open("demo/000.meta.yml") as f:
    meta = yaml.safe_load(f)
K = np.array(meta["left_p"]).reshape(3, 4)[:3, :3].copy()
if "rectified_width" in meta and meta["width"] and meta["rectified_width"]:
    K[:2, :] /= meta["rectified_width"] / meta["width"]
print(f"img {W}x{H}, K[fx,fy,cx,cy]=[{K[0,0]:.1f},{K[1,1]:.1f},{K[0,2]:.1f},{K[1,2]:.1f}]")

t = time.time()
seg = post("/segment", {"rgb_b64": b64(rgb), "H": H, "W": W, "prompt": "a bottle"})
mask = np.frombuffer(base64.b64decode(seg["mask_b64"]), np.uint8).reshape(H, W)
print(
    f"/segment {time.time()-t:.2f}s bbox={seg['bbox']} score={seg['score']:.3f} "
    f"mask_px={int(mask.sum())}"
)

t = time.time()
g = post("/grasp", {
    "rgb_b64": b64(rgb), "depth_b64": b64(depth), "mask_b64": b64(mask),
    "K": K.tolist(), "H": H, "W": W, "top_k": 5,
})
print(f"/grasp {time.time()-t:.2f}s -> {len(g['grasps'])} grasps")
for i, gr in enumerate(g["grasps"]):
    T = np.array(gr["T_cam"])
    xyz, app = T[:3, 3], T[:3, 2]
    print(
        f"  [{i}] score={gr['score']:.3f} w={gr['width']*1000:.0f}mm "
        f"d={gr['depth']*1000:.0f}mm xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
        f"approach=({app[0]:+.2f},{app[1]:+.2f},{app[2]:+.2f})"
    )

t = time.time()
gd = post("/grasp_dummy", {
    "rgb_b64": b64(rgb), "depth_b64": b64(depth), "mask_b64": b64(mask),
    "K": K.tolist(), "H": H, "W": W, "top_k": 1,
})
T = np.array(gd["grasps"][0]["T_cam"])
print(
    f"/grasp_dummy {time.time()-t:.2f}s xyz=({T[0,3]:+.3f},{T[1,3]:+.3f},{T[2,3]:+.3f})"
)

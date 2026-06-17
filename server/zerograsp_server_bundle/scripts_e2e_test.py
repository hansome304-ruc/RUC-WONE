"""End-to-end smoke test: drive /segment + /grasp using demo/000.* on the server.

Intended to be exec'd inside the perception-server container:
    docker exec perception-server python3 /opt/app/scripts_e2e_test.py
"""
import base64
import json
import time
import urllib.request

import cv2
import numpy as np
import yaml

# Load real demo image
rgb_bgr = cv2.imread("demo/000.rgb.png", cv2.IMREAD_COLOR)
rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
depth = cv2.imread("demo/000.depth.png", cv2.IMREAD_UNCHANGED).astype(np.uint16)
H, W = rgb.shape[:2]

# Real intrinsics: extract from meta.yml left_p the same way fetch_data does
with open("demo/000.meta.yml") as f:
    meta = yaml.safe_load(f)
left_p = np.array(meta["left_p"]).reshape(3, 4)
K = left_p[:3, :3].copy()
if "rectified_width" in meta and meta["width"] and meta["rectified_width"]:
    scale = meta["rectified_width"] / meta["width"]
    K[:2, :] /= scale
print(f"image {W}x{H}")
print(f"K =\n{K}")
print(f"depth range (mm): {int(depth[depth>0].min())}-{int(depth.max())}")


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


# /segment
t0 = time.time()
seg = post("/segment", {"rgb_b64": b64(rgb), "H": H, "W": W, "prompt": "a bottle"})
mask = np.frombuffer(base64.b64decode(seg["mask_b64"]), np.uint8).reshape(H, W)
print(
    f"\n/segment {time.time()-t0:.2f}s bbox={seg['bbox']} "
    f"score={seg['score']:.3f} mask_px={int(mask.sum())}"
)

# /grasp using SAM mask
t0 = time.time()
g = post(
    "/grasp",
    {
        "rgb_b64": b64(rgb),
        "depth_b64": b64(depth),
        "mask_b64": b64(mask),
        "K": K.tolist(),
        "H": H, "W": W, "top_k": 5,
    },
)
print(f"\n/grasp {time.time()-t0:.2f}s -> {len(g['grasps'])} grasps")
for i, gr in enumerate(g["grasps"]):
    T = np.array(gr["T_cam"])
    xyz = T[:3, 3]
    approach = T[:3, 2]
    print(
        f"  [{i}] score={gr['score']:.3f} width={gr['width']*1000:.0f}mm "
        f"depth={gr['depth']*1000:.0f}mm"
    )
    print(
        f"       xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:+.3f}) "
        f"approach=({approach[0]:+.2f},{approach[1]:+.2f},{approach[2]:+.2f})"
    )

# /grasp_dummy for comparison
g2 = post(
    "/grasp_dummy",
    {
        "rgb_b64": b64(rgb),
        "depth_b64": b64(depth),
        "mask_b64": b64(mask),
        "K": K.tolist(),
        "H": H, "W": W, "top_k": 1,
    },
)
T = np.array(g2["grasps"][0]["T_cam"])
print(
    f"\n/grasp_dummy (centroid) xyz=({T[0,3]:+.3f},{T[1,3]:+.3f},{T[2,3]:+.3f})"
)

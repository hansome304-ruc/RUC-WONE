#!/usr/bin/env python3
import numpy as np
from agentic_grasp.transforms import R_from_quat_xyzw

def main():
    quat = [0.047, 0.734, -0.077, 0.672]
    R = R_from_quat_xyzw(quat)
    print("Rotation matrix R:")
    print(R)
    print("\nEnd-effector axes in base frame:")
    print(f"X-axis (local x): {R[:, 0].round(3).tolist()}")
    print(f"Y-axis (local y): {R[:, 1].round(3).tolist()}")
    print(f"Z-axis (local z, approach): {R[:, 2].round(3).tolist()}")

if __name__ == "__main__":
    main()

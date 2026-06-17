#!/usr/bin/env python3
import logging
from airbot_py.arm import AIRBOTPlay

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    for name, port in [("left", 50051), ("right", 50053)]:
        arm = AIRBOTPlay(url="localhost", port=port)
        if arm.connect():
            pose = arm.get_end_pose()
            print(f"{name} arm current pose: {pose}")
            arm.disconnect()
        else:
            print(f"Failed to connect to {name} arm")

if __name__ == "__main__":
    main()

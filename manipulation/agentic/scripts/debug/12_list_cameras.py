#!/usr/bin/env python3
import pyrealsense2 as rs

def main():
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"Found {len(devices)} connected RealSense devices:")
    for i, d in enumerate(devices):
        name = d.get_info(rs.camera_info.name)
        serial = d.get_info(rs.camera_info.serial_number)
        print(f"Device {i+1}: {name} (Serial: {serial})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Real-deal scripted demo using ZeroGrasp.

This is just `03_dummy_grasp.py` with the endpoint pre-set to /grasp.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    if "--endpoint" not in sys.argv:
        sys.argv += ["--endpoint", "/grasp"]
    runpy.run_path(str(Path(__file__).with_name("03_dummy_grasp.py")), run_name="__main__")

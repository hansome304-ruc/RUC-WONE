"""Run one glare-obscured Task2 carton recovery on the LAN CPU worker."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    work = Path(sys.argv[1])
    python = sys.argv[2]
    settings = json.loads((work / "settings.json").read_text(encoding="utf-8"))
    package = Path(__file__).resolve().parent
    environment = os.environ.copy()
    environment.update(
        {
            "TASK2_CAPTURE": str(work / "capture"),
            "TASK2_SIFT_REPORT": str(work / "sift.json"),
            "TASK2_TARGET_TOTAL": str(settings["target_total"]),
            "TASK2_VISUAL_OUT": str(work / "output"),
            "TASK2_VISUAL_FIT_SCRIPT": str(package / "task2_visual_quad_fit.py"),
            "TASK2_RECOVERY_PRIORS": json.dumps(settings.get("recovery_priors", [])),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        [python, str(package / "task2_visual_quad_any.py")],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=12,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

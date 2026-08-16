from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medicine_agentic.p0_doctor import run_doctor
from medicine_agentic.pose_store import new_pose_document


class P0DoctorTests(unittest.TestCase):
    def test_missing_physical_inputs_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "configs" / "task1_box.json").write_text(
                json.dumps(
                    {
                        "camera": {
                            "cam_to_left_path": str(root / "missing.json"),
                            "web_console_url": "http://127.0.0.1:1",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "configs" / "p0_poses.json").write_text(
                json.dumps(new_pose_document()),
                encoding="utf-8",
            )
            with patch("medicine_agentic.p0_doctor.shutil.which", return_value=None):
                report = run_doctor(project_root=root)
            self.assertFalse(report["ok"])
            self.assertFalse(report["safe_to_execute_motion"])
            self.assertIn("camera_to_left_base_file", report["blocking_checks"])
            self.assertIn("paired_named_poses", report["blocking_checks"])
            self.assertIn("left_suction_tcp", report["blocking_checks"])


if __name__ == "__main__":
    unittest.main()

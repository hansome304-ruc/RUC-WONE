from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from medicine_agentic.teleop_launcher import (
    TeleopLaunchConflict,
    TeleopLauncher,
)


class TeleopLauncherUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "arm_services.json").write_text(
            json.dumps({"remote_host": "10.20.30.40"}),
            encoding="utf-8",
        )
        self.desired_marker = self.root / "follow-desired"
        self.lead_env = self.root / "teleop.env"
        self.launcher = TeleopLauncher(
            {
                "enabled": True,
                "repo_root": ".",
                "arm_runtime_config": "arm_services.json",
                "follow_desired_marker": str(self.desired_marker),
                "follow_lead_env": str(self.lead_env),
            },
            config_dir=self.root,
        )
        self.ready_endpoints = {
            "lead_host": "10.20.30.40",
            "lead_ports": {"50050": True, "50052": True},
            "follower_ports": {"50051": True, "50053": True},
            "lead_ready": True,
            "follower_ready": True,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def stopped_check() -> dict:
        return {
            "ok": False,
            "tmux": False,
            "state": "missing",
            "heartbeat_age_s": None,
            "error": "",
        }

    def confirmed_payload(self) -> dict:
        return {
            "confirm": "START_FOLLOW",
            "area_clear": True,
            "estop_ready": True,
            "initial_pose_aligned": True,
        }

    def hard_restart_payload(self) -> dict:
        return {
            "confirm": "HARD_RESTART_TELEOP",
            "area_clear": True,
            "estop_ready": True,
            "master_arms_stable": True,
        }

    def test_unknown_or_missing_confirmation_fields_are_rejected(self) -> None:
        for payload in (
            {},
            {"confirm": "START_FOLLOW"},
            {**self.confirmed_payload(), "lead_url": "malicious.example"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.launcher.start(payload)

        for payload in (
            {},
            {"confirm": "HARD_RESTART_TELEOP"},
            {**self.hard_restart_payload(), "lead_url": "malicious.example"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    self.launcher.hard_restart(payload)

    def test_missing_endpoint_never_runs_a_script(self) -> None:
        missing = {
            **self.ready_endpoints,
            "lead_ports": {"50050": True, "50052": False},
            "lead_ready": False,
        }
        with mock.patch.object(
            self.launcher,
            "_check_follow",
            return_value=self.stopped_check(),
        ), mock.patch.object(
            self.launcher,
            "_endpoints",
            return_value=missing,
        ), mock.patch.object(
            self.launcher,
            "_stable_endpoints",
            return_value=missing,
        ), mock.patch.object(
            self.launcher,
            "_run_script",
        ) as run_script:
            with self.assertRaises(TeleopLaunchConflict):
                self.launcher.start(self.confirmed_payload())
        run_script.assert_not_called()

    def test_systemd_follow_marker_matches_8888_lead_configuration(self) -> None:
        self.desired_marker.touch()
        self.lead_env.write_text("LEAD_URL=10.20.30.40\n", encoding="utf-8")
        self.assertTrue(self.launcher._desired_enabled())
        self.assertTrue(self.launcher._desired_matches("10.20.30.40"))
        self.assertFalse(self.launcher._desired_matches("10.20.30.41"))

    def test_local_follow_matches_runtime_without_root_marker(self) -> None:
        self.assertTrue(self.launcher._desired_matches("127.0.0.1"))
        self.assertTrue(self.launcher._desired_matches("localhost"))

    def test_start_is_async_and_uses_only_the_fixed_follow_script(self) -> None:
        follow_ready = False
        calls: list[tuple[str, dict[str, str]]] = []

        def check() -> dict:
            return {
                "ok": follow_ready,
                "tmux": follow_ready,
                "state": "running" if follow_ready else "missing",
                "heartbeat_age_s": 0.1 if follow_ready else None,
                "error": "",
            }

        def run_script(
            script: Path,
            env: dict[str, str],
            timeout_s: float,
        ) -> None:
            nonlocal follow_ready
            self.assertGreater(timeout_s, 0)
            calls.append((script.name, dict(env)))
            follow_ready = True

        with mock.patch.object(
            self.launcher,
            "_check_follow",
            side_effect=check,
        ), mock.patch.object(
            self.launcher,
            "_endpoints",
            return_value=self.ready_endpoints,
        ), mock.patch.object(
            self.launcher,
            "_stable_endpoints",
            return_value=self.ready_endpoints,
        ), mock.patch.object(
            self.launcher,
            "_run_script",
            side_effect=run_script,
        ), mock.patch.object(
            self.launcher,
            "_desired_matches",
            return_value=True,
        ):
            response = self.launcher.start(self.confirmed_payload())
            self.assertTrue(response["ok"])
            self.assertTrue(response["operation_id"])
            deadline = time.monotonic() + 2.0
            while self.launcher.status()["busy"] and time.monotonic() < deadline:
                time.sleep(0.01)
            status = self.launcher.status()

        self.assertTrue(status["running"])
        self.assertEqual(
            [name for name, _ in calls],
            ["start_teleop_follow.sh"],
        )
        env = calls[0][1]
        self.assertEqual(env["LEAD_URL"], "10.20.30.40")
        self.assertEqual(env["FOLLOW_URL"], "localhost")
        self.assertEqual(env["LEFT_LEAD_PORT"], "50050")
        self.assertEqual(env["RIGHT_LEAD_PORT"], "50052")
        self.assertEqual(env["LEFT_PORT"], "50051")
        self.assertEqual(env["RIGHT_PORT"], "50053")

    def test_hard_restart_uses_only_fixed_script_and_saved_lead_host(self) -> None:
        follow_ready = False
        calls: list[tuple[str, dict[str, str], float]] = []

        def check() -> dict:
            return {
                "ok": follow_ready,
                "tmux": follow_ready,
                "state": "running" if follow_ready else "missing",
                "heartbeat_age_s": 0.1 if follow_ready else None,
                "error": "",
            }

        def run_script(
            script: Path,
            env: dict[str, str],
            timeout_s: float,
        ) -> None:
            nonlocal follow_ready
            calls.append((script.name, dict(env), timeout_s))
            if script.name == "hard_restart_teleop_stack.sh":
                follow_ready = True

        with mock.patch.object(
            self.launcher,
            "_check_follow",
            side_effect=check,
        ), mock.patch.object(
            self.launcher,
            "_endpoints",
            return_value={
                **self.ready_endpoints,
                "lead_ready": False,
                "follower_ready": False,
            },
        ), mock.patch.object(
            self.launcher,
            "_stable_endpoints",
            return_value=self.ready_endpoints,
        ), mock.patch.object(
            self.launcher,
            "_run_script",
            side_effect=run_script,
        ), mock.patch.object(
            self.launcher,
            "_desired_matches",
            return_value=True,
        ):
            response = self.launcher.hard_restart(
                self.hard_restart_payload()
            )
            self.assertTrue(response["ok"])
            deadline = time.monotonic() + 2.0
            while self.launcher.status()["busy"] and time.monotonic() < deadline:
                time.sleep(0.01)
            status = self.launcher.status()

        self.assertTrue(status["running"])
        self.assertEqual(status["state"], "running")
        self.assertEqual(len(calls), 1)
        script_name, env, timeout_s = calls[0]
        self.assertEqual(script_name, "hard_restart_teleop_stack.sh")
        self.assertEqual(env["LEAD_URL"], "10.20.30.40")
        self.assertGreaterEqual(timeout_s, 60.0)


if __name__ == "__main__":
    unittest.main()

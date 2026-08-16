from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from medicine_agentic.act_inference import (
    ActInferenceClient,
    ActInferenceProtocolError,
    build_prediction_payload,
    validate_prediction_response,
)


MODEL_SHA = "06be807b34541a9af6e8ea6d5c13d824f11bdac5232112e619ce4b7f96f91116"
MODEL_SHA_2 = "16be807b34541a9af6e8ea6d5c13d824f11bdac5232112e619ce4b7f96f91117"


def _frames() -> dict[str, np.ndarray]:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    return {
        "cam_high": frame.copy(),
        "cam_left_wrist": frame.copy(),
        "cam_right_wrist": frame.copy(),
    }


class ActInferenceTests(unittest.TestCase):
    def test_build_prediction_payload_has_pinned_dimensions(self) -> None:
        payload = build_prediction_payload(
            state=[0.0] * 14,
            frames_bgr=_frames(),
            horizon=25,
            request_id="request-1",
            session_id="session-1",
        )
        self.assertEqual(payload["horizon"], 25)
        self.assertEqual(len(payload["observation"]["state"]), 14)
        self.assertEqual(
            set(payload["observation"]["images"]),
            {"cam_high", "cam_left_wrist", "cam_right_wrist"},
        )

    def test_build_prediction_payload_rejects_bad_state(self) -> None:
        with self.assertRaises(ValueError):
            build_prediction_payload(
                state=[0.0] * 13,
                frames_bgr=_frames(),
                horizon=25,
                request_id="request-1",
                session_id="session-1",
            )

    def test_client_rejects_world_readable_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "group or other"):
                ActInferenceClient(
                    {
                        "enabled": True,
                        "base_url": "http://10.42.57.108:9120",
                        "token_file": str(token),
                        "expected_model_sha256": MODEL_SHA,
                    },
                    config_dir=tmp_path,
                )

    def test_client_validates_model_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o600)
            client = ActInferenceClient(
                {
                    "enabled": True,
                    "base_url": "http://10.42.57.108:9120",
                    "token_file": str(token),
                    "expected_model_sha256": MODEL_SHA,
                },
                config_dir=tmp_path,
            )
            with self.assertRaisesRegex(ActInferenceProtocolError, "SHA256"):
                client._validate_identity(
                    {
                        "ok": True,
                        "service": "medicine_act_inference_v1",
                        "model_sha256": "0" * 64,
                    }
                )

    def test_client_accepts_zr0_service_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o600)
            client = ActInferenceClient(
                {
                    "enabled": True,
                    "base_url": "http://10.42.112.6:9120",
                    "token_file": str(token),
                    "expected_model_sha256": MODEL_SHA,
                },
                config_dir=tmp_path,
            )

            client._validate_identity(
                {
                    "ok": True,
                    "service": "medicine_zr0_inference_v1",
                    "model_sha256": MODEL_SHA,
                }
            )

    def test_client_switches_between_pinned_absolute_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o600)
            client = ActInferenceClient(
                {
                    "enabled": True,
                    "active_profile": "act1",
                    "profiles": {
                        "act1": {
                            "base_url": "http://10.0.0.1:8899",
                            "expected_model_sha256": MODEL_SHA,
                            "expected_action_representation": "absolute_joint_target",
                        },
                        "act2": {
                            "base_url": "http://10.0.0.1:8900",
                            "expected_model_sha256": MODEL_SHA_2,
                            "expected_action_representation": "absolute_joint_target",
                        },
                    },
                    "token_file": str(token),
                },
                config_dir=tmp_path,
            )
            healthy = {
                "ok": True,
                "service": "medicine_zr0_inference_v1",
                "model_sha256": MODEL_SHA_2,
                "action_representation": "absolute_joint_target",
            }
            with mock.patch.object(client, "_request", return_value=healthy):
                status = client.select_profile("act2")

            self.assertTrue(status["ready"])
            self.assertEqual(status["profile_id"], "act2")
            self.assertEqual(client.base_url, "http://10.0.0.1:8900")

    def test_client_rejects_profile_with_wrong_action_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o600)
            client = ActInferenceClient(
                {
                    "enabled": True,
                    "active_profile": "act1",
                    "profiles": {
                        "act1": {
                            "base_url": "http://10.0.0.1:8899",
                            "expected_model_sha256": MODEL_SHA,
                        },
                        "act2": {
                            "base_url": "http://10.0.0.1:8900",
                            "expected_model_sha256": MODEL_SHA_2,
                            "expected_action_representation": "absolute_joint_target",
                        },
                    },
                    "token_file": str(token),
                },
                config_dir=tmp_path,
            )
            wrong = {
                "ok": True,
                "service": "medicine_zr0_inference_v1",
                "model_sha256": MODEL_SHA_2,
                "action_representation": "delta_target_minus_observation_state",
            }
            with mock.patch.object(client, "_request", return_value=wrong):
                with self.assertRaisesRegex(RuntimeError, "action representation"):
                    client.select_profile("act2")

            self.assertEqual(client.active_profile, "act1")

    def test_client_overrides_left_gripper_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o600)
            override = 0.0001414934045968431
            client = ActInferenceClient(
                {
                    "enabled": True,
                    "base_url": "http://10.42.57.108:9120",
                    "token_file": str(token),
                    "expected_model_sha256": MODEL_SHA,
                    "left_gripper_observation_override": override,
                },
                config_dir=tmp_path,
            )
            response = {
                "ok": True,
                "service": "medicine_act_inference_v1",
                "model_sha256": MODEL_SHA,
                "actions": [[0.0] * 14],
                "termination_supported": False,
            }
            raw_state = [0.0] * 14
            raw_state[6] = -0.00012833373808149307

            with mock.patch.object(
                client, "_request", return_value=response
            ) as request:
                result = client.predict(
                    state=raw_state,
                    frames_bgr=_frames(),
                    horizon=1,
                    request_id="request-override",
                    session_id="session-override",
                )

            request_payload = request.call_args.kwargs["payload"]
            self.assertAlmostEqual(
                request_payload["observation"]["state"][6], override
            )
            self.assertAlmostEqual(
                result["observation_overrides"]["left_gripper_raw"], override
            )

    def test_json_response_rejects_non_object(self) -> None:
        response = SimpleNamespace(read=lambda _size: json.dumps([1, 2, 3]).encode())
        with self.assertRaisesRegex(ActInferenceProtocolError, "object"):
            ActInferenceClient._json_response(response)

    def test_cached_status_never_probes_the_model_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            tmp_path = Path(temporary)
            token = tmp_path / "token"
            token.write_text("x" * 64, encoding="utf-8")
            token.chmod(0o600)
            client = ActInferenceClient(
                {
                    "enabled": True,
                    "base_url": "http://10.42.57.108:9120",
                    "token_file": str(token),
                    "expected_model_sha256": MODEL_SHA,
                },
                config_dir=tmp_path,
            )

            with mock.patch.object(client, "_request") as request:
                unchecked = client.cached_status()

            request.assert_not_called()
            self.assertIsNone(unchecked["ready"])
            self.assertFalse(unchecked["cached"])

            healthy_payload = {
                "ok": True,
                "service": "medicine_act_inference_v1",
                "model_sha256": MODEL_SHA,
                "device": "cuda",
            }
            with mock.patch.object(
                client,
                "_request",
                return_value=healthy_payload,
            ) as request:
                client.status(force=True)
                request.reset_mock()
                cached = client.cached_status()

            request.assert_not_called()
            self.assertTrue(cached["ready"])
            self.assertTrue(cached["cached"])

    def test_terminal15_response_keeps_only_14d_robot_actions(self) -> None:
        response = validate_prediction_response(
            {
                "service": "medicine_act_inference_v2_terminal15",
                "action_representation": "delta_target_minus_observation_state",
                "actions": [[0.0] * 14 for _ in range(3)],
                "termination_supported": True,
                "done_probs": [0.01, 0.5, 0.9],
            },
            selected_horizon=3,
        )
        self.assertEqual(np.asarray(response["actions"]).shape, (3, 14))
        self.assertEqual(np.asarray(response["done_probs"]).shape, (3,))
        self.assertFalse(response["execution_enabled"])

    def test_terminal15_response_rejects_bad_done_probability(self) -> None:
        with self.assertRaisesRegex(ActInferenceProtocolError, "done probabilities"):
            validate_prediction_response(
                {
                    "service": "medicine_act_inference_v2_terminal15",
                    "action_representation": "delta_target_minus_observation_state",
                    "actions": [[0.0] * 14 for _ in range(2)],
                    "termination_supported": True,
                    "done_probs": [0.1, 1.1],
                },
                selected_horizon=2,
            )

    def test_legacy_response_explicitly_reports_no_termination(self) -> None:
        response = validate_prediction_response(
            {
                "service": "medicine_act_inference_v1",
                "actions": [[0.0] * 14],
            },
            selected_horizon=1,
        )
        self.assertFalse(response["termination_supported"])
        self.assertNotIn("done_probs", response)

    def test_terminal15_response_requires_explicit_action_representation(self) -> None:
        with self.assertRaisesRegex(ActInferenceProtocolError, "action_representation"):
            validate_prediction_response(
                {
                    "service": "medicine_act_inference_v2_terminal15",
                    "actions": [[0.0] * 14],
                },
                selected_horizon=1,
            )


if __name__ == "__main__":
    unittest.main()

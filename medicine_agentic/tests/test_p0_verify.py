from __future__ import annotations

import unittest

from medicine_agentic.p0_verify import (
    assess_visual_probe_lift,
    fuse_hold_evidence,
)


class P0VerifyTests(unittest.TestCase):
    def test_nominal_twenty_millimetre_lift_passes_without_audio(self) -> None:
        visual = assess_visual_probe_lift(
            [0.30, -0.10, 0.11],
            [0.301, -0.101, 0.130],
            source_vacated=True,
        )
        self.assertTrue(visual.passed)
        decision = fuse_hold_evidence(visual, acoustic="unavailable")
        self.assertTrue(decision.passed)

    def test_sound_cannot_override_failed_visual_evidence(self) -> None:
        visual = assess_visual_probe_lift(
            [0.30, -0.10, 0.11],
            [0.30, -0.10, 0.111],
            source_vacated=False,
        )
        self.assertFalse(visual.passed)
        decision = fuse_hold_evidence(visual, acoustic="attached")
        self.assertFalse(decision.passed)

    def test_audio_can_be_required_only_after_calibration(self) -> None:
        visual = assess_visual_probe_lift(
            [0.30, -0.10, 0.11],
            [0.30, -0.10, 0.13],
            source_vacated=True,
        )
        self.assertFalse(
            fuse_hold_evidence(
                visual,
                acoustic="uncertain",
                acoustic_policy="required",
            ).passed
        )


if __name__ == "__main__":
    unittest.main()

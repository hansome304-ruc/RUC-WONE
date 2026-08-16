from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from resample_act_fixed_rate import CAMERA_NAMES, ResampleError, resample_episode


class FixedRateResampleTests(unittest.TestCase):
    def test_resamples_state_and_action_and_uses_nearest_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.hdf5"
            output = root / "output.hdf5"
            timestamps = np.asarray([10.0, 10.033333333, 10.1])
            with h5py.File(source, "w") as dst:
                observations = dst.create_group("observations")
                observations.create_dataset(
                    "qpos",
                    data=np.repeat(np.asarray([[0.0], [1.0], [3.0]]), 14, axis=1),
                )
                images = observations.create_group("images")
                for camera_index, camera_name in enumerate(CAMERA_NAMES):
                    array = np.stack(
                        [
                            np.full((4, 5, 3), value + camera_index, dtype=np.uint8)
                            for value in (10, 20, 30)
                        ]
                    )
                    images.create_dataset(camera_name, data=array)
                dst.create_dataset(
                    "action",
                    data=np.repeat(np.asarray([[0.0], [2.0], [6.0]]), 14, axis=1),
                )
                dst.create_dataset("source_aligned_index", data=[0, 1, 3])
                timing = dst.create_group("timestamps")
                timing.create_dataset("aligned", data=timestamps)
                timing.create_dataset(
                    "cameras", data=np.repeat(timestamps[:, None], 3, axis=1)
                )
            details = resample_episode(source, output, hz=30.0)
            self.assertEqual(details["sample_count"], 4)
            with h5py.File(output, "r") as result:
                np.testing.assert_allclose(np.diff(result["timestamps/aligned"][:]), 1 / 30)
                np.testing.assert_allclose(
                    result["observations/qpos"][:, 0], [0.0, 1.0, 2.0, 3.0]
                )
                np.testing.assert_allclose(result["action"][:, 0], [0.0, 2.0, 4.0, 6.0])
                np.testing.assert_array_equal(result["resample_source_index"][:], [0, 1, 2, 2])
                np.testing.assert_array_equal(
                    result["observations/images/cam_high"][:, 0, 0, 0],
                    [10, 20, 30, 30],
                )
                metadata = json.loads(result.attrs["temporal_resampling_json"])
                self.assertEqual(metadata["state_action_resampling"], "linear")

    def test_refuses_to_overwrite_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "existing.hdf5"
            output.touch()
            with self.assertRaisesRegex(ResampleError, "refusing to overwrite"):
                resample_episode(root / "missing.hdf5", output)


if __name__ == "__main__":
    unittest.main()

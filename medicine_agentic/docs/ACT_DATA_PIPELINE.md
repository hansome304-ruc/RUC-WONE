# ACT data collection on `dosw1`

The existing private console on port 8899 records strict, training-aligned ACT
demonstrations. It reads:

- front, left-wrist and right-wrist RGB cameras as one synchronized bundle;
- follower joint/gripper feedback as `observation.state`;
- leader joint/gripper feedback as expert `action`.

The default follower endpoints are `localhost:50051/50053`. The default leader
endpoints are `10.47.157.9:50050/50052`; verify these values in
`configs/packaging_console.json` before the first hardware recording.

## Offline development status

Development and automated tests do not bind port 8899 and use fake cameras and
fake arms. Nothing in the new scripts kills, starts, resets, or changes a robot
service. Start the live console only after the operator permits it:

```bash
cd ~/RUC-WONE/medicine_agentic
./scripts/start_packaging_console_8899.sh
```

Open the existing 8899 page and choose `ACT 双臂示教`. Both arms must be recorded
even when one remains stationary. Start teleoperation through the existing Follow
control, record one clean demonstration, then stop and save. Prefix labels with
`act1_`/`act01_`, `act2_`/`act02_`, or `act3_`/`act03_` so the converter creates
separate task datasets.

## Episode contract

A successful episode is written atomically under
`recordings/act/finalized/<episode-id>/`:

```text
meta.json
READY
checksums.sha256
sensors/cam_front_rgb.mp4
sensors/cam_front_rgb.mp4.tsf
sensors/cam_front_frames.jsonl
sensors/cam_left_wrist_rgb.mp4
sensors/cam_left_wrist_rgb.mp4.tsf
sensors/cam_left_wrist_frames.jsonl
sensors/cam_right_wrist_rgb.mp4
sensors/cam_right_wrist_rgb.mp4.tsf
sensors/cam_right_wrist_frames.jsonl
observations/{left,right}_arm.jsonl
actions/{left,right}_arm.jsonl
aligned/samples.jsonl
```

`READY` is created only after every stream is closed and the checksum manifest is
written. Failed recordings go to `recordings/act/failed/` and never get a READY
marker. The training pipeline must ignore any directory without READY.

`aligned/samples.jsonl` is the only training-authoritative timeline. Every row
contains one synchronized three-camera bundle, interpolated follower observation,
interpolated leader action, and the exact `camera_frame_index` to decode. Never
align by JSONL/video array position: edge frames may be trimmed by the strict
quality gate.

All image and arm alignment uses the cameras' RealSense global-device exposure
timestamps. USB arrival time is retained only as `arrival_captured_at` for
diagnostics and must never be used for training alignment. Frame metadata records
`sync_timestamp_ms`, `timestamp_domain`, and `timestamp_source`; the validator and
HDF5 exporter reject an episode if this provenance is absent, if an aligned
timestamp differs from the source exposure timestamp, or if any configured timing
limit is exceeded. This intentionally prevents episodes produced by the former
arrival-time alignment logic from entering a training dataset.

Validate locally before transfer:

```bash
./scripts/validate_act_data.sh
./scripts/validate_act_data.sh recordings/act/finalized/<episode-id>
```

Build canonical ACT HDF5 datasets locally before training or transfer:

```bash
PYTHONPATH=src conda run --no-capture-output -n dos-w1 \
  python scripts/prepare_act_training_data.py --replace
```

The exporter conservatively compresses sustained stationary intervals by default.
Both follower observation and leader action for both arms must remain within
`2.5e-3` radians for joints and `5e-4` for raw grippers. A candidate run must last
at least 0.3 seconds and remain within twice those tolerances from its anchor. A
compressed run retains its first, last, and every fifth stationary frame. This
avoids treating slow motion as sensor noise while retaining 20% of the interior
stationary samples. All three cameras, state, action, and timestamps use the same
selected indices.

Tune or disable the behavior explicitly when needed:

```bash
python scripts/prepare_act_training_data.py --replace \
  --stationary-joint-tolerance-rad 0.0025 \
  --stationary-gripper-tolerance 0.0005 \
  --stationary-min-duration-seconds 0.3 \
  --stationary-keep-every-n-frames 5

python scripts/prepare_act_training_data.py --replace --no-stationary-dedup
```

Generate side-by-side three-camera review videos from the processed bundle:

```bash
PYTHONPATH=src conda run --no-capture-output -n dos-w1 \
  python scripts/preview_act_stationary_dedup.py
```

The generated tree is `recordings/act/processed/{act1,act2,act3}/`. Each task
directory contains `episode_0.hdf5`, `episode_1.hdf5`, ... and a
`dataset_manifest.json`. Episodes are grouped from labels beginning with
`act1_`/`act01_`, `act2_`/`act02_`, or `act3_`/`act03_`.

The HDF5 contract follows the common ACT/Aloha layout:

```text
/observations/qpos                       # T x 14
/action                                  # T x 14
/observations/images/cam_high            # T x 480 x 640 x 3 RGB
/observations/images/cam_left_wrist      # T x 480 x 640 x 3 RGB
/observations/images/cam_right_wrist     # T x 480 x 640 x 3 RGB
/camera_frame_index                      # exact source-video indices
/source_aligned_index                    # exact source aligned-row indices
/timestamps/aligned
/timestamps/cameras
```

The 14-value state order is left joints 1-6, raw left gripper, right joints
1-6, raw right gripper. Images are converted to RGB and letterboxed to 640x480
without changing aspect ratio. Inference must use the same state order and raw
gripper convention.
The root HDF5 attribute `timestamp_basis` must be `device_global_time`, and the
same requirement is repeated in each task's `dataset_manifest.json`.
The HDF5 `stationary_dedup_json` attribute and the manifest record the thresholds,
source/retained sample counts, retention ratio, and number of compressed runs.

## Transfer to `zjlab`

The route from `zjlab` back to `dosw1` is currently unavailable, while
`dosw1 -> zjlab` is reachable. Therefore transfers are deliberately initiated
from `dosw1`:

```bash
./scripts/push_act_to_zjlab.sh
```

SSH may ask for the `zjlab` account password; no password is stored by the
project. The script validates strict aligned rows, atomically rebuilds the HDF5
datasets, transfers raw episodes to `data/incoming`, transfers processed datasets
to `data/processed/medicine_pack`, and runs the receiver-side raw validator.

After training, copy a release into the local staging area with:

```bash
./scripts/pull_act_model_from_zjlab.sh
```

This is a staging operation only. It does not load a model, switch a symlink, or
send any robot command.

## First hardware acceptance check

1. Keep both arms stationary and record 5 seconds.
2. Confirm RGB frame count is nonzero and both observation/action streams validate.
3. Move only the left leader slightly during a second recording.
4. Plot or inspect the first seven left-arm values and verify the leader stream
   changes before treating it as the expert action target.
5. Record at least 10 short pilot episodes, convert and overfit a tiny ACT run,
   then begin large-scale collection.

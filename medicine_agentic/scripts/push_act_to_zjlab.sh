#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${ACT_SOURCE_ROOT:-${PROJECT_ROOT}/recordings/act/finalized}"
PROCESSED_ROOT="${ACT_PROCESSED_ROOT:-${PROJECT_ROOT}/recordings/act/processed}"
TRAIN_HOST="${ACT_TRAIN_HOST:-ubuntu@10.42.57.108}"
TRAIN_ROOT="${ACT_TRAIN_ROOT:-/home/ubuntu/act}"

"${PROJECT_ROOT}/scripts/validate_act_data.sh" --root "${SOURCE_ROOT}"
conda run --no-capture-output -n dos-w1 \
  env PYTHONPATH="${PROJECT_ROOT}/src" \
  python "${PROJECT_ROOT}/scripts/prepare_act_training_data.py" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${PROCESSED_ROOT}" \
  --replace

ssh "${TRAIN_HOST}" \
  "mkdir -p '${TRAIN_ROOT}/data/incoming' '${TRAIN_ROOT}/data/processed/medicine_pack'"
rsync -a --human-readable --info=stats2,progress2 --partial \
  --exclude='.*' \
  "${SOURCE_ROOT}/" "${TRAIN_HOST}:${TRAIN_ROOT}/data/incoming/"
rsync -a --human-readable --info=stats2,progress2 --partial --delete \
  --exclude='.*' \
  "${PROCESSED_ROOT}/" "${TRAIN_HOST}:${TRAIN_ROOT}/data/processed/medicine_pack/"
ssh "${TRAIN_HOST}" \
  "cd '${TRAIN_ROOT}' && .venv/bin/python scripts/validate_incoming.py --root data/incoming"

echo "Validated ACT episodes are available at ${TRAIN_HOST}:${TRAIN_ROOT}/data/incoming"
echo "Aligned ACT HDF5 datasets are available at ${TRAIN_HOST}:${TRAIN_ROOT}/data/processed/medicine_pack"

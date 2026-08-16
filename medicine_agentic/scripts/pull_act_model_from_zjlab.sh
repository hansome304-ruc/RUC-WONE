#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_HOST="${ACT_TRAIN_HOST:-ubuntu@10.42.57.108}"
REMOTE_RELEASE="${ACT_REMOTE_RELEASE:-/home/ubuntu/act/releases/latest/}"
LOCAL_STAGING="${ACT_MODEL_STAGING:-${PROJECT_ROOT}/models/act/staging}"

mkdir -p "${LOCAL_STAGING}"
rsync -a --human-readable --info=stats2,progress2 --partial \
  "${TRAIN_HOST}:${REMOTE_RELEASE}" "${LOCAL_STAGING}/"

echo "Model copied into ${LOCAL_STAGING}. No robot deployment or current-model switch was performed."

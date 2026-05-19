#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/train.py \
  --task Dodgeball-G1-v0 \
  --headless \
  --enable_cameras \
  --num_envs 512 \
  --max_iterations 200 \
  --logger wandb \
  --log_project_name Dodgeball-G1 \
  --run_name dodgeball_train \
  --video \
  --video_length 300 \
  --video_interval_iter 50 \
  --video_robot_env_index 0 \
  "$@"

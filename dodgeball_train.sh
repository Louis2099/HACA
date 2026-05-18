#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python scripts/train.py \
  --task Dodgeball-G1-v0 \
  --headless \
  --num_envs 512 \
  --max_iterations 200 \
  --logger wandb \
  --log_project_name Dodgeball-G1 \
  --run_name ppo_smoke \
  "$@"

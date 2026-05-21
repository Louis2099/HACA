#!/usr/bin/env bash
# Train the Dodgeball-G1 staged curriculum.
#
# Usage examples:
#   ./dodgeball_train.sh                              # start from Stage 1 with plain PPO (default)
#   STAGE=2 ./dodgeball_train.sh                      # start from Stage 2
#   STAGE=3 ./dodgeball_train.sh --run_name my_run    # stage 3, custom W&B run name
#   ./dodgeball_train.sh --num_envs 512               # pass any extra train.py flags
#   USE_CBF=1 ./dodgeball_train.sh                    # train with PPO + CBF action filter
#   STAGE=2 RESUME_CHECKPOINT=/path/to/model_stage1_best.pt ./dodgeball_train.sh
#                                                       # branch-finetune Stage 2 from a checkpoint
#   STAGE=2 RESUME_MODE=continue RESUME_LOAD_RUN=... ./dodgeball_train.sh
#                                                       # exact continuation using RSL-RL resolver
#
# The STAGE env var sets both --curriculum_start_stage and the W&B run name suffix.
# Resume env vars:
#   RESUME_CHECKPOINT: absolute/relative checkpoint path, or checkpoint pattern with RESUME_LOAD_RUN
#   RESUME_LOAD_RUN:   optional existing run directory name/pattern for IsaacLab checkpoint resolver
#   RESUME_MODE:       finetune (default: load weights only, reset iteration) or continue
# Any flag passed as a positional argument overrides the defaults below.
set -euo pipefail

cd "$(dirname "$0")"

STAGE="${STAGE:-1}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
USE_CBF="${USE_CBF:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
RESUME_LOAD_RUN="${RESUME_LOAD_RUN:-}"
RESUME_MODE="${RESUME_MODE:-finetune}"

CBF_ARGS=()
case "${USE_CBF,,}" in
  1|true|yes|on)
    CBF_ARGS+=(--use_cbf)
    ;;
  0|false|no|off)
    ;;
  *)
    echo "USE_CBF must be one of: 1, true, yes, on, 0, false, no, off" >&2
    exit 2
    ;;
esac

RESUME_ARGS=()
RUN_NAME="dodgeball_stage${STAGE}_cbf${USE_CBF}"
if [[ -n "${RESUME_CHECKPOINT}" || -n "${RESUME_LOAD_RUN}" ]]; then
  RESUME_ARGS+=(--resume True)
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    RESUME_ARGS+=(--checkpoint "${RESUME_CHECKPOINT}")
  fi
  if [[ -n "${RESUME_LOAD_RUN}" ]]; then
    RESUME_ARGS+=(--load_run "${RESUME_LOAD_RUN}")
  fi

  case "${RESUME_MODE,,}" in
    finetune)
      RESUME_ARGS+=(--load_optimizer False --reset_iteration_on_resume)
      RUN_NAME="dodgeball_stage${STAGE}_finetune_cbf${USE_CBF}"
      ;;
    continue)
      RESUME_ARGS+=(--load_optimizer True)
      RUN_NAME="dodgeball_stage${STAGE}_continue_cbf${USE_CBF}"
      ;;
    *)
      echo "RESUME_MODE must be one of: finetune, continue" >&2
      exit 2
      ;;
  esac
fi

python scripts/train.py \
  --task Dodgeball-G1-v0 \
  --headless \
  --enable_cameras \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --logger wandb \
  --log_project_name Dodgeball-G1 \
  --run_name "${RUN_NAME}" \
  --curriculum_start_stage "${STAGE}" \
  --video \
  --video_length 300 \
  --video_interval_iter 50 \
  --video_robot_env_index 0 \
  "${CBF_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "$@"

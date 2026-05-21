#!/usr/bin/env bash
# Play or inspect the Dodgeball-G1 staged curriculum.
#
# Common usage:
#   ./dodgeball_play.sh
#     Run Stage 4 with zero actions for quick environment/contact visualization.
#
#   CHECKPOINT=logs/rsl_rl/<experiment>/<run>/model_1000.pt ./dodgeball_play.sh
#     Load a trained RSL-RL checkpoint and run it in policy mode.
#
#   POLICY_PATH=logs/rsl_rl/<experiment>/<run>/exported/policy.pt ./dodgeball_play.sh
#     Load an exported TorchScript policy instead of a training checkpoint.
#
#   STAGE=2 ./dodgeball_play.sh
#     Lock playback to one curriculum stage. Stage 1 is static ball; stages 2-4
#     use progressively broader moving-ball target sets.
#
#   MODE=sine ./dodgeball_play.sh
#     Use scripted actions instead of a policy. Supported modes: zero, sine,
#     random, policy. If CHECKPOINT or POLICY_PATH is set, MODE defaults to policy.
#
#   VIDEO=1 VIDEO_LENGTH=600 ./dodgeball_play.sh
#     Record one video. VIDEO_ROBOT_ENV_INDEX chooses which env the camera follows.
#
#   NUM_ENVS=16 NUM_STEPS=2000 ./dodgeball_play.sh --enable-traj-plot
#     Run multiple envs, limit playback length, and pass through any extra
#     play_dodgeball.py flags after the script arguments.
#
#   STAGE=2 BALL_SPEED=6.5 ./dodgeball_play.sh
#     Use a fixed dodgeball launch speed for testing. Leave BALL_SPEED unset for
#     the current randomized stage speed behavior.
#
# Environment variables:
#   STAGE                 Curriculum stage to lock, 1-4. Default: 4.
#   BALL_SPEED            Optional fixed dodgeball launch speed in m/s for testing.
#   MODE                  zero | sine | random | policy. Default: policy when a
#                         checkpoint/policy is provided, otherwise zero.
#   CHECKPOINT            Regular RSL-RL checkpoint, e.g. model_*.pt.
#   POLICY_PATH           Exported TorchScript policy, e.g. exported/policy.pt.
#   NUM_ENVS              Number of parallel environments. Default: 1.
#   NUM_STEPS             Number of control steps, 0 means unlimited. Default: 3000.
#   ACTION_SCALE          Scale for sine/random actions. Default: 0.25.
#   VIDEO                 Set to 1 to record a video. Default: 0.
#   VIDEO_LENGTH          Video length in steps. Default: 600.
#   VIDEO_DIR             Directory for video output. Default: logs/videos/play_dodgeball.
#   VIDEO_ROBOT_ENV_INDEX Environment index used for video camera/diagnostics. Default: 0.
#   REAL_TIME             Set to 1 to sleep toward real-time playback. Default: 0.
#   HEADLESS              Set to 1 to run without GUI. Default: 0.
set -euo pipefail

cd "$(dirname "$0")"

STAGE="${STAGE:-4}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-3000}"
ACTION_SCALE="${ACTION_SCALE:-0.25}"
VIDEO="${VIDEO:-0}"
VIDEO_LENGTH="${VIDEO_LENGTH:-600}"
VIDEO_DIR="${VIDEO_DIR:-}"
VIDEO_ROBOT_ENV_INDEX="${VIDEO_ROBOT_ENV_INDEX:-0}"
REAL_TIME="${REAL_TIME:-0}"
HEADLESS="${HEADLESS:-0}"
CHECKPOINT="${CHECKPOINT:-}"
POLICY_PATH="${POLICY_PATH:-}"
BALL_SPEED="${BALL_SPEED:-}"

if [[ -n "${CHECKPOINT}" || -n "${POLICY_PATH}" ]]; then
  MODE="${MODE:-policy}"
else
  MODE="${MODE:-zero}"
fi

args=(
  scripts/play_dodgeball.py
  --task Dodgeball-G1-v0
  --num_envs "${NUM_ENVS}"
  --num_steps "${NUM_STEPS}"
  --mode "${MODE}"
  --action_scale "${ACTION_SCALE}"
  --curriculum_stage "${STAGE}"
  --video_robot_env_index "${VIDEO_ROBOT_ENV_INDEX}"
)

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi

if [[ -n "${POLICY_PATH}" ]]; then
  args+=(--policy_path "${POLICY_PATH}")
fi

if [[ -n "${BALL_SPEED}" ]]; then
  args+=(--ball_speed "${BALL_SPEED}")
fi

if [[ "${VIDEO}" == "1" ]]; then
  args+=(--video --video_length "${VIDEO_LENGTH}")
  if [[ -n "${VIDEO_DIR}" ]]; then
    args+=(--video_dir "${VIDEO_DIR}")
  fi
fi

if [[ "${REAL_TIME}" == "1" ]]; then
  args+=(--real_time)
fi

if [[ "${HEADLESS}" == "1" ]]; then
  args+=(--headless)
fi

python "${args[@]}" "$@"

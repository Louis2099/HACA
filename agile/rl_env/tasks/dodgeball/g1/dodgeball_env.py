# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom ManagerBasedRLEnv subclass for the staged dodgeball curriculum.

This class adds:
  - 4-stage curriculum with automatic advancement based on epoch-averaged success rate.
  - Per-stage ball spawn behaviour (static in Stage 1, moving in Stages 2–4).
  - Instant reward-weight switch when entering a new stage.
  - Per-stage best-checkpoint requests for the training runner.
  - W&B-compatible logging of all curriculum metrics via extras["log"].

The observation space is **unchanged** across all stages (97-D policy obs).
Collision detection uses the same whole-body sensor in every stage; only the
ball's aiming *target* link set changes per stage.
"""

from __future__ import annotations

from collections import deque

import torch

from isaaclab.envs import ManagerBasedRLEnv

from .dodgeball_env_cfg import G1DodgeballEnvCfg, REWARD_WEIGHTS_BY_STAGE, STAGE_BALL_SPEED


class DodgeballEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with a 4-stage dodgeball curriculum."""

    cfg: G1DodgeballEnvCfg

    def __init__(self, cfg: G1DodgeballEnvCfg, **kwargs):
        # Pre-create _stage1_ball_pose as None so the reset event (called inside
        # super().__init__) can detect its presence via hasattr() and write to it
        # once the tensor is allocated below.
        self._stage1_ball_pose: torch.Tensor | None = None
        # Reset events run inside super().__init__(), so expose the requested
        # stage before the first reset samples the dodgeball state.
        if cfg.use_staged_curriculum:
            self.curriculum_stage: int = int(cfg.curriculum_start_stage)
            self.stage_start_step: int = 0

        super().__init__(cfg, **kwargs)

        if not cfg.use_staged_curriculum:
            return

        # ── Curriculum state ──────────────────────────────────────────────────
        self.curriculum_stage = int(getattr(self, "curriculum_stage", cfg.curriculum_start_stage))
        # Global step at which the current stage started (for per-stage speed ramp).
        self.stage_start_step = int(getattr(self, "stage_start_step", 0))

        # Per-epoch success tracking (one RL epoch = curriculum_epoch_steps env steps).
        self._cur_epoch_successes: list[float] = []
        self._epoch_success_deque: deque[float] = deque(maxlen=cfg.curriculum_advance_epochs)
        self._last_epoch_step: int = 0
        self._total_episodes: int = 0
        self._stage_episodes: int = 0

        # Per-stage best checkpoint (requested by runner when epoch rate improves).
        self._stage_best_success_rate: float = 0.0
        self._best_ckpt_request: dict | None = None
        self._harness_scale: float = 1.0 if self.curriculum_stage == 1 else 0.0
        self._stage1_harness_removed: bool = self.curriculum_stage != 1

        # ── Per-episode accumulators (shape [num_envs]) ───────────────────────
        # Whether the dodgeball contacted the robot at any point this episode.
        self._ep_had_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── Stage-1 ball freeze pose ──────────────────────────────────────────
        # In Stage 1 the ball has zero velocity but gravity is still active, which
        # would cause it to fall and hit the ground (triggering the contact sensor).
        # We prevent this by writing the frozen pose + zero velocity back to the
        # physics simulation at the start of every control step while in Stage 1.
        #
        # _stage1_ball_pose [N, 7] = (pos_xyz, quat_wxyz) of the target freeze position
        # for each env.  It is updated by reset_dodgeball_towards_curriculum_target
        # (Stage 1 branch) each time an episode resets so the pose always matches
        # the randomised spawn position chosen for that episode.
        dodgeball = self.scene["dodgeball"]
        self._stage1_ball_pose = torch.cat(
            [dodgeball.data.root_pos_w.clone(), dodgeball.data.root_quat_w.clone()],
            dim=-1,
        )  # [N, 7]

        # ── Termination term indices (resolved once after super().__init__) ───
        self._term_idx_ball_hit: int | None = self._find_term_idx("dodgeball_hit_upper_body")
        self._term_idx_illegal: int | None = self._find_term_idx("illegal_contacts")

        # ── Apply initial stage weights immediately ───────────────────────────
        self._apply_stage_weights(self.curriculum_stage)
        self._update_harness_scale()

    # ─────────────────────────────────────────────────────────────────────────
    # Step override
    # ─────────────────────────────────────────────────────────────────────────

    def step(self, action: torch.Tensor):
        if not self.cfg.use_staged_curriculum:
            return super().step(action)

        self._update_harness_scale()

        # Stage 1: keep the ball stationary by writing its frozen pose and zero
        # velocity to the physics sim BEFORE each control step.  Without this,
        # gravity pulls the ball to the ground every episode, which (a) causes
        # ground-contact forces that can leak through the filtered contact sensor
        # and (b) triggers the ball-contact termination spuriously.
        if self.curriculum_stage == 1 and self._stage1_ball_pose is not None:
            self._freeze_stage1_ball()

        obs, rewards, terminated, time_outs, extras = super().step(action)

        # Update per-episode ball-contact accumulator using the termination
        # manager's per-term buffers (populated during compute() this step).
        self._update_ep_contact()

        # Identify envs that just completed an episode.
        reset_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self._process_finished_episodes(reset_ids)
            # Clear accumulators for envs that just reset.
            self._ep_had_ball_contact[reset_ids] = False

        # Attempt stage advancement.
        stage_changed = False
        if self.cfg.curriculum_auto_advance:
            stage_changed = self._maybe_advance_stage()

        # super().step() has already reset reset_ids using the previous stage.
        # If the reset closed an epoch that advanced the curriculum, immediately
        # re-sample only the dodgeball state for those just-reset envs so the
        # next episode matches the new stage.
        if stage_changed and len(reset_ids) > 0:
            self._reset_stage_dependent_terms_for_current_stage(reset_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.obs_buf = self.observation_manager.compute(update_history=True)
            obs = self.obs_buf

        # Log curriculum info into extras["log"] for W&B.
        self._log_curriculum(extras)

        return obs, rewards, terminated, time_outs, extras

    # ─────────────────────────────────────────────────────────────────────────
    # Stage-1 ball freeze
    # ─────────────────────────────────────────────────────────────────────────

    def _freeze_stage1_ball(self) -> None:
        """Write the stored frozen pose + zero velocity to the ball each step.

        Called before super().step() so the physics loop sees a stationary ball.
        The pose is updated by reset_dodgeball_towards_curriculum_target (Stage 1
        branch) after each episode reset, ensuring the ball stays at its
        randomised spawn position rather than falling to the ground.
        """
        dodgeball = self.scene["dodgeball"]
        all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        zero_vel = torch.zeros(self.num_envs, 6, device=self.device)
        dodgeball.write_root_pose_to_sim(self._stage1_ball_pose, env_ids=all_ids)
        dodgeball.write_root_velocity_to_sim(zero_vel, env_ids=all_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-episode contact tracking
    # ─────────────────────────────────────────────────────────────────────────

    def _update_ep_contact(self) -> None:
        """OR the ball-hit termination signal into the per-episode contact flag."""
        if self._term_idx_ball_hit is None:
            return
        ball_hit_this_step: torch.Tensor = (
            self.termination_manager._term_dones[:, self._term_idx_ball_hit]
        )
        self._ep_had_ball_contact |= ball_hit_this_step

    # ─────────────────────────────────────────────────────────────────────────
    # Episode success computation
    # ─────────────────────────────────────────────────────────────────────────

    def _process_finished_episodes(self, reset_ids: torch.Tensor) -> None:
        """Compute per-episode success for envs in reset_ids and push to current epoch."""
        # Success = explicit ball-rested settle success, or the legacy fallback
        # of reaching timeout without an early termination/contact.
        # Settle imbalance is a terminated episode and does not count as success.
        had_contact = self._ep_had_ball_contact[reset_ids]
        timeout_success = ~self.reset_terminated[reset_ids] & ~had_contact
        settle_success_buf = getattr(self, "_dodgeball_settle_success", None)
        if settle_success_buf is None:
            settle_success = torch.zeros_like(timeout_success)
        else:
            settle_success = settle_success_buf[reset_ids]
        success = settle_success | timeout_success

        for s in success.tolist():
            self._cur_epoch_successes.append(float(s))
        self._total_episodes += len(reset_ids)
        self._stage_episodes += len(reset_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Stage advancement
    # ─────────────────────────────────────────────────────────────────────────

    def _close_epoch(self) -> float:
        """Finalize the current epoch, update best-checkpoint tracking, return epoch rate."""
        if self._cur_epoch_successes:
            epoch_rate = sum(self._cur_epoch_successes) / len(self._cur_epoch_successes)
        else:
            epoch_rate = 0.0

        self._epoch_success_deque.append(epoch_rate)
        self._cur_epoch_successes.clear()

        if self._best_checkpoint_allowed() and epoch_rate > self._stage_best_success_rate:
            self._stage_best_success_rate = epoch_rate
            self._best_ckpt_request = {
                "stage": self.curriculum_stage,
                "filename": f"model_stage{self.curriculum_stage}_best.pt",
            }

        return epoch_rate

    def _maybe_advance_stage(self) -> bool:
        """Advance when mean success over the last N epochs meets the threshold."""
        if self.curriculum_stage >= 4:
            return False

        current_step = int(self.common_step_counter)
        epoch_steps = self.cfg.curriculum_epoch_steps
        min_eps = self.cfg.curriculum_min_episodes_before_advance
        advance_epochs = self.cfg.curriculum_advance_epochs
        threshold = self.cfg.curriculum_success_threshold

        if self._stage_episodes < min_eps:
            return False

        # Close epoch when enough env steps have elapsed since the last boundary.
        if current_step - self._last_epoch_step >= epoch_steps:
            epoch_rate = self._close_epoch()
            self._last_epoch_step = current_step
            print(
                f"[DodgeballCurriculum] Epoch closed at step {current_step} "
                f"(stage={self.curriculum_stage}, epoch_rate={epoch_rate:.3f})"
            )

        if self.curriculum_stage == 1 and not self._stage1_harness_removed:
            return False

        if len(self._epoch_success_deque) < advance_epochs:
            return False

        mean_epoch_rate = sum(self._epoch_success_deque) / len(self._epoch_success_deque)
        if mean_epoch_rate < threshold:
            return False

        prev_stage = self.curriculum_stage
        self.curriculum_stage += 1
        self.stage_start_step = current_step
        self._apply_stage_weights(self.curriculum_stage)

        # Reset epoch tracking for the new stage.
        self._cur_epoch_successes.clear()
        self._epoch_success_deque.clear()
        self._last_epoch_step = current_step
        self._stage_episodes = 0
        self._stage_best_success_rate = 0.0
        self._stage1_harness_removed = True
        self._update_harness_scale()

        print(
            f"[DodgeballCurriculum] Stage {prev_stage} → {self.curriculum_stage} "
            f"at step {current_step} (mean_epoch_success_rate={mean_epoch_rate:.2f})"
        )
        return True

    def _reset_stage_dependent_terms_for_current_stage(self, env_ids: torch.Tensor) -> None:
        """Re-apply stage-dependent reset terms after a same-step stage change."""
        try:
            joints_cfg = self.event_manager.get_term_cfg("reset_robot_joints")
        except ValueError:
            joints_cfg = None
        if joints_cfg is not None:
            joints_cfg.func(self, env_ids, **joints_cfg.params)
            self.scene.write_data_to_sim()
            self.sim.forward()

        try:
            dodgeball_cfg = self.event_manager.get_term_cfg("reset_dodgeball")
        except ValueError:
            return
        dodgeball_cfg.func(self, env_ids, **dodgeball_cfg.params)

    # ─────────────────────────────────────────────────────────────────────────
    # Harness schedule
    # ─────────────────────────────────────────────────────────────────────────

    def _update_harness_scale(self) -> None:
        """Apply the staged harness schedule to the 0-D harness action."""
        if self.curriculum_stage != 1:
            self._set_harness_scale(0.0)
            return

        start = int(self.cfg.stage1_harness_decay_start_step)
        decay_steps = max(int(self.cfg.stage1_harness_decay_steps), 1)
        step = int(self.common_step_counter)

        if step <= start:
            scale = 1.0
        elif step >= start + decay_steps:
            scale = 0.0
        else:
            scale = 1.0 - float(step - start) / float(decay_steps)

        self._set_harness_scale(scale)
        if scale <= 0.0 and not self._stage1_harness_removed:
            self._stage1_harness_removed = True
            self._cur_epoch_successes.clear()
            self._epoch_success_deque.clear()
            self._last_epoch_step = step
            self._stage_episodes = 0
            self._stage_best_success_rate = 0.0
            self._best_ckpt_request = None
            print(
                f"[DodgeballCurriculum] Stage 1 harness fully removed at step {step}; "
                "resetting success history for unassisted advancement."
            )

    def _set_harness_scale(self, scale: float) -> None:
        self._harness_scale = float(max(0.0, min(1.0, scale)))
        harness = self.action_manager._terms.get("harness")
        if harness is not None and hasattr(harness, "scale_forces"):
            harness.scale_forces(self._harness_scale)

    def _best_checkpoint_allowed(self) -> bool:
        return self.curriculum_stage != 1 or self._stage1_harness_removed

    # ─────────────────────────────────────────────────────────────────────────
    # Reward weights
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_stage_weights(self, stage: int) -> None:
        """Write current-stage reward weights to the reward manager."""
        curr_weights = REWARD_WEIGHTS_BY_STAGE.get(stage, {})
        for term_name, curr_w in curr_weights.items():
            try:
                term_cfg = self.reward_manager.get_term_cfg(term_name)
            except ValueError:
                continue
            term_cfg.weight = curr_w

    # ─────────────────────────────────────────────────────────────────────────
    # Curriculum logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_curriculum(self, extras: dict) -> None:
        """Write curriculum metrics to extras['log'] for W&B consumption."""
        if "log" not in extras:
            extras["log"] = {}

        if self._epoch_success_deque:
            epoch_success_rate_mean20 = sum(self._epoch_success_deque) / len(self._epoch_success_deque)
        else:
            epoch_success_rate_mean20 = 0.0

        if self._cur_epoch_successes:
            cur_epoch_rate = sum(self._cur_epoch_successes) / len(self._cur_epoch_successes)
        else:
            cur_epoch_rate = 0.0

        stage_speed = STAGE_BALL_SPEED.get(self.curriculum_stage, (0.0, 0.0, 0))
        steps_in_stage = int(self.common_step_counter) - self.stage_start_step
        ramp_steps = max(stage_speed[2], 1)
        speed_progress = min(float(steps_in_stage) / ramp_steps, 1.0)
        # 10-section discretisation (matches dodgeball_events.py).
        section = min(int(speed_progress * 10), 9)
        stage_progress = (section + 1) / 10.0
        speed_min, speed_max, _ = stage_speed
        max_launch_speed = speed_min + (section + 1) * (speed_max - speed_min) / 10.0

        extras["log"].update({
            "curriculum/stage":                      float(self.curriculum_stage),
            "curriculum/stage_progress":             stage_progress,
            "curriculum/cur_epoch_success_rate":     cur_epoch_rate,
            "curriculum/epoch_success_rate_mean20":  epoch_success_rate_mean20,
            "curriculum/stage_best_success_rate":    self._stage_best_success_rate,
            "curriculum/max_launch_speed":           max_launch_speed,
            "curriculum/total_episodes":             float(self._total_episodes),
            "curriculum/stage_episodes":             float(self._stage_episodes),
            "curriculum/harness_scale":              self._harness_scale,
            "curriculum/stage1_harness_removed":     float(self._stage1_harness_removed),
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _find_term_idx(self, name: str) -> int | None:
        """Return the index into termination_manager._term_dones for the given term name."""
        idx_map: dict[str, int] = getattr(self.termination_manager, "_term_name_to_term_idx", {})
        return idx_map.get(name, None)

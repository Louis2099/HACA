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

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def _resolve_body_ids(robot: Articulation, patterns: tuple[str, ...], device: torch.device) -> torch.Tensor:
    body_ids: list[int] = []
    for pattern in patterns:
        try:
            found_ids, _ = robot.find_bodies(pattern)
        except Exception:
            continue
        if isinstance(found_ids, torch.Tensor):
            body_ids.extend(found_ids.to(dtype=torch.long, device=device).tolist())
        elif isinstance(found_ids, list):
            body_ids.extend(int(i) for i in found_ids)
    if not body_ids:
        return torch.empty(0, dtype=torch.long, device=device)
    unique_ids = sorted(set(body_ids))
    return torch.tensor(unique_ids, dtype=torch.long, device=device)


def _validate_and_resolve_body_ids(
    robot: Articulation,
    patterns: tuple[str, ...],
    device: torch.device,
    context_label: str = "",
) -> torch.Tensor:
    """Like _resolve_body_ids, but warns when a pattern matches no bodies."""
    body_ids: list[int] = []
    for pattern in patterns:
        try:
            found_ids, _ = robot.find_bodies(pattern)
        except Exception:
            print(f"[DodgeballCurriculum{context_label}] WARNING: pattern '{pattern}' raised an exception.")
            continue
        if isinstance(found_ids, torch.Tensor):
            matched = found_ids.to(dtype=torch.long, device=device).tolist()
        elif isinstance(found_ids, list):
            matched = [int(i) for i in found_ids]
        else:
            matched = []
        if not matched:
            print(f"[DodgeballCurriculum{context_label}] WARNING: pattern '{pattern}' matched no bodies — skipping.")
        body_ids.extend(matched)
    if not body_ids:
        return torch.empty(0, dtype=torch.long, device=device)
    unique_ids = sorted(set(body_ids))
    return torch.tensor(unique_ids, dtype=torch.long, device=device)


def _curriculum_target_patterns(
    env: ManagerBasedRLEnv,
    easy_target_body_patterns: tuple[str, ...],
    medium_target_body_patterns: tuple[str, ...],
    hard_target_body_patterns: tuple[str, ...],
    curriculum_switch_steps: tuple[int, int],
) -> tuple[str, ...]:
    step = int(env.common_step_counter)
    first_switch, second_switch = curriculum_switch_steps
    if step < first_switch:
        return easy_target_body_patterns
    if step < second_switch:
        return medium_target_body_patterns
    return hard_target_body_patterns


def reset_dodgeball_towards_curriculum_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    launch_distance_range: tuple[float, float],
    launch_height_range: tuple[float, float],
    front_half_angle_deg: float,
    time_to_impact_range: tuple[float, float],
    lateral_noise_range: tuple[float, float],
    vertical_noise_range: tuple[float, float],
    # ── Staged curriculum mode ────────────────────────────────────────────────
    # When True, reads curriculum_stage from env and uses the module-level
    # STAGE_TARGET_PATTERNS / STAGE_BALL_SPEED constants (imported below).
    # These are NOT passed as params to avoid Isaac Lab's config serializer
    # choking on int-keyed dicts.
    use_staged_curriculum: bool = False,
    # ── Legacy fallback params (used when use_staged_curriculum=False) ────────
    max_launch_speed_start: float = 4.5,
    max_launch_speed_end: float = 10.0,
    max_launch_speed_curriculum_steps: int = 300_000,
    easy_target_body_patterns: tuple[str, ...] = (),
    medium_target_body_patterns: tuple[str, ...] = (),
    hard_target_body_patterns: tuple[str, ...] = (),
    curriculum_switch_steps: tuple[int, int] = (100_000, 250_000),
    debug_print_world_z: bool = False,
    randomize_curriculum_speed_for_debug: bool = False,
    custom_launch_speed: float | None = None,
    target_root_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),  # noqa: B008
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),      # noqa: B008
) -> None:
    """Reset dodgeball with stage-aware targeting and speed, or legacy step-counter ramp.

    Staged mode (use_staged_curriculum=True):
      - Stage 1: ball is placed at a random position in the frontal FOV sector with
        zero velocity (static ball).  Ball observation is still present so the 97-D
        policy observation dimension stays constant.
      - Stages 2–4: moving ball aimed at STAGE_TARGET_PATTERNS[stage]; speed ramps
        from STAGE_BALL_SPEED[stage][0] to [1] over [2] steps since stage_start.
        Both constants are imported from dodgeball_env_cfg at call time to avoid
        int-keyed-dict serialization issues with Isaac Lab's config system.

    Legacy mode (use_staged_curriculum=False):
      - Step-counter ramp from max_launch_speed_start to max_launch_speed_end.
      - Target-body three-level curriculum via curriculum_switch_steps.

    Collision detection uses the same whole-body dodgeball_robot_contact sensor in
    every stage — only the aiming target changes.

    If custom_launch_speed is set, playback/debug runs use that fixed speed in
    place of the staged or legacy random speed sampler.  Leave it as None for
    normal randomized training/evaluation behavior.

    If target_root_height is set, the target body's z coordinate is corrected so
    the launch aims at the same articulated pose after removing the reset-time
    root height offset.  This avoids aiming above the robot just because the
    reset pose spawns the feet slightly in the air before contacts settle.
    """
    object_asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # ── Torso-aligned frontal launch sector ──────────────────────────────────
    torso_ids = _resolve_body_ids(robot, ("torso_link",), object_asset.device)
    if len(torso_ids) == 0:
        torso_pos_w = robot.data.root_pos_w[env_ids]
        torso_quat_w = robot.data.root_quat_w[env_ids]
    else:
        torso_body_id = int(torso_ids[0])
        torso_pos_w = robot.data.body_pos_w[env_ids, torso_body_id, :]
        torso_quat_w = robot.data.body_quat_w[env_ids, torso_body_id, :]

    num_envs = len(env_ids)
    launch_distance = math_utils.sample_uniform(
        launch_distance_range[0], launch_distance_range[1], (num_envs,), device=object_asset.device
    )
    front_half_angle_rad = torch.tensor(front_half_angle_deg * torch.pi / 180.0, device=object_asset.device)
    launch_heading = math_utils.sample_uniform(
        -front_half_angle_rad, front_half_angle_rad, (num_envs,), device=object_asset.device
    )
    local_xy = torch.zeros((num_envs, 3), device=object_asset.device)
    local_xy[:, 0] = launch_distance * torch.cos(launch_heading)
    local_xy[:, 1] = launch_distance * torch.sin(launch_heading)

    yaw_only_quat = math_utils.yaw_quat(torso_quat_w)
    world_xy_offset = math_utils.quat_apply(yaw_only_quat, local_xy)
    launch_pos = torso_pos_w + world_xy_offset
    launch_height = math_utils.sample_uniform(
        launch_height_range[0], launch_height_range[1], (num_envs,), device=object_asset.device
    )
    launch_pos[:, 2] = env.scene.env_origins[env_ids, 2] + torch.clamp(launch_height, min=1.0)

    rot_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]],
        device=object_asset.device,
    )
    rot_offsets = math_utils.sample_uniform(
        rot_ranges[:, 0], rot_ranges[:, 1], (num_envs, 3), device=object_asset.device
    )
    launch_rot = math_utils.quat_from_euler_xyz(rot_offsets[:, 0], rot_offsets[:, 1], rot_offsets[:, 2])

    # ── Stage-aware vs legacy branch ─────────────────────────────────────────
    if use_staged_curriculum:
        # Import the stage-keyed constants here rather than receiving them as
        # EventTerm params, because Isaac Lab's config serializer cannot handle
        # dicts with int keys.
        from agile.rl_env.tasks.dodgeball.g1.dodgeball_env_cfg import (
            STAGE_BALL_SPEED,
            STAGE_TARGET_PATTERNS,
        )

        stage = int(getattr(env, "curriculum_stage", 1))

        if stage == 1:
            # Static ball: place at randomized position in the FOV, zero velocity.
            # The ball obs (position, velocity, time_to_impact) is still computed
            # so the 97-D policy observation dimension is unchanged.
            #
            # We also store the pose in env._stage1_ball_pose so that
            # DodgeballEnv._freeze_stage1_ball() can write it back to the physics
            # sim every control step, preventing gravity from pulling the ball to
            # the ground and causing spurious contact-sensor terminations.
            zero_vel = torch.zeros((num_envs, 6), device=object_asset.device)
            pose = torch.cat([launch_pos, launch_rot], dim=-1)
            object_asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            object_asset.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
            if getattr(env, "_stage1_ball_pose", None) is not None:
                env._stage1_ball_pose[env_ids] = pose.to(env._stage1_ball_pose.device)
            return

        # Stages 2–4: moving ball.
        target_patterns = STAGE_TARGET_PATTERNS.get(stage, ())
        target_body_ids = _validate_and_resolve_body_ids(
            robot, target_patterns, object_asset.device, context_label=f"stage{stage}"
        )
        if len(target_body_ids) == 0:
            target_body_ids = _resolve_body_ids(robot, ("torso_link", "pelvis"), object_asset.device)

        speed_min, speed_max, ramp_steps = STAGE_BALL_SPEED.get(stage, (4.0, 10.0, 200_000))
        stage_start_step = int(getattr(env, "stage_start_step", 0))
        steps_in_stage = int(env.common_step_counter) - stage_start_step
        if ramp_steps <= 0:
            speed_progress = 1.0
        else:
            speed_progress = min(float(steps_in_stage) / float(ramp_steps), 1.0)
        # 10-section discretisation: upper bound grows in steps of (max-min)/10.
        section = min(int(speed_progress * 10), 9)
        current_max_launch_speed = speed_min + (section + 1) * (speed_max - speed_min) / 10.0
        use_staged_speed_sampling = True
        if custom_launch_speed is not None:
            speed_min = float(custom_launch_speed)
            current_max_launch_speed = float(custom_launch_speed)

    else:
        use_staged_speed_sampling = False
        # ── Legacy step-counter ramp ──────────────────────────────────────────
        target_patterns = _curriculum_target_patterns(
            env=env,
            easy_target_body_patterns=easy_target_body_patterns,
            medium_target_body_patterns=medium_target_body_patterns,
            hard_target_body_patterns=hard_target_body_patterns,
            curriculum_switch_steps=curriculum_switch_steps,
        )
        target_body_ids = _resolve_body_ids(robot, target_patterns, object_asset.device)
        if len(target_body_ids) == 0:
            target_body_ids = _resolve_body_ids(robot, ("torso_link", "pelvis"), object_asset.device)

        if max_launch_speed_curriculum_steps <= 0:
            curriculum_scale = 1.0
        else:
            curriculum_scale = min(
                float(env.common_step_counter) / float(max_launch_speed_curriculum_steps), 1.0
            )
        current_max_launch_speed = (
            max_launch_speed_start + (max_launch_speed_end - max_launch_speed_start) * curriculum_scale
        )
        if randomize_curriculum_speed_for_debug:
            current_max_launch_speed = float(
                math_utils.sample_uniform(
                    max_launch_speed_start, max_launch_speed_end, (1,), device=object_asset.device
                )[0].item()
            )
        if custom_launch_speed is not None:
            current_max_launch_speed = float(custom_launch_speed)

    # ── Sample target position with noise ────────────────────────────────────
    sampled_target_idx = torch.randint(0, len(target_body_ids), (num_envs,), device=object_asset.device)
    selected_body_ids = target_body_ids[sampled_target_idx]
    target_pos = robot.data.body_pos_w[env_ids, selected_body_ids, :].clone()
    if target_root_height is not None:
        nominal_root_z = env.scene.env_origins[env_ids, 2] + float(target_root_height)
        spawn_z_offset = robot.data.root_pos_w[env_ids, 2] - nominal_root_z
        target_pos[:, 2] -= spawn_z_offset

    target_pos[:, 1] += math_utils.sample_uniform(
        lateral_noise_range[0], lateral_noise_range[1], (num_envs,), device=object_asset.device
    )
    target_pos[:, 2] += math_utils.sample_uniform(
        vertical_noise_range[0], vertical_noise_range[1], (num_envs,), device=object_asset.device
    )

    # ── Ballistic velocity with gravity compensation ──────────────────────────
    displacement = target_pos - launch_pos
    distance = torch.norm(displacement, dim=-1).clamp_min(1.0e-6)

    if custom_launch_speed is not None:
        sampled_speed = torch.full(
            (num_envs,),
            float(custom_launch_speed),
            device=object_asset.device,
        ).clamp_min(1.0e-3)
        time_to_impact = (distance / sampled_speed).clamp(
            time_to_impact_range[0], time_to_impact_range[1]
        )
    elif use_staged_curriculum and use_staged_speed_sampling:
        # Sample speed uniformly in [speed_min, current_max_launch_speed] so each
        # episode draws from the full enlarged region, not a fixed speed.
        sampled_speed = math_utils.sample_uniform(
            speed_min, current_max_launch_speed, (num_envs,), device=object_asset.device
        ).clamp_min(1.0e-3)
        time_to_impact = (distance / sampled_speed).clamp(
            time_to_impact_range[0], time_to_impact_range[1]
        )
    else:
        sampled_time_to_impact = math_utils.sample_uniform(
            time_to_impact_range[0], time_to_impact_range[1], (num_envs,), device=object_asset.device
        ).clamp_min(1.0e-3)
        min_time_to_impact = distance / max(current_max_launch_speed, 1.0e-6)
        time_to_impact = torch.maximum(sampled_time_to_impact, min_time_to_impact)

    gravity = 9.81
    desired_lin_vel = displacement / time_to_impact.unsqueeze(-1)
    desired_lin_vel[:, 2] = desired_lin_vel[:, 2] + 0.5 * gravity * time_to_impact

    ang_vel = math_utils.sample_uniform(-2.0, 2.0, (num_envs, 3), device=object_asset.device)
    launch_vel = torch.cat([desired_lin_vel, ang_vel], dim=-1)

    object_asset.write_root_pose_to_sim(torch.cat([launch_pos, launch_rot], dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(launch_vel, env_ids=env_ids)


def reset_dodgeball_joints(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    use_staged_curriculum: bool = False,
    hip_roll_abs_min: float = 0.05,
    hip_roll_abs_max: float = 0.30,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> None:
    """Reset robot joints via scale, then override hip-roll in Stage 1.

    When ``use_staged_curriculum=True`` and the current stage is 1, the hip-roll
    joints are re-sampled from a natural Uniform range instead of scale × default_pos
    (which always produces a very wide stance):

        left_hip_roll_joint:  Uniform(+hip_roll_abs_min, +hip_roll_abs_max)  [rad]
        right_hip_roll_joint: Uniform(-hip_roll_abs_max, -hip_roll_abs_min)  [rad]

    For Stages 2–4, or when ``use_staged_curriculum=False``, the standard
    scale × default_pos behaviour is used unchanged.
    """
    import isaaclab.utils.math as math_utils_local
    from isaaclab.assets import Articulation

    robot: Articulation = env.scene[asset_cfg.name]

    if asset_cfg.joint_ids != slice(None):
        iter_env_ids = env_ids[:, None]
    else:
        iter_env_ids = env_ids

    # Standard scale-based randomization.
    joint_pos = robot.data.default_joint_pos[iter_env_ids, asset_cfg.joint_ids].clone()
    joint_vel = robot.data.default_joint_vel[iter_env_ids, asset_cfg.joint_ids].clone()

    joint_pos *= math_utils_local.sample_uniform(*position_range, joint_pos.shape, robot.device)
    joint_vel *= math_utils_local.sample_uniform(*velocity_range, joint_vel.shape, robot.device)

    joint_pos_limits = robot.data.soft_joint_pos_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = robot.data.soft_joint_vel_limits[iter_env_ids, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # ── Stage-1 hip-roll override ─────────────────────────────────────────────
    stage = int(getattr(env, "curriculum_stage", 1))
    if use_staged_curriculum and stage == 1:
        num_envs_local = len(env_ids)

        def _find_joint_local_idx(name_pattern: str) -> int | None:
            try:
                ids, _ = robot.find_joints(name_pattern)
            except Exception:
                return None
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if not ids:
                return None
            joint_id_global = int(ids[0])
            # Map global joint id to local index in the joint_ids slice.
            if asset_cfg.joint_ids == slice(None):
                return joint_id_global
            if isinstance(asset_cfg.joint_ids, list):
                try:
                    return asset_cfg.joint_ids.index(joint_id_global)
                except ValueError:
                    return None
            return None

        left_idx = _find_joint_local_idx("left_hip_roll_joint")
        right_idx = _find_joint_local_idx("right_hip_roll_joint")

        if left_idx is not None:
            joint_pos[:, left_idx] = math_utils_local.sample_uniform(
                hip_roll_abs_min, hip_roll_abs_max, (num_envs_local,), robot.device
            )
        if right_idx is not None:
            joint_pos[:, right_idx] = math_utils_local.sample_uniform(
                -hip_roll_abs_max, -hip_roll_abs_min, (num_envs_local,), robot.device
            )

    robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=asset_cfg.joint_ids, env_ids=env_ids)

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
    max_launch_speed_start: float,
    max_launch_speed_end: float,
    max_launch_speed_curriculum_steps: int,
    time_to_impact_range: tuple[float, float],
    lateral_noise_range: tuple[float, float],
    vertical_noise_range: tuple[float, float],
    easy_target_body_patterns: tuple[str, ...],
    medium_target_body_patterns: tuple[str, ...],
    hard_target_body_patterns: tuple[str, ...],
    curriculum_switch_steps: tuple[int, int],
    debug_print_world_z: bool = False,
    randomize_curriculum_speed_for_debug: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset dodgeball so its velocity is directed to intersect selected robot link targets."""
    object_asset: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # Determine a torso-aligned frontal launch sector (120 deg total by default).
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
    # Ensure launch z is never too low for human-like dodgeball throws.
    launch_pos[:, 2] = env.scene.env_origins[env_ids, 2] + torch.clamp(launch_height, min=1.0)

    rot_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]],
        device=object_asset.device,
    )
    rot_offsets = math_utils.sample_uniform(rot_ranges[:, 0], rot_ranges[:, 1], (num_envs, 3), device=object_asset.device)
    launch_rot = math_utils.quat_from_euler_xyz(rot_offsets[:, 0], rot_offsets[:, 1], rot_offsets[:, 2])

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

    sampled_target_idx = torch.randint(0, len(target_body_ids), (len(env_ids),), device=object_asset.device)
    selected_body_ids = target_body_ids[sampled_target_idx]
    target_pos = robot.data.body_pos_w[env_ids, selected_body_ids, :]

    target_pos[:, 1] += math_utils.sample_uniform(
        lateral_noise_range[0], lateral_noise_range[1], (len(env_ids),), device=object_asset.device
    )
    target_pos[:, 2] += math_utils.sample_uniform(
        vertical_noise_range[0], vertical_noise_range[1], (len(env_ids),), device=object_asset.device
    )

    sampled_time_to_impact = math_utils.sample_uniform(
        time_to_impact_range[0], time_to_impact_range[1], (num_envs,), device=object_asset.device
    ).clamp_min(1.0e-3)
    # Curriculum on maximum launch speed.
    if max_launch_speed_curriculum_steps <= 0:
        curriculum_scale = 1.0
    else:
        curriculum_scale = min(float(env.common_step_counter) / float(max_launch_speed_curriculum_steps), 1.0)
    current_max_launch_speed = max_launch_speed_start + (max_launch_speed_end - max_launch_speed_start) * curriculum_scale
    if randomize_curriculum_speed_for_debug:
        current_max_launch_speed = float(
            math_utils.sample_uniform(
                max_launch_speed_start,
                max_launch_speed_end,
                (1,),
                device=object_asset.device,
            )[0].item()
        )

    displacement = target_pos - launch_pos
    distance = torch.norm(displacement, dim=-1).clamp_min(1.0e-6)
    min_time_to_impact = distance / max(current_max_launch_speed, 1.0e-6)
    time_to_impact = torch.maximum(sampled_time_to_impact, min_time_to_impact)

    # Ballistic velocity with gravity compensation on z-axis.
    gravity = 9.81
    desired_lin_vel = displacement / time_to_impact.unsqueeze(-1)
    desired_lin_vel[:, 2] = desired_lin_vel[:, 2] + 0.5 * gravity * time_to_impact

    ang_vel = math_utils.sample_uniform(-2.0, 2.0, (len(env_ids), 3), device=object_asset.device)
    launch_vel = torch.cat([desired_lin_vel, ang_vel], dim=-1)

    object_asset.write_root_pose_to_sim(torch.cat([launch_pos, launch_rot], dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(launch_vel, env_ids=env_ids)

    # if debug_print_world_z:
    #     z_vals = launch_pos[:, 2]
    #     launch_speed = torch.norm(desired_lin_vel, dim=-1)
    #     below_min = int((z_vals < 1.0).sum().item())
    #     print(
    #         "[DODGEBALL_LAUNCH_DEBUG] "
    #         f"num_envs={len(env_ids)} "
    #         f"world_z_min={float(z_vals.min().item()):.4f} "
    #         f"world_z_max={float(z_vals.max().item()):.4f} "
    #         f"below_1p0_count={below_min} "
    #         f"speed_max_limit={current_max_launch_speed:.3f} "
    #         f"sampled_speed_min={float(launch_speed.min().item()):.3f} "
    #         f"sampled_speed_max={float(launch_speed.max().item()):.3f}"
    #     )

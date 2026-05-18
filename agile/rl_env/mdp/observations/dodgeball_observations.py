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

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse


def _ball_relative_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> tuple[torch.Tensor, torch.Tensor]:
    robot: Articulation = env.scene[asset_cfg.name]
    dodgeball: RigidObject = env.scene[object_cfg.name]

    rel_pos_w = dodgeball.data.root_pos_w - robot.data.root_pos_w
    rel_vel_w = dodgeball.data.root_lin_vel_w - robot.data.root_lin_vel_w

    rel_pos_b = quat_apply_inverse(robot.data.root_quat_w, rel_pos_w)
    rel_vel_b = quat_apply_inverse(robot.data.root_quat_w, rel_vel_w)
    return rel_pos_b, rel_vel_b


def ball_pos_rel_root(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> torch.Tensor:
    """Return dodgeball position relative to robot root frame."""
    rel_pos_b, _ = _ball_relative_state(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    return rel_pos_b


def ball_vel_rel_root(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> torch.Tensor:
    """Return dodgeball linear velocity relative to robot root frame."""
    _, rel_vel_b = _ball_relative_state(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    return rel_vel_b


def ball_time_to_impact(
    env: ManagerBasedRLEnv,
    safe_distance: float = 0.6,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> torch.Tensor:
    """Estimate time-to-impact with the safety shell around the robot."""
    rel_pos_b, rel_vel_b = _ball_relative_state(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    distance = torch.norm(rel_pos_b, dim=-1).clamp_min(1.0e-6)
    radial_speed = -(rel_pos_b * rel_vel_b).sum(dim=-1) / distance
    approaching_speed = torch.clamp(radial_speed, min=0.0)
    distance_to_shell = torch.clamp(distance - safe_distance, min=0.0)
    tti = distance_to_shell / (approaching_speed + 1.0e-6)
    # Keep finite values with bounded dynamic range for stable learning.
    return torch.clamp(tti, 0.0, 10.0).unsqueeze(-1)

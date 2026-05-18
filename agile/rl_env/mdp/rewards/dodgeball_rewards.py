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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from agile.rl_env.mdp.observations.dodgeball_observations import ball_pos_rel_root, ball_vel_rel_root


def dodgeball_survival_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant alive bonus to favor surviving longer under incoming throws."""
    return torch.ones(env.num_envs, device=env.device)


def ball_clearance_reward(
    env: ManagerBasedRLEnv,
    safe_distance: float = 0.7,
    distance_std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> torch.Tensor:
    """Reward maintaining clearance from the incoming dodgeball."""
    rel_pos_b = ball_pos_rel_root(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    distance = torch.norm(rel_pos_b, dim=-1)
    distance_margin = torch.clamp(distance - safe_distance, min=0.0)
    return 1.0 - torch.exp(-torch.square(distance_margin) / (distance_std * distance_std))


def ball_closing_speed_penalty(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> torch.Tensor:
    """Penalty for high incoming radial speed of the dodgeball.

    Radial speed is the component of relative velocity along the line connecting robot and ball.
    Positive incoming radial speed means the ball is moving toward the robot center.
    """
    rel_pos_b = ball_pos_rel_root(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    rel_vel_b = ball_vel_rel_root(env, asset_cfg=asset_cfg, object_cfg=object_cfg)
    distance = torch.norm(rel_pos_b, dim=-1).clamp_min(1.0e-6)
    radial_speed = -(rel_pos_b * rel_vel_b).sum(dim=-1) / distance
    approaching_speed = torch.clamp(radial_speed - speed_threshold, min=0.0)
    return approaching_speed


def ball_robot_contact_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball_robot_contact"),
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Binary penalty when the dodgeball contacts robot above threshold force.

    This sensor is attached to the ball and filtered to robot prim paths, so
    ground contacts are excluded.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        force_norm = torch.norm(net_contact_forces, dim=-1)  # [N, history, bodies]
    elif isinstance(body_ids, slice):
        force_norm = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)
    else:
        if len(body_ids) == 0:
            force_norm = torch.norm(net_contact_forces, dim=-1)
        else:
            force_norm = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)

    max_force_over_history = torch.max(force_norm, dim=1)[0]
    if max_force_over_history.ndim == 1:
        return (max_force_over_history > force_threshold).float()
    return torch.any(max_force_over_history > force_threshold, dim=1).float()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def dodgeball_robot_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball_robot_contact"),
    force_threshold: float = 2.0,
    fallback_distance_threshold: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
) -> torch.Tensor:
    """Detect ball-to-robot contact while rejecting ground-only ball contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    filtered_hit, filtered_max = _filtered_force_hit(contact_sensor, force_threshold)
    net_hit = _net_force_hit(contact_sensor, force_threshold)
    close_to_robot = _ball_close_to_robot(env, asset_cfg, object_cfg, fallback_distance_threshold)

    fallback_hit = net_hit & close_to_robot
    fallback_needed = filtered_max is None or torch.all(filtered_max <= 1.0e-6)
    if fallback_needed and bool(torch.any(net_hit).item()):
        _warn_contact_fallback_once(
            env,
            "missing_or_zero_filtered_matrix",
            "[DodgeballContact] Filtered ball-robot force matrix is missing or zero while "
            "net ball contact is nonzero; using robot-proximity-gated fallback.",
        )

    if filtered_hit is None:
        return fallback_hit
    return filtered_hit | ((filtered_max <= 1.0e-6) & fallback_hit)


def _filtered_force_hit(
    contact_sensor: ContactSensor,
    force_threshold: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    maxima: list[torch.Tensor] = []

    fmat_history = getattr(contact_sensor.data, "force_matrix_w_history", None)
    if fmat_history is not None and fmat_history.numel() > 0:
        maxima.append(torch.norm(fmat_history, dim=-1).amax(dim=(1, 2, 3)))

    fmat = getattr(contact_sensor.data, "force_matrix_w", None)
    if fmat is not None and fmat.numel() > 0:
        maxima.append(torch.norm(fmat, dim=-1).amax(dim=(1, 2)))

    if not maxima:
        return None, None

    filtered_max = torch.stack(maxima, dim=0).amax(dim=0)
    return filtered_max > force_threshold, filtered_max


def _net_force_hit(contact_sensor: ContactSensor, force_threshold: float) -> torch.Tensor:
    maxima: list[torch.Tensor] = []

    net_history = getattr(contact_sensor.data, "net_forces_w_history", None)
    if net_history is not None and net_history.numel() > 0:
        maxima.append(torch.norm(net_history, dim=-1).amax(dim=(1, 2)))

    net = getattr(contact_sensor.data, "net_forces_w", None)
    if net is not None and net.numel() > 0:
        maxima.append(torch.norm(net, dim=-1).amax(dim=1))

    if not maxima:
        return torch.zeros(contact_sensor._num_envs, dtype=torch.bool, device=contact_sensor._device)

    net_max = torch.stack(maxima, dim=0).amax(dim=0)
    return net_max > force_threshold


def _ball_close_to_robot(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    distance_threshold: float,
) -> torch.Tensor:
    dodgeball: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]

    ball_pos_w = dodgeball.data.root_pos_w
    body_ids = asset_cfg.body_ids
    if body_ids is None:
        body_pos_w = robot.data.body_pos_w
    elif isinstance(body_ids, slice):
        body_pos_w = robot.data.body_pos_w[:, body_ids, :]
    elif len(body_ids) == 0:
        body_pos_w = robot.data.body_pos_w
    else:
        body_pos_w = robot.data.body_pos_w[:, body_ids, :]

    distances = torch.norm(body_pos_w - ball_pos_w.unsqueeze(1), dim=-1)
    return torch.any(distances < distance_threshold, dim=1)


def _warn_contact_fallback_once(env: ManagerBasedRLEnv, key: str, message: str) -> None:
    warned = getattr(env, "_dodgeball_contact_warnings", None)
    if warned is None:
        warned = set()
        setattr(env, "_dodgeball_contact_warnings", warned)
    if key in warned:
        return
    warned.add(key)
    print(message)

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

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from agile.rl_env.mdp.observations.dodgeball_observations import ball_pos_rel_root, ball_vel_rel_root


# ── Support-polygon geometry helpers ─────────────────────────────────────────

def _quat_to_yaw(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Yaw (Z-rotation) from quaternion [..., 4] in (w, x, y, z) order."""
    w, x, y, z = quat_wxyz[..., 0], quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _foot_corners_2d(
    ankle_xy: torch.Tensor,  # [N, 2]
    foot_yaw: torch.Tensor,  # [N]
    half_len: float,
    half_width: float,
    toe_offset: float = 0.0,
) -> torch.Tensor:  # [N, 4, 2]
    """Rectangular foot-patch corners in world XY, rotated by the foot's yaw.

    The patch centre is shifted ``toe_offset`` metres forward (in foot-local +x)
    relative to the ankle link.  This accounts for the ankle being closer to the
    heel than to the geometric centre of the foot.

    Effective extent along foot-local x:
        heel side : -(half_len - toe_offset) from ankle
        toe  side : +(half_len + toe_offset) from ankle
    """
    cos_y = torch.cos(foot_yaw)  # [N]
    sin_y = torch.sin(foot_yaw)  # [N]
    # Foot-local corners shifted forward by toe_offset
    toe  = half_len + toe_offset
    heel = half_len - toe_offset
    local = torch.tensor(
        [[ toe,   half_width],
         [ toe,  -half_width],
         [-heel, -half_width],
         [-heel,  half_width]],
        device=ankle_xy.device, dtype=ankle_xy.dtype,
    )  # [4, 2]
    rot_x = cos_y.unsqueeze(-1) * local[:, 0] - sin_y.unsqueeze(-1) * local[:, 1]  # [N, 4]
    rot_y = sin_y.unsqueeze(-1) * local[:, 0] + cos_y.unsqueeze(-1) * local[:, 1]  # [N, 4]
    rotated = torch.stack([rot_x, rot_y], dim=-1)  # [N, 4, 2]
    return ankle_xy.unsqueeze(1) + rotated          # [N, 4, 2]


def _point_to_hull_dist_batch(
    com_xy: torch.Tensor,    # [N, 2]
    hull_pts: torch.Tensor,  # [N, K, 2]
) -> torch.Tensor:           # [N]  — 0 if inside, positive if outside
    """Distance from each query point to the convex hull of its K input points.

    Algorithm:
      1. Sort the K points by angle from their centroid → CCW convex-hull order.
      2. For each edge of the resulting polygon, compute the signed 2-D cross product.
         All non-negative → point is inside → distance = 0.
      3. Otherwise: distance = min distance to any edge of the polygon.
    """
    N, K, _ = hull_pts.shape
    # CCW vertex order by angle from centroid
    centroid = hull_pts.mean(dim=1, keepdim=True)              # [N, 1, 2]
    angles = torch.atan2(
        (hull_pts - centroid)[..., 1],
        (hull_pts - centroid)[..., 0],
    )                                                           # [N, K]
    order = angles.argsort(dim=1)                              # [N, K]
    pts = hull_pts.gather(1, order.unsqueeze(-1).expand(-1, -1, 2))  # [N, K, 2]

    a  = pts                                                    # [N, K, 2]
    b  = torch.roll(pts, -1, dims=1)                           # [N, K, 2]
    ab = b - a                                                  # [N, K, 2]
    p  = com_xy.unsqueeze(1)                                   # [N, 1, 2]
    ap = p - a                                                  # [N, K, 2]

    # Inside test: all 2-D cross products ≥ 0 for CCW polygon
    cross  = ab[..., 0] * ap[..., 1] - ab[..., 1] * ap[..., 0]  # [N, K]
    inside = (cross >= -1e-6).all(dim=1)                          # [N]

    # Closest point on each edge → min distance
    t = ((ap * ab).sum(dim=-1) / (ab * ab).sum(dim=-1).clamp_min(1e-8)).clamp(0.0, 1.0)
    closest  = a + t.unsqueeze(-1) * ab                           # [N, K, 2]
    min_dist = (p - closest).norm(dim=-1).min(dim=1).values       # [N]

    return torch.where(inside, torch.zeros_like(min_dist), min_dist)


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


def com_balance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),      # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    foot_half_len: float = 0.08,
    foot_half_width: float = 0.04,
    foot_toe_offset: float = 0.02,
    sigma: float = 0.1,
    force_threshold: float = 5.0,
    grace_steps: int = 10,
) -> torch.Tensor:
    """Reward the robot for keeping its CoM projection inside the support polygon.

    Each grounded foot is modelled as a rectangular contact patch centred on the
    ankle link, with dimensions ``foot_half_len × foot_half_width`` (half-extents),
    rotated by the foot's yaw angle.  The support polygon is the 2-D convex hull
    of all corners from all grounded foot patches:

    - Two feet grounded: convex hull of 8 corners (both patches + area between them).
    - One foot grounded: convex hull of 4 corners (that foot's patch alone).
    - No feet grounded: reward = 0.

    Reward kernel (HuB, Zhang et al., CoRL 2025):
        r = exp(-dist² / σ²)
    where ``dist`` is 0 when the CoM projection is inside the hull and is the
    minimum boundary distance otherwise.

    A ``grace_steps`` window after each reset returns 1.0 unconditionally, covering
    the ~7-step airborne settling phase before the robot establishes contacts.

    Args:
        env: The RL environment.
        asset_cfg: Robot articulation (all bodies, for CoM).
        sensor_cfg: Contact sensor scoped to ankle_roll_link bodies.
        foot_asset_cfg: Robot articulation scoped to ankle_roll_link bodies.
        foot_half_len: Half-length of the foot patch (m) along foot local x.
        foot_half_width: Half-width of the foot patch (m) along foot local y.
        foot_toe_offset: Forward shift (m) of the patch centre in foot local x.
            Use a positive value when the ankle link is closer to the heel.
            Effective extent: heel = half_len − offset, toe = half_len + offset.
        sigma: Exponential kernel width (m).  0.1 m following HuB.
        force_threshold: Minimum contact force (N) to treat a foot as grounded.
        grace_steps: Post-reset grace period (steps) returning reward = 1.0.

    Returns:
        Reward tensor shape [num_envs] in [0, 1].
    """
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Grace period — robot settles from z≈0.9 m, contacts appear after ~7 steps.
    in_grace = env.episode_length_buf < grace_steps  # [N]
    if in_grace.all():
        return torch.ones(env.num_envs, device=env.device)

    # ── 1. CoM XY projection ──────────────────────────────────────────────────
    # default_mass lives on CPU at load time; move to simulation device.
    masses = robot.data.default_mass.to(env.device)              # [N, B]
    total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-6) # [N, 1]
    body_pos_xy = robot.data.body_pos_w[:, :, :2]                # [N, B, 2]
    com_xy = (masses.unsqueeze(-1) * body_pos_xy).sum(dim=1) / total_mass  # [N, 2]

    # ── 2. Foot contact state ─────────────────────────────────────────────────
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N, 2, 3]
    foot_in_contact = torch.norm(foot_forces, dim=-1) > force_threshold         # [N, 2]
    n_contacts = foot_in_contact.sum(dim=1)                                      # [N]

    # ── 3. Foot patch corners in world XY ─────────────────────────────────────
    foot_pos_w  = robot.data.body_pos_w[:,  foot_asset_cfg.body_ids, :]   # [N, 2, 3]
    foot_quat_w = robot.data.body_quat_w[:, foot_asset_cfg.body_ids, :]   # [N, 2, 4]
    foot_xy  = foot_pos_w[..., :2]                                         # [N, 2, 2]
    # Yaw only — support polygon is on the XY ground plane.
    foot_yaw = _quat_to_yaw(foot_quat_w.reshape(-1, 4)).reshape(env.num_envs, 2)  # [N, 2]

    corners_L = _foot_corners_2d(foot_xy[:, 0], foot_yaw[:, 0], foot_half_len, foot_half_width, foot_toe_offset)  # [N, 4, 2]
    corners_R = _foot_corners_2d(foot_xy[:, 1], foot_yaw[:, 1], foot_half_len, foot_half_width, foot_toe_offset)  # [N, 4, 2]

    # ── 4. Distance from CoM to support hull ─────────────────────────────────
    # Three sub-cases, each with its own correctly-sized hull:
    #   both feet → 8-corner hull (full support polygon between both patches)
    #   left only → 4-corner hull (left foot patch)
    #   right only → 4-corner hull (right foot patch)
    # Results are combined without conditional branches so gradients stay intact.
    both_grounded = n_contacts >= 2
    only_L = foot_in_contact[:, 0] & ~foot_in_contact[:, 1]
    only_R = foot_in_contact[:, 1] & ~foot_in_contact[:, 0]

    all8 = torch.cat([corners_L, corners_R], dim=1)   # [N, 8, 2]
    dist_both = _point_to_hull_dist_batch(com_xy, all8)
    dist_L    = _point_to_hull_dist_batch(com_xy, corners_L)
    dist_R    = _point_to_hull_dist_batch(com_xy, corners_R)

    dist = torch.where(both_grounded, dist_both,
           torch.where(only_L,         dist_L,
           torch.where(only_R,         dist_R,
                                       dist_both)))   # n_contacts==0 masked below

    # ── 5. Reward ─────────────────────────────────────────────────────────────
    # dist == 0 when CoM projection is inside the hull → reward = 1.0.
    # Decays exponentially with distance outside the hull.
    reward = torch.exp(-torch.square(dist) / (sigma * sigma))
    reward = torch.where(n_contacts == 0, torch.zeros_like(reward), reward)
    reward = torch.where(in_grace,        torch.ones_like(reward),  reward)
    return reward


def foot_locomotion_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),        # noqa: B008
    force_threshold: float = 10.0,
    foot_vel_threshold: float = 0.15,
    both_moving_scale: float = 1.0,
    one_moving_scale: float = 0.2,
    drift_scale: float = 0.5,
    drift_clip: float = 2.0,
    grace_steps: int = 10,
) -> torch.Tensor:
    """Penalise locomotion/displacement without penalising valid balance footwork.

    Approach — each foot is classified each step as *planted* or *moving*:
    - **Planted**: in contact (normal force > ``force_threshold``) AND horizontal
      speed < ``foot_vel_threshold``.
    - **Moving**: everything else.

    Penalty contributions:
    1. **Both feet moving** (locomotion): ``both_moving_scale × mean_foot_speed``.
    2. **One foot moving** (corrective step): ``one_moving_scale × mean_foot_speed``.
    3. **Stance-centre drift**: slow drift of the mean foot XY position away from
       the episode-start anchor.  Catches gradual walking that alternates planted
       feet.  ``drift_scale × clamp(drift, 0, drift_clip)``.

    The penalty is zero during the ``grace_steps`` post-reset settling window.
    The stance anchor ``env._stance_anchor_w`` is initialised at the first step
    of each episode.

    Returns:
        Penalty tensor shape [num_envs], values ≥ 0.  Apply a negative weight
        in the ``RewTerm`` to convert to a negative reward.
    """
    robot: Articulation = env.scene[foot_asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Grace period — contacts not yet established.
    in_grace = env.episode_length_buf < grace_steps  # [N]
    if in_grace.all():
        return torch.zeros(env.num_envs, device=env.device)

    # ── Foot positions and velocities ─────────────────────────────────────────
    foot_pos_w  = robot.data.body_pos_w[:, foot_asset_cfg.body_ids, :]  # [N, F, 3]
    foot_vel_w  = robot.data.body_vel_w[:, foot_asset_cfg.body_ids, :3] # [N, F, 3]
    foot_xy     = foot_pos_w[..., :2]                                    # [N, F, 2]
    mean_foot_xy = foot_xy.mean(dim=1)                                   # [N, 2]

    # ── Stance anchor: reset at first step of each episode ────────────────────
    if not hasattr(env, "_stance_anchor_w") or env._stance_anchor_w.shape[0] != env.num_envs:
        env._stance_anchor_w = mean_foot_xy.clone()

    just_reset = env.episode_length_buf == 1  # [N]
    if just_reset.any():
        env._stance_anchor_w = env._stance_anchor_w.clone()
        env._stance_anchor_w[just_reset] = mean_foot_xy[just_reset]

    # ── Planted / moving classification ───────────────────────────────────────
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N, F, 3]
    in_contact  = torch.norm(foot_forces, dim=-1) > force_threshold             # [N, F]

    # Horizontal (XY) foot speed.
    foot_speed_h = torch.norm(foot_vel_w[..., :2], dim=-1)   # [N, F]
    is_slow      = foot_speed_h < foot_vel_threshold          # [N, F]

    planted = in_contact & is_slow    # [N, F]
    moving  = ~planted                # [N, F]
    n_moving = moving.sum(dim=1)      # [N]   0 / 1 / 2

    mean_speed = foot_speed_h.mean(dim=1)  # [N]

    both_moving_mask = n_moving >= 2   # [N]
    one_moving_mask  = n_moving == 1   # [N]

    locomotion_penalty = torch.where(
        both_moving_mask,
        both_moving_scale * mean_speed,
        torch.where(one_moving_mask, one_moving_scale * mean_speed, torch.zeros_like(mean_speed)),
    )

    # ── Stance-centre drift penalty ────────────────────────────────────────────
    drift = torch.norm(mean_foot_xy - env._stance_anchor_w, dim=-1)  # [N]
    drift_penalty = drift_scale * drift.clamp(max=drift_clip)

    # ── Combine, zero out grace steps ─────────────────────────────────────────
    penalty = locomotion_penalty + drift_penalty
    penalty = torch.where(in_grace, torch.zeros_like(penalty), penalty)
    return penalty

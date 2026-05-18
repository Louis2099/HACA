# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import subtract_frame_transforms

from agile.rl_env.mdp.utils import get_robot_cfg


def ground_slam(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    velocity_threshold: float = 2.0,
    force_threshold: float = 10.0,
    prerequisite_curriculum: str | None = None,
    prerequisite_threshold: float = 0.01,
    prerequisite_direction: str = "below",
) -> torch.Tensor:
    """Terminate if any monitored body slams the ground with high impact velocity.

    Uses the contact sensor's velocity history buffer to detect impact velocity at contact onset.
    Only activates after an optional prerequisite curriculum reaches a threshold, so it doesn't
    interfere with early training (e.g., while the robot is still learning to stand up).

    Args:
        env: The environment instance.
        sensor_cfg: Contact sensor config with body_names specifying which bodies to monitor.
        velocity_threshold: Maximum allowed impact velocity (m/s).
        force_threshold: Minimum contact force (N) to consider a body as "in contact".
        prerequisite_curriculum: If set, termination is only active when this curriculum
            value crosses the threshold (e.g., "adaptive_lift" must reach ~0 before enabling).
        prerequisite_threshold: Threshold value for the prerequisite curriculum.
        prerequisite_direction: "below" = active when curriculum < threshold,
            "above" = active when curriculum > threshold.
    """
    # Check prerequisite curriculum gate
    if prerequisite_curriculum is not None:
        prereq_value = env.curriculum_manager._curriculum_state.get(prerequisite_curriculum)
        if prereq_value is None:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if prerequisite_direction == "below" and prereq_value >= prerequisite_threshold:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if prerequisite_direction == "above" and prereq_value < prerequisite_threshold:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history
    in_contact = (
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > force_threshold
    )

    body_velocities = torch.max(
        torch.norm(contact_sensor.data.velocities_w_history[:, :, sensor_cfg.body_ids], dim=-1),
        dim=1,
    )[0]

    # Terminate if ANY monitored body is in contact AND exceeds velocity threshold
    slam_detected = in_contact & (body_velocities > velocity_threshold)
    return torch.any(slam_detected, dim=1)


def illegal_ground_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    min_height: float,
) -> torch.Tensor:
    """Terminate when the contact force exceeds the force threshold and the asset is below the min_height."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold

    asset_height = torch.min(env.scene[asset_cfg.name].data.body_pos_w[:, asset_cfg.body_ids, 2], dim=1)[0]
    on_ground = asset_height < min_height

    in_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )

    return in_contact & on_ground


def illegal_base_height(
    env: ManagerBasedRLEnv,
    height_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_measurement_sensor"),
) -> torch.Tensor:
    """Terminate if the base height is below the threshold."""
    robot, _ = get_robot_cfg(env, asset_cfg)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    base_height = robot.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    return base_height < height_threshold


class fall_from_max_height(ManagerTermBase):
    """Terminate if the robot falls significantly below its maximum achieved height.

    This termination tracks the highest point the robot has reached (clamped at a maximum
    trackable height to ignore jumping) and terminates when the robot falls more than
    a threshold below that peak. This is more adaptive than fixed height thresholds
    as it's relative to the robot's progress.

    Example:
        - Robot reaches 0.6m -> terminates if it falls below 0.4m (with 0.2m fall_threshold)
        - Robot reaches 0.8m -> terminates if it falls below 0.6m (with 0.2m fall_threshold)
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Track max height achieved per environment (start at 0)
        self.max_height_achieved = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        fall_threshold: float,
        max_trackable_height: float,
        min_height_to_track: float = 0.0,
    ) -> torch.Tensor:
        """Terminate if the robot falls significantly below its max achieved height.

        Args:
            env: The environment instance.
            asset_cfg: Configuration for the robot asset.
            sensor_cfg: Configuration for the height sensor.
            fall_threshold: How far below max height triggers termination (e.g., 0.2m).
            max_trackable_height: Upper bound for height tracking (e.g., standing height).
                Heights above this are clamped so jumping doesn't inflate the max.
            min_height_to_track: Minimum height before tracking begins. Heights below
                this won't update max_height_achieved. Useful to ignore initial fallen poses.

        Returns:
            Boolean tensor indicating which environments should terminate.
        """
        robot, _ = get_robot_cfg(env, asset_cfg)
        sensor: RayCaster = env.scene[sensor_cfg.name]
        base_height = robot.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)

        # Only track heights above the minimum threshold
        trackable_height = torch.where(
            base_height > min_height_to_track,
            base_height,
            self.max_height_achieved,  # Keep previous max if below tracking threshold
        )

        # Clamp at max trackable height (so jumping doesn't count)
        trackable_height = torch.clamp(trackable_height, max=max_trackable_height)

        # Update max height achieved
        self.max_height_achieved = torch.maximum(self.max_height_achieved, trackable_height)

        # Terminate if current height is more than fall_threshold below max achieved
        return base_height < (self.max_height_achieved - fall_threshold)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)
        self.max_height_achieved[env_ids] = 0.0


class no_height_progress(ManagerTermBase):
    """Terminate if the robot hasn't made upward progress within a time window.

    This termination encourages the robot to attempt standing up without punishing
    it for falling after reaching standing height. It tracks the initial height
    at episode start and terminates if the robot hasn't increased its height
    above that initial height + threshold within N seconds.

    This is more forgiving than fall_from_max_height because:
    - It doesn't punish the robot for attempting to stand and then falling
    - It only terminates robots that aren't making any upward progress
    - Once the robot has made progress, the termination is satisfied
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Track initial height at episode start
        self.initial_height = torch.zeros(env.num_envs, device=env.device)
        # Track whether robot has made progress (reached threshold)
        self.made_progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        height_increase_threshold: float,
        time_limit_s: float,
    ) -> torch.Tensor:
        """Terminate if no height progress within time limit.

        Args:
            env: The environment instance.
            asset_cfg: Configuration for the robot asset.
            sensor_cfg: Configuration for the height sensor.
            height_increase_threshold: Required height increase above initial to count as progress.
            time_limit_s: Time in seconds within which progress must be made.

        Returns:
            Boolean tensor indicating which environments should terminate.
        """
        robot, _ = get_robot_cfg(env, asset_cfg)
        sensor: RayCaster = env.scene[sensor_cfg.name]
        current_height = robot.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)

        # Check if robot has made progress (current height > initial + threshold)
        self.made_progress |= current_height > (self.initial_height + height_increase_threshold)

        # Calculate time elapsed
        time_elapsed = env.episode_length_buf * env.step_dt

        # Terminate if time limit exceeded AND no progress made
        return (time_elapsed > time_limit_s) & ~self.made_progress

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)

        # Capture initial height for resetting environments
        robot, _ = get_robot_cfg(self._env, self.cfg.params["asset_cfg"])
        sensor: RayCaster = self._env.scene[self.cfg.params["sensor_cfg"].name]
        current_height = robot.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)

        self.initial_height[env_ids] = current_height[env_ids]
        self.made_progress[env_ids] = False


def link_distance(
    env: ManagerBasedRLEnv,
    min_distance_threshold: float = 0.05,
    max_distance_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Terminate if the distance between the two links is outside the allowed range.

    Args:
        env: Environment instance
        min_distance_threshold: Minimum distance threshold. Terminate if links closer than this.
        max_distance_threshold: Maximum distance threshold. Terminate if links farther than this. None to disable.
        asset_cfg: Asset configuration (must specify exactly 2 links)

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    robot, _ = get_robot_cfg(env, asset_cfg)
    link_pos = robot.data.body_pos_w[:, asset_cfg.body_ids]

    if len(asset_cfg.body_ids) != 2:
        raise ValueError("Link distance is only supported for 2 links")
    link_distance = torch.norm(link_pos[:, 0] - link_pos[:, 1], dim=1)

    # Check minimum distance
    too_close = link_distance < min_distance_threshold

    # Check maximum distance if specified
    if max_distance_threshold is not None:
        too_far = link_distance > max_distance_threshold
        return too_close | too_far

    return too_close


def illegal_link_distance_lateral(
    env: ManagerBasedRLEnv,
    min_distance_threshold: float = 0.05,
    max_distance_threshold: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
) -> torch.Tensor:
    """Terminate if the lateral (Y-axis in root frame) distance between two links is outside range.

    Projects the link positions into the root body frame and checks the Y-axis separation.
    This allows large forward/backward steps while constraining lateral splay.

    Args:
        env: Environment instance
        min_distance_threshold: Minimum lateral distance. Terminate if closer than this.
        max_distance_threshold: Maximum lateral distance. Terminate if farther than this. None to disable.
        asset_cfg: Asset configuration (must specify exactly 2 links)

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    robot, _ = get_robot_cfg(env, asset_cfg)
    link_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids]

    if len(asset_cfg.body_ids) != 2:
        raise ValueError("Link distance is only supported for 2 links")

    # Transform link positions into root frame
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    link0_b, _ = subtract_frame_transforms(root_pos, root_quat, link_pos_w[:, 0])
    link1_b, _ = subtract_frame_transforms(root_pos, root_quat, link_pos_w[:, 1])

    # Lateral distance = Y-axis separation in root frame
    lateral_dist = torch.abs(link0_b[:, 1] - link1_b[:, 1])

    too_close = lateral_dist < min_distance_threshold

    if max_distance_threshold is not None:
        too_far = lateral_dist > max_distance_threshold
        return too_close | too_far

    return too_close


class standing(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.standing_timer = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        min_height: float,
        duration_s: float,
        sensor_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        """Terminate if the robot stands for a given time."""

        asset: RigidObject = env.scene[asset_cfg.name]
        if sensor_cfg is not None:
            sensor: RayCaster = env.scene[sensor_cfg.name]
            # Adjust the target height using the sensor data
            current_height = asset.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
        else:
            # Use the provided target height directly for flat terrain
            current_height = asset.data.root_pos_w[:, 2]

        is_standing = current_height > min_height

        self.standing_timer += 1
        self.standing_timer[~is_standing] = 0

        return self.standing_timer > int(duration_s / env.step_dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs)
        self.standing_timer[env_ids] = 0


def bad_base_pose(
    env: ManagerBasedRLEnv,
    base_pos_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the anchor position error exceeds a threshold.

    Works with any command term that exposes ``command_anchor_pos_w`` and
    ``robot_anchor_pos_w`` (e.g. ``TrackingCommand``, ``MotionCommand``).
    """
    command = env.command_manager.get_term(command_name)
    base_pos_error = torch.norm(command.command_anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
    return base_pos_error > base_pos_threshold


def bad_base_rotation(
    env: ManagerBasedRLEnv,
    base_ori_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the anchor orientation error exceeds a threshold.

    Works with any command term that exposes ``command_anchor_quat_w`` and
    ``robot_anchor_quat_w`` (e.g. ``TrackingCommand``, ``MotionCommand``).
    """
    command = env.command_manager.get_term(command_name)
    base_ori_error = math_utils.quat_error_magnitude(command.command_anchor_quat_w, command.robot_anchor_quat_w)
    return base_ori_error > base_ori_threshold


def bad_joint_pos(
    env: ManagerBasedRLEnv,
    joint_pos_threshold: float,
    command_name: str,
) -> torch.Tensor:
    """Terminate when the robot is away from the trajectory."""
    command = env.command_manager.get_term(command_name)
    joint_pos_error = torch.norm(command.command_tracked_joint_pos - command.robot_tracked_joint_pos, dim=-1)
    return joint_pos_error > joint_pos_threshold


##
# Whole-body motion tracking terminations
##


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Terminate if the anchor Z position error exceeds the threshold.

    Works with any command term that exposes ``command_anchor_pos_w`` and
    ``robot_anchor_pos_w`` (e.g. ``TrackingCommand``, ``MotionCommand``).
    """
    command = env.command_manager.get_term(command_name)
    return torch.abs(command.command_anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Terminate if the anchor orientation error exceeds the threshold.

    Uses projected-gravity comparison, which is different from ``bad_base_rotation``
    (that uses ``quat_error_magnitude``).

    Works with any command term that exposes ``command_anchor_quat_w`` and
    ``robot_anchor_quat_w`` (e.g. ``TrackingCommand``, ``MotionCommand``).
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.command_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if any body position error exceeds the threshold."""
    from agile.rl_env.mdp.commands.motion_tracking_commands import MotionCommand
    from agile.rl_env.mdp.rewards.motion_tracking_rewards import _get_body_indices

    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indices = _get_body_indices(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate if any body Z position error exceeds the threshold."""
    from agile.rl_env.mdp.commands.motion_tracking_commands import MotionCommand
    from agile.rl_env.mdp.rewards.motion_tracking_rewards import _get_body_indices

    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indices = _get_body_indices(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indices, -1] - command.robot_body_pos_w[:, body_indices, -1])
    return torch.any(error > threshold, dim=-1)


def invalid_state(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    max_joint_vel: float = 100.0,
    max_root_height: float = 10.0,
    max_root_xy_distance: float = 200.0,
    max_lin_vel: float = 50.0,
    max_ang_vel: float = 100.0,
) -> torch.Tensor:
    """Terminate when physics values explode (NaN or exceeding thresholds).

    This termination detects simulation instabilities before they cause NaN propagation.
    It checks for NaN states, joint velocities, root position, and root velocities.

    Args:
        env: The environment instance.
        asset_cfg: Configuration for the robot asset.
        max_joint_vel: Maximum allowed joint velocity magnitude (rad/s). Default: 100.0
        max_root_height: Maximum allowed root height above env origin (m). Default: 10.0
        max_root_xy_distance: Maximum allowed XY distance from env origin (m). Default: 50.0
        max_lin_vel: Maximum allowed linear velocity magnitude (m/s). Default: 50.0
        max_ang_vel: Maximum allowed angular velocity magnitude (rad/s). Default: 100.0

    Returns:
        Boolean tensor indicating which environments have invalid states.
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # Check for NaN in any state
    has_nan = (
        torch.isnan(robot.data.joint_pos).any(dim=-1)
        | torch.isnan(robot.data.joint_vel).any(dim=-1)
        | torch.isnan(robot.data.root_pos_w).any(dim=-1)
        | torch.isnan(robot.data.root_lin_vel_w).any(dim=-1)
        | torch.isnan(robot.data.root_ang_vel_w).any(dim=-1)
    )

    # Check joint velocities
    joint_vel_exceeded = torch.abs(robot.data.joint_vel).max(dim=-1).values > max_joint_vel

    # Check root position (relative to env origin)
    root_pos_rel = robot.data.root_pos_w - env.scene.env_origins
    root_height_exceeded = root_pos_rel[:, 2] > max_root_height
    root_xy_exceeded = torch.norm(root_pos_rel[:, :2], dim=-1) > max_root_xy_distance

    # Check linear velocity
    lin_vel_exceeded = torch.norm(robot.data.root_lin_vel_w, dim=-1) > max_lin_vel

    # Check angular velocity
    ang_vel_exceeded = torch.norm(robot.data.root_ang_vel_w, dim=-1) > max_ang_vel

    # Combine all checks
    return has_nan | joint_vel_exceeded | root_height_exceeded | root_xy_exceeded | lin_vel_exceeded | ang_vel_exceeded


def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] | None = None,
    reference_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Termination condition for the object falls out of bound.

    Args:
        env: The environment.
        asset_cfg: The object configuration. Defaults to SceneEntityCfg("object").
        in_bound_range: The range in x, y, z such that the object is considered in range.
        reference_asset_cfg: Optional reference asset configuration with body_names specified.
            If provided, the body_names must resolve to exactly one body to use as reference frame.

    Raises:
        ValueError: If reference_asset_cfg is provided but body_names is not specified or resolves to multiple bodies.
    """
    object: RigidObject = env.scene[asset_cfg.name]
    if in_bound_range is None:
        in_bound_range = {}
    # Default to (-inf, inf) for unspecified dimensions - always in bounds
    range_list = [in_bound_range.get(key, (float("-inf"), float("inf"))) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)

    # Get object position in world frame
    object_pos_w = object.data.root_pos_w

    # Transform object position to reference frame
    if reference_asset_cfg is not None:
        # Use asset link as reference frame
        reference_asset: Articulation = env.scene[reference_asset_cfg.name]
        # Get the body IDs from the reference_asset_cfg
        body_ids = reference_asset_cfg.body_ids
        if body_ids is None or len(body_ids) == 0:
            raise ValueError(
                f"reference_asset_cfg must have body_names specified. "
                f"Please provide body_names in SceneEntityCfg for asset '{reference_asset_cfg.name}'."
            )
        if len(body_ids) != 1:
            raise ValueError(
                f"reference_asset_cfg must resolve to exactly one body, but got {len(body_ids)} bodies: {body_ids}. "
                f"Please specify a single body name in reference_asset_cfg."
            )
        body_id = body_ids[0]
        # Get the reference link pose in world frame
        reference_pos_w = reference_asset.data.body_pos_w[:, body_id, :]
        reference_quat_w = reference_asset.data.body_quat_w[:, body_id, :]
        # Transform object position from world frame to reference link frame
        object_pos_local, _ = subtract_frame_transforms(reference_pos_w, reference_quat_w, object_pos_w)
    else:
        # Use environment origins as reference frame (no rotation)
        object_pos_local = object_pos_w - env.scene.env_origins

    # Check if object is outside bounds
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def ball_hit_protected_body(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.28,
) -> torch.Tensor:
    """Terminate if the dodgeball gets too close to protected robot bodies."""
    dodgeball: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[asset_cfg.name]

    ball_pos_w = dodgeball.data.root_pos_w
    if asset_cfg.body_ids is None or len(asset_cfg.body_ids) == 0:
        protected_pos_w = robot.data.root_pos_w.unsqueeze(1)
    else:
        protected_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :]

    distances = torch.norm(protected_pos_w - ball_pos_w.unsqueeze(1), dim=-1)
    return torch.any(distances < distance_threshold, dim=1)


def ball_contact_protected_body(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball_robot_contact"),
    force_threshold: float = 2.0,
) -> torch.Tensor:
    """Terminate if dodgeball receives robot contact force above threshold.

    The contact sensor is attached to the ball and filtered to robot prims, so this
    condition is robust to long-link geometry and avoids ground-contact false positives.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history

    # Ball has a single rigid body in this setup; keep generic indexing fallback.
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        body_force_norm = torch.norm(net_contact_forces, dim=-1)  # [N, history, bodies]
    elif isinstance(body_ids, slice):
        body_force_norm = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)
    else:
        if len(body_ids) == 0:
            body_force_norm = torch.norm(net_contact_forces, dim=-1)
        else:
            body_force_norm = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)

    max_force_over_history = torch.max(body_force_norm, dim=1)[0]
    if max_force_over_history.ndim == 1:
        return max_force_over_history > force_threshold
    return torch.any(max_force_over_history > force_threshold, dim=1)


def ball_passed_humanoid(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("dodgeball"),
    reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["torso_link"]),
    pass_x_threshold: float = -0.15,
) -> torch.Tensor:
    """Terminate if the dodgeball has passed the humanoid in torso local x-axis."""
    dodgeball: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[reference_asset_cfg.name]

    body_ids = reference_asset_cfg.body_ids
    if body_ids is None or len(body_ids) == 0:
        ref_pos_w = robot.data.root_pos_w
        ref_quat_w = robot.data.root_quat_w
    else:
        ref_body_id = int(body_ids[0])
        ref_pos_w = robot.data.body_pos_w[:, ref_body_id, :]
        ref_quat_w = robot.data.body_quat_w[:, ref_body_id, :]

    ball_pos_local, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, dodgeball.data.root_pos_w)
    return ball_pos_local[:, 0] < pass_x_threshold


def com_outside_support_polygon(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),  # noqa: B008
    foot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # noqa: B008
    foot_width: float = 0.07,
    single_foot_margin: float = 0.05,
    double_foot_margin: float = 0.15,
    force_threshold: float = 5.0,
    terminate_on_no_contact: bool = True,
) -> torch.Tensor:
    """Terminate if the robot's center of mass projects outside its support polygon.

    This is the physically necessary and sufficient condition for quasi-static balance.
    It replaces orientation-based terminations, allowing extreme maneuvers (e.g. swallow
    balance with a horizontal torso) as long as the CoM remains above the support area.

    The support polygon is approximated as:
    - Two feet in contact: nearest point on the line segment between the two ankle positions.
    - One foot in contact: the ankle position itself (support degenerates to a point).
    - No contact: treated as outside (terminates if ``terminate_on_no_contact`` is True).

    An effective foot-patch radius (``foot_width``) is subtracted so the CoM is allowed
    to be up to ``foot_width`` beyond the ankle-centre before the margin kicks in.

    Args:
        env: The environment instance.
        asset_cfg: Robot asset config with no body_names restriction — used to access all
            body masses and positions for CoM computation.
        sensor_cfg: Contact sensor config filtered to the foot bodies (ankle_roll_link).
        foot_asset_cfg: Robot asset config with body_names matching the feet, used to
            obtain world-frame foot XY positions.
        foot_width: Effective contact-patch half-width (m).  Distance from CoM to the
            support polygon is reduced by this before comparing against the margin.
        single_foot_margin: Extra distance (m) tolerated when only one foot is in contact.
        double_foot_margin: Extra distance (m) tolerated when both feet are in contact.
        force_threshold: Minimum contact force (N) to consider a foot as grounded.
        terminate_on_no_contact: If True, terminate whenever no foot is in contact.

    Returns:
        Boolean tensor [num_envs] — True for environments that should terminate.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # --- 1. Compute centre-of-mass XY projection ---
    # default_mass is populated at load time and lives on CPU; move it to the
    # simulation device so it can be multiplied with body_pos_w (on CUDA).
    masses = robot.data.default_mass.to(env.device)  # [N, B]
    total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-6)  # [N, 1]
    body_pos_xy = robot.data.body_pos_w[:, :, :2]  # [N, B, 2]
    com_xy = (masses.unsqueeze(-1) * body_pos_xy).sum(dim=1) / total_mass  # [N, 2]

    # --- 2. Foot contact state ---
    foot_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]  # [N, 2, 3]
    foot_in_contact = torch.norm(foot_forces, dim=-1) > force_threshold  # [N, 2]
    n_contacts = foot_in_contact.sum(dim=1)  # [N]

    # --- 3. Foot positions (XY) ---
    foot_pos_xy = robot.data.body_pos_w[:, foot_asset_cfg.body_ids, :2]  # [N, 2, 2]
    A = foot_pos_xy[:, 0]  # left ankle  [N, 2]
    B = foot_pos_xy[:, 1]  # right ankle [N, 2]

    # Distance to line-segment AB (two-foot support polygon approximation)
    AB = B - A  # [N, 2]
    AP = com_xy - A  # [N, 2]
    ab_sq = (AB * AB).sum(dim=-1).clamp_min(1e-6)  # [N]
    t = torch.clamp((AP * AB).sum(dim=-1) / ab_sq, 0.0, 1.0)  # [N]
    closest_on_segment = A + t.unsqueeze(-1) * AB  # [N, 2]
    dist_two_foot = torch.norm(com_xy - closest_on_segment, dim=-1)  # [N]

    # Distance to individual foot positions (single-foot support)
    dist_left = torch.norm(com_xy - A, dim=-1)   # [N]
    dist_right = torch.norm(com_xy - B, dim=-1)  # [N]

    # --- 4. Select distance and margin based on contact state ---
    left_contact = foot_in_contact[:, 0]
    right_contact = foot_in_contact[:, 1]
    both_contact = n_contacts >= 2

    # Raw distance to support polygon
    dist = torch.where(
        both_contact,
        dist_two_foot,
        torch.where(left_contact, dist_left, dist_right),
    )

    # Subtract effective foot-patch radius to get the distance *beyond* the polygon
    dist_beyond = torch.clamp(dist - foot_width, min=0.0)

    # Allowed slack depends on number of feet in contact
    margin = torch.where(both_contact, double_foot_margin, single_foot_margin)

    outside = dist_beyond > margin

    if terminate_on_no_contact:
        return outside | (n_contacts == 0)
    return outside

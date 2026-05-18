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

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from agile.rl_env import mdp
from agile.rl_env.tasks.locomotion_height.g1.velocity_height_env_cfg import (
    G1LowerVelocityHeightEnvCfg,
    LocomotionEventCfg,
    MySceneCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class DodgeballSceneCfg(MySceneCfg):
    """Locomotion-height scene extended with an incoming dodgeball object."""

    dodgeball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Dodgeball",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
            activate_contact_sensors=True,
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=200.0,
                max_linear_velocity=200.0,
                max_depenetration_velocity=10.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.15, 0.15), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[2.0, 0.0, 1.0], rot=[1.0, 0.0, 0.0, 0.0]),
    )

    # Ball-centric contact sensor filtered to robot collisions.
    dodgeball_robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Dodgeball",
        history_length=3,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
        debug_vis=False,
    )


@configclass
class DodgeballPolicyObsCfg(ObsGroup):
    """Command-free policy observations for in-place dodgeball behavior."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
    projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.01, n_max=0.01))
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), params={"asset_cfg": SceneEntityCfg("robot")})
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        noise=Unoise(n_min=-1.5, n_max=1.5),
        scale=0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
    ball_pos_rel_root = ObsTerm(func=mdp.ball_pos_rel_root)
    ball_vel_rel_root = ObsTerm(func=mdp.ball_vel_rel_root)
    ball_time_to_impact = ObsTerm(func=mdp.ball_time_to_impact, params={"safe_distance": 0.6})

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = False


@configclass
class DodgeballCriticObsCfg(ObsGroup):
    """Command-free critic observations for in-place dodgeball behavior."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
    projected_gravity = ObsTerm(func=mdp.projected_gravity)
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
    actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
    height_scan_feet = ObsTerm(
        func=mdp.height_scan_feet,
        params={
            "sensor_cfg_left": SceneEntityCfg("height_scanner_left_foot"),
            "sensor_cfg_right": SceneEntityCfg("height_scanner_right_foot"),
        },
        clip=(-1.0, 1.0),
    )
    ball_pos_rel_root = ObsTerm(func=mdp.ball_pos_rel_root)
    ball_vel_rel_root = ObsTerm(func=mdp.ball_vel_rel_root)
    ball_time_to_impact = ObsTerm(func=mdp.ball_time_to_impact, params={"safe_distance": 0.6})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class DodgeballObservationsCfg:
    policy: DodgeballPolicyObsCfg = DodgeballPolicyObsCfg()
    critic: DodgeballCriticObsCfg = DodgeballCriticObsCfg()


@configclass
class DodgeballRewardsCfg(RewardsCfg):
    """Base locomotion rewards with dodgeball-specific safety shaping."""

    # Disable command-tracking rewards from locomotion-height.
    track_lin_vel_xy_exp = None
    track_ang_vel_z_exp = None
    track_base_height_exp_smooth = None
    no_undersired_base_velocity_exp = None
    equal_foot_force_if_null_cmd = None
    stand_with_both_feet_if_null_cmd = None
    relax_if_null_cmd = None
    # Isaac Lab ContactSensorData in current runtime does not expose velocities_w_history.
    # Disable inherited impact-velocity term that depends on this field.
    impact_velocity = None

    dodgeball_survival = RewTerm(func=mdp.dodgeball_survival_reward, weight=0.2)
    dodgeball_clearance = RewTerm(
        func=mdp.ball_clearance_reward,
        weight=1.0,
        params={"safe_distance": 0.7, "distance_std": 0.5},
    )
    dodgeball_closing_speed = RewTerm(
        func=mdp.ball_closing_speed_penalty,
        weight=-0.2,
        params={"speed_threshold": 0.2},
    )
    # Contact penalty only for ball↔robot collisions (ground contact filtered out by sensor).
    dodgeball_robot_contact_penalty = RewTerm(
        func=mdp.ball_robot_contact_penalty,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("dodgeball_robot_contact"), "force_threshold": 2.0},
    )


@configclass
class DodgeballTerminationsCfg(TerminationsCfg):
    """Base locomotion terminations with dodgeball impact termination."""

    dodgeball_hit_upper_body = DoneTerm(
        func=mdp.ball_contact_protected_body,
        params={
            "sensor_cfg": SceneEntityCfg("dodgeball_robot_contact"),
            "force_threshold": 2.0,
        },
    )
    dodgeball_passed_humanoid = DoneTerm(
        func=mdp.ball_passed_humanoid,
        params={
            "object_cfg": SceneEntityCfg("dodgeball"),
            "reference_asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "pass_x_threshold": -0.15,
        },
    )


@configclass
class DodgeballEventCfg(LocomotionEventCfg):
    """Locomotion events with one-shot dodgeball launch at episode reset."""

    reset_dodgeball = EventTerm(
        func=mdp.reset_dodgeball_towards_curriculum_target,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("dodgeball"),
            "pose_range": {"roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "launch_distance_range": (2.5, 4.0),
            "launch_height_range": (1.5, 2.0),
            "front_half_angle_deg": 30.0,
            # Speed curriculum: gradually increase max ball speed up to 10 m/s.
            "max_launch_speed_start": 4.5,
            "max_launch_speed_end": 10.0,
            "max_launch_speed_curriculum_steps": 300_000,
            "time_to_impact_range": (0.45, 0.9),
            "lateral_noise_range": (-0.08, 0.08),
            "vertical_noise_range": (-0.05, 0.08),
            # Start easy with external links (head/arms/legs), then progress to torso, then lower body.
            "easy_target_body_patterns": (
                "head_link",
                ".*wrist.*",
                ".*elbow.*",
                ".*ankle_roll_link",
                ".*knee_link",
            ),
            "medium_target_body_patterns": ("torso_link", "pelvis"),
            "hard_target_body_patterns": (".*_hip_.*_link", ".*knee_link", ".*ankle_roll_link"),
            "curriculum_switch_steps": (100_000, 250_000),
        },
    )


@configclass
class G1DodgeballEnvCfg(G1LowerVelocityHeightEnvCfg):
    """G1 dodgeball environment built on top of agile locomotion-height control."""

    scene: DodgeballSceneCfg = DodgeballSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: DodgeballObservationsCfg = DodgeballObservationsCfg()
    rewards: DodgeballRewardsCfg = DodgeballRewardsCfg()
    terminations: DodgeballTerminationsCfg = DodgeballTerminationsCfg()
    events: DodgeballEventCfg = DodgeballEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        # Remove command sampling/following path for pure in-place dodgeball.
        self.commands = None
        self.curriculum = None

        # Remove command-dependent action terms.
        self.actions.random_upper_body_pos = None
        # Keep harness optional but command-free.
        if hasattr(self.actions, "harness"):
            self.actions.harness.command_name = None

    def eval(self):
        """Command-free eval override for dodgeball task."""
        # Avoid parent eval() because it injects command-based eval observations.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.viewer.eye = (-2.5, -5.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.75)
        self.viewer.origin_type = "world"
        self.commands = None
        self.curriculum = None
        if hasattr(self.actions, "random_upper_body_pos"):
            self.actions.random_upper_body_pos = None
        if hasattr(self.actions, "harness"):
            self.actions.harness.command_name = None
        # Keep observations as policy/critic only; no command-dependent eval group.

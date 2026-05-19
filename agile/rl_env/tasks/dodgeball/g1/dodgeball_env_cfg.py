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
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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
from agile.rl_env.assets.robots import unitree_g1
from agile.rl_env.tasks.locomotion_height.g1.velocity_height_env_cfg import (
    ActionsCfg,
    G1LowerVelocityHeightEnvCfg,
    LocomotionEventCfg,
    MySceneCfg,
    RewardsCfg,
    TerminationsCfg,
)

# ---------------------------------------------------------------------------
# Upper-body joint groups for the dodgeball full-body action space.
# Wrists are excluded: their stiffness (4 N·m/rad) yields action scale ≈1.5,
# too high for stable training at the default clip of ±6.
# ---------------------------------------------------------------------------
DODGEBALL_UPPER_BODY_JOINT_NAMES = [
    "waist_.*_joint",
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
]

# Scale = 0.25 × effort_limit / stiffness, derived from G1_29DOF actuators.
DODGEBALL_UPPER_BODY_SCALE: dict[str, float] = {
    "waist_yaw_joint":           0.073,   # 0.25 * 88  / 300
    "waist_roll_joint":          0.042,   # 0.25 * 50  / 300
    "waist_pitch_joint":         0.042,
    ".*_shoulder_pitch_joint":   0.069,   # 0.25 * 25  / 90
    ".*_shoulder_roll_joint":    0.104,   # 0.25 * 25  / 60
    ".*_shoulder_yaw_joint":     0.313,   # 0.25 * 25  / 20
    ".*_elbow_joint":            0.104,   # 0.25 * 25  / 60
}


@configclass
class DodgeballSceneCfg(MySceneCfg):
    """Locomotion-height scene extended with an incoming dodgeball object."""

    # Override robot init state:
    #   pos_z = 0.80 m  — feet start ~5 cm above ground (vs. 14.6 cm at z=0.9),
    #                     contacts establish in 1-2 steps instead of 7.
    #                     Derivation: at standing, root_z≈0.782 and lowest_z≈0.036,
    #                     so lowest_z ≈ root_z − 0.746.  For 5 cm buffer: z=0.796→0.80.
    #   hip_roll ±0.20  — natural wide stance (~20 cm extra foot separation), giving
    #                     the support polygon meaningful width from the first step.
    robot = unitree_g1.G1_29DOF.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.80),
            joint_pos={
                ".*_hip_pitch_joint":    -0.10,
                ".*_knee_joint":          0.30,
                ".*_ankle_pitch_joint":  -0.20,
                # Wide stance: hip abduction for lateral foot separation.
                # Opposite signs because G1 hip-roll convention: positive =
                # leg moves in +Y (abduction on left, adduction on right).
                "left_hip_roll_joint":    0.20,
                "right_hip_roll_joint":  -0.20,
            },
            joint_vel={".*": 0.0},
        ),
    )

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
class DodgeballActionsCfg(ActionsCfg):
    """Full-body action space: policy controls legs (12) + upper body (11) = 23-D.

    The parent `ActionsCfg` provides `joint_pos` (legs) and `harness` (0-D assist).
    `random_upper_body_pos` is removed — the policy now explicitly commands the
    upper body via `upper_body_pos`.

    Upper-body joints: waist (3) + shoulders (6) + elbows (2) = 11.
    Wrists excluded: their low stiffness (4 N·m/rad) gives action scale ≈1.5,
    unstable at the default ±6 clip.
    """

    # Disable the inherited random arm randomizer — policy controls arms directly.
    random_upper_body_pos = None

    upper_body_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=DODGEBALL_UPPER_BODY_JOINT_NAMES,
        scale=DODGEBALL_UPPER_BODY_SCALE,
        use_default_offset=True,
        clip={".*": (-6.0, 6.0)},
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
    # height_scan_feet removed: dodgeball always uses flat terrain, so the
    # ray-cast residuals are constant ~0 and carry no useful information for
    # the critic.  Removing it simplifies the critic (146→97-D) and eliminates
    # spurious gradient signal from an uninformative locomotion-specific input.
    height_scan_feet = None
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
    """Base locomotion rewards with dodgeball-specific safety shaping.

    Orientation-based shaping rewards that conflict with extreme in-place dodging
    maneuvers are removed:
    - flat_orientation: penalised non-upright torso — incompatible with swallow balance.
    - feet_yaw_diff / feet_yaw_mean: foot orientation relative to base — irrelevant when
      the robot may pivot a foot to maintain CoM balance.
    - hip_pos_pen: reduced to a soft guard only (weight -0.1 instead of -1.0).

    These are replaced by com_balance, which rewards keeping the projected CoM inside
    the support polygon (following HuB's balance shaping reward, σ = 0.1 m).
    """

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

    # Remove orientation-based penalties incompatible with extreme dodging poses.
    flat_orientation = None
    feet_yaw_diff = None
    feet_yaw_mean = None

    # Reduce hip deviation penalty to a soft guard (was -1.0).
    hip_pos_pen = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.1,
        params={
            "robot_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_roll_joint",
                    ".*_hip_yaw_joint",
                ],
            ),
        },
    )

    # CoM balance reward — replaces flat_orientation.
    # Returns exp(-max(dist_beyond_support, 0)^2 / sigma^2); sigma=0.1 m from HuB.
    com_balance = RewTerm(
        func=mdp.com_balance_reward,
        weight=3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "foot_asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            # Each foot is a rectangular patch on ankle_roll_link.
            # G1 foot: ~18 cm long × 9 cm wide; ankle sits ~2 cm behind centre.
            "foot_half_len": 0.09,      # half foot length (m)
            "foot_half_width": 0.045,   # half foot width (m)
            "foot_toe_offset": 0.02,    # forward shift: ankle is closer to heel
            "sigma": 0.1,
            "force_threshold": 10.0,
            "grace_steps": 10,
        },
    )

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

    # ── Upper-body effort penalties (lighter than lower body) ──────────────────
    # Lower body (inherited): torque_limits=-0.01, dof_acc_l2=-2.5e-7, dof_vel_l2=-0.001.
    # Upper body uses 3–5× lighter weights so the policy can use arms and torso
    # freely for balance and ball avoidance without being over-penalised.
    torque_limits_upper = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.003,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DODGEBALL_UPPER_BODY_JOINT_NAMES)},
    )
    dof_acc_l2_upper = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-0.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DODGEBALL_UPPER_BODY_JOINT_NAMES)},
    )
    dof_vel_l2_upper = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0003,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=DODGEBALL_UPPER_BODY_JOINT_NAMES)},
    )

    # ── Locomotion / displacement penalty ─────────────────────────────────────
    # Penalises actual walking (both feet moving) more than single corrective
    # steps (one foot moving), and adds a slow drift term to catch gradual
    # displacement even when alternating planted feet.
    # Base displacement (root_acc, weight −2e-5) is kept for jerk suppression.
    foot_locomotion = RewTerm(
        func=mdp.foot_locomotion_penalty,
        weight=-0.3,
        params={
            "sensor_cfg":         SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "foot_asset_cfg":     SceneEntityCfg("robot",          body_names=".*ankle_roll_link"),
            "force_threshold":    10.0,
            "foot_vel_threshold": 0.15,
            "both_moving_scale":  1.0,
            "one_moving_scale":   0.2,
            "drift_scale":        0.5,
            "drift_clip":         2.0,
            "grace_steps":        10,
        },
    )


@configclass
class DodgeballTerminationsCfg(TerminationsCfg):
    """Base locomotion terminations with dodgeball impact termination.

    Orientation-based terminations (base_orientation, feet_distance, knee_distance)
    are removed so the robot is free to perform extreme in-place dodging maneuvers
    (e.g. swallow balance with a horizontal torso or single-leg stances).  Balance
    is instead enforced by com_outside_support_polygon, which checks the physically
    necessary condition: the projected CoM must remain inside the support polygon.
    """

    # Remove orientation / geometry terminations that conflict with extreme poses.
    base_orientation = None
    feet_distance = None
    knee_distance = None

    # CoM-based balance termination — replaces base_orientation.
    # com_outside_support_polygon = DoneTerm(
    #     func=mdp.com_outside_support_polygon,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
    #         "foot_asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
    #         "foot_width": 0.07,
    #         "single_foot_margin": 0.05,
    #         "double_foot_margin": 0.15,
    #         "force_threshold": 5.0,
    #     },
    # )

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
    actions: DodgeballActionsCfg = DodgeballActionsCfg()
    observations: DodgeballObservationsCfg = DodgeballObservationsCfg()
    rewards: DodgeballRewardsCfg = DodgeballRewardsCfg()
    terminations: DodgeballTerminationsCfg = DodgeballTerminationsCfg()
    events: DodgeballEventCfg = DodgeballEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        # Always keep dodgeball on flat terrain for both training and evaluation.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove command sampling/following path for pure in-place dodgeball.
        self.commands = None
        self.curriculum = None

        # random_upper_body_pos is already None in DodgeballActionsCfg.
        # Keep harness command-free (fixed target height, no velocity command).
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
        if hasattr(self.actions, "harness"):
            self.actions.harness.command_name = None
        # Keep observations as policy/critic only; no command-dependent eval group.

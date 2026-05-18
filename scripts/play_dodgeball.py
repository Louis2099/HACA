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

"""Quick visual runner for the Dodgeball-G1 task."""

# flake8: noqa

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play Dodgeball-G1 environment without a policy checkpoint.")
parser.add_argument("--task", type=str, default="Dodgeball-G1-v0", help="Gym task id to run.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--num_steps", type=int, default=3000, help="Number of steps to run (0 = unlimited).")
parser.add_argument("--mode", type=str, default="zero", choices=["zero", "sine", "random"], help="Action source.")
parser.add_argument("--action_scale", type=float, default=0.25, help="Scale for sine/random actions.")
parser.add_argument("--real_time", action="store_true", default=False, help="Attempt real-time stepping.")
parser.add_argument(
    "--freeze-robot",
    action="store_true",
    default=False,
    help="If enabled, lock robot root+joints to a fixed state for pure dodgeball visualization.",
)
parser.add_argument(
    "--policy_path",
    type=str,
    default=None,
    help="Optional TorchScript policy path. Used when --freeze-robot is disabled.",
)
parser.add_argument("--video", action="store_true", default=False, help="Record one video.")
parser.add_argument("--video_length", type=int, default=600, help="Recorded video length.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric.")
parser.add_argument(
    "--follow_robot_camera",
    action="store_true",
    default=False,
    help="Keep IsaacLab camera follow mode (asset_root). Off by default to allow manual camera control.",
)
parser.add_argument(
    "--use_task_eval_cfg",
    action="store_true",
    default=False,
    help="Apply task eval() overrides (disabled by default for command-free dodgeball).",
)
parser.add_argument(
    "--enable-traj-plot",
    action="store_true",
    default=False,
    help="Enable trajectory debug plotting (topdown + side view) in humanoid-relative frame.",
)
parser.add_argument(
    "--traj_plot_autosave_steps",
    type=int,
    default=200,
    help="Autosave interval (steps) for trajectory plot/data snapshots.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.utils.dict import print_dict  # noqa: E402

import agile.rl_env.tasks  # noqa: F401, E402
from dodgeball_plot_utils import DodgeballTrajectoryPlotter  # noqa: E402


def _prepare_env_cfg(env_cfg: ManagerBasedRLEnvCfg) -> ManagerBasedRLEnvCfg:
    env_cfg.curriculum = None
    if hasattr(env_cfg.actions, "harness"):
        del env_cfg.actions.harness
    if hasattr(env_cfg.actions, "random_upper_body_pos"):
        del env_cfg.actions.random_upper_body_pos
    # Stabilize reset behavior to avoid an initial drop impulse.
    if hasattr(env_cfg, "events") and env_cfg.events is not None:
        if hasattr(env_cfg.events, "reset_dodgeball") and env_cfg.events.reset_dodgeball is not None:
            env_cfg.events.reset_dodgeball.params["debug_print_world_z"] = True
            # For play-mode verification, randomize speed limit between curriculum bounds each reset.
            env_cfg.events.reset_dodgeball.params["randomize_curriculum_speed_for_debug"] = True
        if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
            pose_range = env_cfg.events.reset_base.params.get("pose_range", {})
            velocity_range = env_cfg.events.reset_base.params.get("velocity_range", {})
            pose_range.update(
                {
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                }
            )
            velocity_range.update(
                {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                }
            )
            env_cfg.events.reset_base.params["pose_range"] = pose_range
            env_cfg.events.reset_base.params["velocity_range"] = velocity_range
        if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
            env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    if args_cli.freeze_robot and hasattr(env_cfg.scene, "robot") and env_cfg.scene.robot is not None:
        if hasattr(env_cfg.scene.robot, "spawn") and hasattr(env_cfg.scene.robot.spawn, "articulation_props"):
            env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True

    # Prevent camera snap-back each frame by default.
    # IsaacLab's viewport controller continuously updates camera for asset_root/body origin types.
    if not args_cli.follow_robot_camera and hasattr(env_cfg, "viewer") and env_cfg.viewer is not None:
        env_cfg.viewer.origin_type = "world"
    return env_cfg


def _sine_actions(timestep: int, num_envs: int, action_dim: int, dt: float, scale: float) -> torch.Tensor:
    t = timestep * dt
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)
    for i in range(action_dim):
        freq = 0.4 + 0.2 * (i % 6)
        phase = i * np.pi / max(action_dim, 1)
        actions[:, i] = scale * np.sin(2.0 * np.pi * freq * t + phase)
    return torch.from_numpy(actions)


def _flatten_observation_for_policy(obs) -> torch.Tensor:
    """Flatten dict/TensorDict observations into a single policy tensor."""
    if isinstance(obs, torch.Tensor):
        return obs
    if isinstance(obs, dict):
        return torch.cat([value.flatten(start_dim=1) for value in obs.values()], dim=-1)
    if hasattr(obs, "values") and callable(getattr(obs, "values", None)):
        return torch.cat([value.flatten(start_dim=1) for value in obs.values()], dim=-1)
    raise TypeError(f"Unsupported observation type for policy inference: {type(obs)}")


def _capture_freeze_state(env: ManagerBasedRLEnv) -> dict[str, torch.Tensor]:
    robot = env.unwrapped.scene["robot"]
    root_pose = torch.cat([robot.data.root_pos_w.clone(), robot.data.root_quat_w.clone()], dim=-1)
    root_vel = torch.zeros_like(robot.data.root_vel_w)
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = torch.zeros_like(robot.data.joint_vel)
    env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.long)
    return {
        "root_pose": root_pose,
        "root_vel": root_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "env_ids": env_ids,
    }


def _enforce_freeze_state(env: ManagerBasedRLEnv, freeze_state: dict[str, torch.Tensor]) -> None:
    robot = env.unwrapped.scene["robot"]
    env_ids = freeze_state["env_ids"]
    robot.write_root_pose_to_sim(freeze_state["root_pose"], env_ids=env_ids)
    robot.write_root_velocity_to_sim(freeze_state["root_vel"], env_ids=env_ids)
    robot.write_joint_state_to_sim(freeze_state["joint_pos"], freeze_state["joint_vel"], env_ids=env_ids)


def _resolve_plot_params(env_cfg: ManagerBasedRLEnvCfg) -> tuple[float, float]:
    front_half_angle_deg = 30.0
    initial_height_m = 1.0
    if hasattr(env_cfg, "events") and env_cfg.events is not None and hasattr(env_cfg.events, "reset_dodgeball"):
        params = env_cfg.events.reset_dodgeball.params
        front_half_angle_deg = float(params.get("front_half_angle_deg", front_half_angle_deg))
        launch_height_range = params.get("launch_height_range", (initial_height_m, initial_height_m))
        if isinstance(launch_height_range, (list, tuple)) and len(launch_height_range) > 0:
            initial_height_m = float(launch_height_range[0])
    return front_half_angle_deg, initial_height_m


def _get_ball_in_humanoid_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.unwrapped.scene["robot"]
    dodgeball = env.unwrapped.scene["dodgeball"]
    try:
        torso_ids, _ = robot.find_bodies("torso_link")
    except Exception:
        torso_ids = None
    if torso_ids is not None and len(torso_ids) > 0:
        torso_id = int(torso_ids[0])
        ref_pos_w = robot.data.body_pos_w[:, torso_id, :]
        ref_quat_w = robot.data.body_quat_w[:, torso_id, :]
    else:
        ref_pos_w = robot.data.root_pos_w
        ref_quat_w = robot.data.root_quat_w
    ball_pos_local, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, dodgeball.data.root_pos_w)
    return ball_pos_local


def _get_humanoid_reference_height(env: ManagerBasedRLEnv) -> float:
    robot = env.unwrapped.scene["robot"]
    try:
        torso_ids, _ = robot.find_bodies("torso_link")
    except Exception:
        torso_ids = None
    if torso_ids is not None and len(torso_ids) > 0:
        torso_id = int(torso_ids[0])
        ref_pos_w = robot.data.body_pos_w[:, torso_id, :]
    else:
        ref_pos_w = robot.data.root_pos_w
    return float(ref_pos_w[:, 2].mean().item())


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    if args_cli.use_task_eval_cfg and hasattr(env_cfg, "eval"):
        env_cfg.eval()
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg = _prepare_env_cfg(env_cfg)
    front_half_angle_deg, initial_height_m = _resolve_plot_params(env_cfg)

    env = ManagerBasedRLEnv(env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_dir = os.path.abspath(os.path.join("logs", "videos", "play_dodgeball"))
        video_kwargs = {
            "video_folder": video_dir,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video for dodgeball run.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    dt = env.unwrapped.step_dt
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Mode: {args_cli.mode}")
    print(f"[INFO] Num envs: {num_envs}, action dim: {action_dim}, dt: {dt:.4f}s")
    if args_cli.freeze_robot:
        print("[INFO] Freeze mode enabled: robot actions are fixed at zero.")

    traj_plotter = None
    if args_cli.enable_traj_plot:
        print(f"[INFO] Trajectory plotting enabled. Autosave every {args_cli.traj_plot_autosave_steps} steps.")

    policy = None
    if not args_cli.freeze_robot and args_cli.policy_path:
        policy = torch.jit.load(args_cli.policy_path, map_location=device)
        policy.eval()
        print(f"[INFO] Loaded TorchScript policy from: {args_cli.policy_path}")

    timestep = 0
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            freeze_state = _capture_freeze_state(env) if args_cli.freeze_robot else None
            if args_cli.enable_traj_plot and traj_plotter is None:
                humanoid_ref_height = _get_humanoid_reference_height(env)
                traj_plotter = DodgeballTrajectoryPlotter(
                    output_root=str(Path("log_plot")),
                    front_half_angle_deg=front_half_angle_deg,
                    initial_height_m=initial_height_m,
                    initial_height_rel_m=initial_height_m - humanoid_ref_height,
                    autosave_every_steps=args_cli.traj_plot_autosave_steps,
                )
                print(
                    "[INFO] Plot guide conversion: world launch z "
                    f"{initial_height_m:.3f}m -> relative z {initial_height_m - humanoid_ref_height:.3f}m."
                )
            while simulation_app.is_running():
                start = time.time()
                if args_cli.freeze_robot:
                    _enforce_freeze_state(env, freeze_state)
                    actions = torch.zeros(num_envs, action_dim, device=device)
                elif policy is not None:
                    policy_obs = _flatten_observation_for_policy(obs)
                    actions = policy(policy_obs)
                elif args_cli.mode == "sine":
                    actions = _sine_actions(timestep, num_envs, action_dim, dt, args_cli.action_scale).to(device=device)
                else:
                    actions = args_cli.action_scale * (2.0 * torch.rand(num_envs, action_dim, device=device) - 1.0)

                obs, _, terminated, truncated, _ = env.step(actions)
                if args_cli.freeze_robot:
                    _enforce_freeze_state(env, freeze_state)

                if traj_plotter is not None:
                    rel_pos_local = _get_ball_in_humanoid_frame(env).detach().cpu().numpy()
                    done_mask = (terminated | truncated).detach().cpu().numpy().astype(bool)
                    traj_plotter.update(rel_pos_local, done_mask=done_mask)

                timestep += 1

                if args_cli.video and timestep >= args_cli.video_length:
                    break
                if args_cli.num_steps > 0 and timestep >= args_cli.num_steps:
                    break

                if args_cli.real_time:
                    sleep_time = dt - (time.time() - start)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
    finally:
        if traj_plotter is not None:
            traj_plotter.finalize()
            print(f"[INFO] Trajectory debug artifacts saved to: {traj_plotter.output_dir}")
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

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

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agile.sim2mujoco.simulation import JointCommand, SimState


def get_command_info(config: dict) -> tuple[str, list[str], int, str | None]:
    """Infer command metadata from the exported policy config.

    Returns:
        Tuple of ``(command_type, command_names, command_dim, observation_term)``.

    Notes:
        Motion-tracking policies also use ``generated_commands``, but those
        commands come from the motion file rather than the interactive
        ``CommandManager``. For those policies this helper returns
        ``command_dim=0`` so the eval script does not create a velocity command
        manager unnecessarily.
    """

    observations = config.get("observations", {}).get("policy", [])
    has_motion_tracking = "motion_tracking" in config

    for obs_cfg in observations:
        term_name = obs_cfg.get("name")
        shape = obs_cfg.get("shape", [0])
        term_dim = shape[0] if isinstance(shape, list) and shape else int(shape)

        if term_name in {"locomotion_command", "navigation_command"}:
            return ("velocity", ["vx", "vy", "wz"], 3, term_name)

        if term_name == "velocity_and_height_command":
            return ("velocity_height", ["vx", "vy", "wz", "height"], 4, term_name)

        if term_name == "generated_commands":
            if has_motion_tracking or term_dim > 4:
                return ("motion_tracking", [], 0, term_name)
            if term_dim == 4:
                return ("velocity_height", ["vx", "vy", "wz", "height"], 4, term_name)
            if term_dim == 3:
                return ("velocity", ["vx", "vy", "wz"], 3, term_name)

    return ("none", [], 0, None)


class Sim2MuJoCoDataLogger:
    """Record per-step simulation data for offline analysis."""

    def __init__(
        self,
        output_dir: str | Path,
        config: dict,
        joint_names: list[str],
        control_dt: float,
        provenance: dict | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.trajectories_dir = self.output_dir / "trajectories"
        self.trajectories_dir.mkdir(parents=True, exist_ok=True)

        self.config = config
        self.joint_names = joint_names
        self.num_joints = len(joint_names)
        self.control_dt = control_dt
        self.provenance = provenance or {}

        self.command_type, self.command_names, self.command_dim, _ = get_command_info(config)

        self._prev_vel: np.ndarray | None = None
        self._rows: list[dict[str, float]] = []
        self._step_idx = 0

        self._save_metadata()
        print(f"Sim2MuJoCoDataLogger: saving to {self.trajectories_dir}")
        print(f"  Command type: {self.command_type} (dim={self.command_dim})")

    @property
    def has_data(self) -> bool:
        return len(self._rows) > 0

    def record_step(
        self,
        sim_state: SimState,
        joint_cmd: JointCommand,
        actions: torch.Tensor,
        commands: torch.Tensor | None = None,
        episode_id: int = 0,
    ) -> None:
        row: dict[str, float] = {}

        row["episode_id"] = episode_id
        row["env_id"] = 0
        row["frame_idx"] = self._step_idx
        row["timestep"] = self._step_idx * self.control_dt
        row["is_success"] = 1.0

        jp = sim_state.joint_pos.detach().cpu().numpy()
        jv = sim_state.joint_vel.detach().cpu().numpy()

        for i in range(self.num_joints):
            row[f"joint_pos_{i}"] = float(jp[i])
            row[f"joint_vel_{i}"] = float(jv[i])

        if self._prev_vel is not None:
            acc = (jv - self._prev_vel) / self.control_dt
            for i in range(self.num_joints):
                row[f"joint_acc_{i}"] = float(acc[i])
        else:
            for i in range(self.num_joints):
                row[f"joint_acc_{i}"] = 0.0
        self._prev_vel = jv.copy()

        if sim_state.joint_effort is not None:
            jt = sim_state.joint_effort.detach().cpu().numpy()
            for i in range(self.num_joints):
                row[f"joint_torque_{i}"] = float(jt[i])

        jp_des = joint_cmd.position.detach().cpu().numpy()
        for i in range(self.num_joints):
            row[f"joint_pos_des_{i}"] = float(jp_des[i])

        rp = sim_state.root_pos.detach().cpu().numpy()
        rlv = sim_state.root_lin_vel.detach().cpu().numpy()
        rav = sim_state.root_ang_vel.detach().cpu().numpy()
        for i in range(3):
            row[f"root_pos_{i}"] = float(rp[i])
            row[f"root_lin_vel_robot_{i}"] = float(rlv[i])
            row[f"root_ang_vel_robot_{i}"] = float(rav[i])

        if commands is not None and self.command_dim > 0:
            cmd = commands.detach().cpu().numpy()
            for i in range(min(len(cmd), self.command_dim)):
                row[f"commands_{i}"] = float(cmd[i])

        act = actions.detach().cpu().numpy()
        for i in range(len(act)):
            row[f"actions_{i}"] = float(act[i])

        self._rows.append(row)
        self._step_idx += 1

    def save_episode(self, episode_id: int) -> Path | None:
        if not self._rows:
            print(f"No data to save for episode {episode_id}")
            return None

        df = pd.DataFrame(self._rows)
        filepath = self.trajectories_dir / f"episode_{episode_id:03d}.parquet"
        df.to_parquet(filepath, compression="snappy", index=False)
        print(f"Saved {len(self._rows)} steps to {filepath}")
        return filepath

    def reset(self) -> None:
        self._rows = []
        self._step_idx = 0
        self._prev_vel = None

    def save(self) -> Path | None:
        return self.save_episode(0)

    def _map_limits_to_sim_order(self, limits: list, config_joint_names: list[str]) -> list:
        if not limits or not config_joint_names:
            return []

        is_pos_limits = isinstance(limits[0], list | tuple)
        result = []
        for sim_joint_name in self.joint_names:
            if sim_joint_name in config_joint_names:
                cfg_idx = config_joint_names.index(sim_joint_name)
                if cfg_idx < len(limits):
                    result.append(limits[cfg_idx])
                else:
                    result.append([-float("inf"), float("inf")] if is_pos_limits else 1000.0)
            else:
                result.append([-float("inf"), float("inf")] if is_pos_limits else 1000.0)
        return result

    def _save_metadata(self) -> None:
        robot_cfg = self.config.get("articulations", {}).get("robot", {})
        scene_cfg = self.config.get("scene", {})

        physics_dt = scene_cfg.get("physics_dt", scene_cfg.get("dt", 0.005))
        config_joint_names = robot_cfg.get("joint_names", [])
        raw_pos_limits = robot_cfg.get("default_joint_pos_limits", [])
        raw_vel_limits = robot_cfg.get("soft_joint_vel_limits", [])

        joint_pos_limits = self._map_limits_to_sim_order(raw_pos_limits, config_joint_names)
        joint_vel_limits = self._map_limits_to_sim_order(raw_vel_limits, config_joint_names)

        metadata = {
            "physics_dt": physics_dt,
            "control_dt": self.control_dt,
            "num_joints": self.num_joints,
            "joint_names": self.joint_names,
            "joint_pos_limits": joint_pos_limits,
            "joint_vel_limits": joint_vel_limits,
            "command_type": self.command_type,
            "command_names": self.command_names,
            "command_dim": self.command_dim,
            "task_name": "sim2mujoco",
            "noise_scale": self.provenance.get("noise_scale", 0.0),
            "noise_seed": self.provenance.get("noise_seed"),
            "provenance": self.provenance,
        }

        metadata_path = self.trajectories_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

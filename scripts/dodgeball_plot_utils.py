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

import os
import time
from pathlib import Path

import numpy as np


class DodgeballTrajectoryPlotter:
    """Collect dodgeball trajectories and generate topdown/side-view debug plots."""

    def __init__(
        self,
        output_root: str = "log_plot",
        front_half_angle_deg: float = 30.0,
        initial_height_m: float = 1.0,
        initial_height_rel_m: float | None = None,
        autosave_every_steps: int = 200,
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_root) / f"dodgeball_traj_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.front_half_angle_deg = float(front_half_angle_deg)
        self.initial_height_m = float(initial_height_m)
        self.initial_height_rel_m = None if initial_height_rel_m is None else float(initial_height_rel_m)
        self.autosave_every_steps = int(max(1, autosave_every_steps))

        self._step_counter = 0
        self._last_save_step = 0
        self._active_trajs: dict[int, list[np.ndarray]] = {}
        self._completed_trajs: list[np.ndarray] = []

    def update(self, rel_pos_local: np.ndarray, done_mask: np.ndarray | None = None):
        """Update trajectory buffers from local-frame ball positions."""
        num_envs = rel_pos_local.shape[0]
        if done_mask is None:
            done_mask = np.zeros(num_envs, dtype=bool)

        for env_idx in range(num_envs):
            if env_idx not in self._active_trajs:
                self._active_trajs[env_idx] = []
            self._active_trajs[env_idx].append(rel_pos_local[env_idx].copy())
            if done_mask[env_idx]:
                traj = np.asarray(self._active_trajs[env_idx], dtype=np.float32)
                if traj.shape[0] > 1:
                    self._completed_trajs.append(traj)
                self._active_trajs[env_idx] = []

        self._step_counter += 1
        if self._step_counter - self._last_save_step >= self.autosave_every_steps:
            self.save_snapshot(tag="autosave")
            self._last_save_step = self._step_counter

    def finalize(self):
        """Flush active trajectories and save final artifacts."""
        for env_idx, traj_points in self._active_trajs.items():
            if len(traj_points) > 1:
                self._completed_trajs.append(np.asarray(traj_points, dtype=np.float32))
            self._active_trajs[env_idx] = []
        self.save_snapshot(tag="final")

    def save_snapshot(self, tag: str = "snapshot"):
        """Persist trajectory data and plots."""
        traj_array = np.array(self._completed_trajs, dtype=object)
        np.savez_compressed(
            self.output_dir / f"trajectories_{tag}.npz",
            trajectories=traj_array,
            front_half_angle_deg=self.front_half_angle_deg,
            initial_height_m=self.initial_height_m,
            initial_height_rel_m=self.initial_height_rel_m,
            step_counter=self._step_counter,
        )
        self._save_plot(tag=tag)

    def _save_plot(self, tag: str):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        ax_top, ax_side = axes

        all_trajs = list(self._completed_trajs)
        for traj_points in self._active_trajs.values():
            if len(traj_points) > 1:
                all_trajs.append(np.asarray(traj_points, dtype=np.float32))

        max_extent_xy = 3.0
        max_extent_z = 2.0
        if all_trajs:
            stacked = np.vstack(all_trajs)
            max_extent_xy = float(max(max_extent_xy, np.max(np.abs(stacked[:, :2])) + 0.5))
            max_extent_z = float(max(max_extent_z, np.max(stacked[:, 2]) + 0.5))

        # Topdown view: humanoid as circle
        humanoid_radius = 0.3
        humanoid_circle = plt.Circle((0.0, 0.0), humanoid_radius, color="gray", alpha=0.4)
        ax_top.add_patch(humanoid_circle)
        ax_top.scatter([0.0], [0.0], color="black", s=20, label="Humanoid center")

        # Front-angle guide lines
        angle_rad = np.deg2rad(self.front_half_angle_deg)
        guide_len = max_extent_xy
        for sign in (-1.0, 1.0):
            x = guide_len * np.cos(sign * angle_rad)
            y = guide_len * np.sin(sign * angle_rad)
            ax_top.plot([0.0, x], [0.0, y], "r--", linewidth=1.2, alpha=0.85)

        # Adaptive alpha: stay readable for both sparse and dense plots.
        num_trajs = max(1, len(all_trajs))
        traj_alpha = float(np.clip(30.0 / num_trajs, 0.35, 0.85))
        traj_linewidth = 1.3

        for traj in all_trajs:
            ax_top.plot(
                traj[:, 0], traj[:, 1], color="tab:blue", alpha=traj_alpha, linewidth=traj_linewidth
            )

        ax_top.set_title("Topdown: Ball Trajectories in Humanoid Frame")
        ax_top.set_xlabel("x (forward) [m]")
        ax_top.set_ylabel("y (left) [m]")
        ax_top.set_aspect("equal")
        ax_top.set_xlim(-max_extent_xy, max_extent_xy)
        ax_top.set_ylim(-max_extent_xy, max_extent_xy)
        ax_top.grid(True, alpha=0.3)

        # Side view: humanoid as rectangle
        humanoid_width = 0.5
        humanoid_height = 1.7
        rect = plt.Rectangle(
            (-humanoid_width / 2.0, 0.0),
            humanoid_width,
            humanoid_height,
            facecolor="gray",
            edgecolor="black",
            alpha=0.35,
        )
        ax_side.add_patch(rect)

        # Initial launch height guide line: convert to plotted frame (humanoid-relative) if provided.
        side_height_line = self.initial_height_m if self.initial_height_rel_m is None else self.initial_height_rel_m
        ax_side.axhline(side_height_line, color="red", linestyle="--", linewidth=1.2, alpha=0.85)

        for traj in all_trajs:
            ax_side.plot(
                traj[:, 0], traj[:, 2], color="tab:blue", alpha=traj_alpha, linewidth=traj_linewidth
            )

        ax_side.set_title("Side View: Ball Trajectories in Humanoid Frame")
        ax_side.set_xlabel("x (forward) [m]")
        ax_side.set_ylabel("z (up) [m]")
        ax_side.set_xlim(-max_extent_xy, max_extent_xy)
        ax_side.set_ylim(-0.1, max_extent_z)
        ax_side.grid(True, alpha=0.3)

        fig.suptitle("Dodgeball Trajectory Debug Plot", fontsize=12)
        fig.savefig(self.output_dir / f"trajectories_{tag}.png", dpi=180)
        plt.close(fig)

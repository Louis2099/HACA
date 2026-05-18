# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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

from __future__ import annotations

import logging
import os
import pathlib
import json
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter

log = logging.getLogger(__name__)

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("Wandb is required to log to Weights and Biases.")


class WandbSummaryWriter(SummaryWriter):
    """Summary writer for Weights and Biases."""

    def __init__(self, log_dir: str, flush_secs: int, cfg):
        super().__init__(log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]

        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError(
                "Please specify wandb_project in the runner config, e.g. legged_gym."
            )

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.init(project=project, entity=entity, name=run_name)

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }
        
        self.video_files = []

    def store_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        wandb.config.update({"runner_cfg": runner_cfg})
        wandb.config.update({"policy_cfg": policy_cfg})
        wandb.config.update({"alg_cfg": alg_cfg})
        try:
            wandb.config.update({"env_cfg": env_cfg.to_dict()})
        except Exception:
            wandb.config.update({"env_cfg": asdict(env_cfg)})

    def add_scalar(
        self, tag, scalar_value, global_step=None, walltime=None, new_style=False
    ):
        super().add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
            walltime=walltime,
            new_style=new_style,
        )
        wandb.log({self._map_path(tag): scalar_value}, step=global_step)

    def stop(self):
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        self.store_config(env_cfg, runner_cfg, alg_cfg, policy_cfg)

    def save_model(self, model_path, iter):
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        wandb.save(path, base_path=os.path.dirname(path))
        
    def add_video_files(self, log_dir: str, step: int, fps: int = 30):
        """Upload newly discovered .mp4 files in log_dir to W&B.

        Each file is logged under the key ``video/train_policy`` using
        ``wandb.Video``.  Upload failures are caught and logged as warnings so
        that training is never interrupted by W&B issues.
        """
        if not os.path.exists(log_dir):
            return
        for root, _dirs, files in os.walk(log_dir):
            for video_file in sorted(files):
                if not video_file.endswith(".mp4"):
                    continue
                if video_file in self.video_files:
                    continue
                self.video_files.append(video_file)
                video_path = os.path.join(root, video_file)
                try:
                    wandb.log(
                        {"video/train_policy": wandb.Video(video_path, fps=fps, format="mp4")},
                        step=step,
                    )
                except Exception as exc:
                    log.warning(
                        "[WandbSummaryWriter] Failed to upload video %s to W&B: %s",
                        video_path,
                        exc,
                    )


    """
    Private methods.
    """

    def _map_path(self, path):
        if path in self.name_map:
            return self.name_map[path]
        else:
            return path

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

import torch
from tensordict.tensordict import TensorDict

from .barrier_terms import distance_velocity_barrier


class CBFActionFilter:
    """Lightweight CBF-style action projector operating on batched observations."""

    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.observation_pos_key = cfg.get("observation_pos_key", "ball_pos_rel_root")
        self.observation_vel_key = cfg.get("observation_vel_key", "ball_vel_rel_root")
        self.safe_distance = float(cfg.get("safe_distance", 0.6))
        self.reaction_time = float(cfg.get("reaction_time", 0.25))
        self.projection_gain = float(cfg.get("projection_gain", 1.0))
        self.max_projection_norm = float(cfg.get("max_projection_norm", 0.5))
        self.action_clip = cfg.get("action_clip", None)

    def _extract_observation(self, obs, key: str) -> torch.Tensor | None:
        if isinstance(obs, TensorDict):
            return obs.get(key, None)
        if isinstance(obs, dict):
            return obs.get(key, None)
        return None

    def filter_actions(self, obs, actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.enabled:
            zeros = torch.zeros(actions.shape[0], device=actions.device)
            return actions, {
                "cbf_barrier": zeros,
                "cbf_violation": zeros,
                "cbf_projection_norm": zeros,
                "cbf_safe_action_ratio": zeros,
            }

        rel_pos = self._extract_observation(obs, self.observation_pos_key)
        rel_vel = self._extract_observation(obs, self.observation_vel_key)
        if rel_pos is None or rel_vel is None:
            zeros = torch.zeros(actions.shape[0], device=actions.device)
            return actions, {
                "cbf_barrier": zeros,
                "cbf_violation": zeros,
                "cbf_projection_norm": zeros,
                "cbf_safe_action_ratio": zeros,
            }

        rel_pos = rel_pos[..., :3]
        rel_vel = rel_vel[..., :3]
        barrier, _ = distance_velocity_barrier(
            relative_position=rel_pos,
            relative_velocity=rel_vel,
            safe_distance=self.safe_distance,
            reaction_time=self.reaction_time,
        )
        violation = torch.clamp(-barrier, min=0.0)

        direction = rel_pos / torch.norm(rel_pos, dim=-1, keepdim=True).clamp_min(1.0e-6)
        correction = self.projection_gain * violation.unsqueeze(-1) * direction

        correction_norm = torch.norm(correction, dim=-1, keepdim=True).clamp_min(1.0e-6)
        scale = torch.clamp(self.max_projection_norm / correction_norm, max=1.0)
        correction = correction * scale

        safe_actions = actions.clone()
        correction_dims = min(safe_actions.shape[-1], correction.shape[-1])
        safe_actions[..., :correction_dims] = safe_actions[..., :correction_dims] + correction[..., :correction_dims]
        if self.action_clip is not None:
            safe_actions = torch.clamp(safe_actions, -float(self.action_clip), float(self.action_clip))

        projection_norm = torch.norm(safe_actions - actions, dim=-1)
        safe_action_ratio = (projection_norm > 1.0e-6).float()
        return safe_actions, {
            "cbf_barrier": barrier,
            "cbf_violation": violation,
            "cbf_projection_norm": projection_norm,
            "cbf_safe_action_ratio": safe_action_ratio,
        }

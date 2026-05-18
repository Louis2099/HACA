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


def distance_velocity_barrier(
    relative_position: torch.Tensor,
    relative_velocity: torch.Tensor,
    safe_distance: float,
    reaction_time: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a velocity-aware distance barrier and incoming radial speed."""
    distance = torch.norm(relative_position, dim=-1).clamp_min(1.0e-6)
    radial_speed = -(relative_position * relative_velocity).sum(dim=-1) / distance
    approaching_speed = torch.clamp(radial_speed, min=0.0)
    barrier = distance - safe_distance - reaction_time * approaching_speed
    return barrier, approaching_speed

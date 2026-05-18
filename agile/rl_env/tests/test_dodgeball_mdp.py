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

import unittest
from unittest.mock import MagicMock

import torch

from isaaclab.managers import SceneEntityCfg

from agile.rl_env.mdp.observations.dodgeball_observations import ball_pos_rel_root, ball_time_to_impact, ball_vel_rel_root
from agile.rl_env.mdp.rewards.dodgeball_rewards import ball_clearance_reward, ball_closing_speed_penalty
from agile.rl_env.mdp.terminations import ball_hit_protected_body


class TestDodgeballMdp(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock()
        self.env.device = "cpu"
        self.env.num_envs = 2

        robot = MagicMock()
        robot.data = MagicMock()
        robot.data.root_pos_w = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        robot.data.root_lin_vel_w = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        robot.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        robot.data.body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]],
            ]
        )

        dodgeball = MagicMock()
        dodgeball.data = MagicMock()
        dodgeball.data.root_pos_w = torch.tensor([[1.0, 0.0, 1.0], [0.05, 0.0, 1.0]])
        dodgeball.data.root_lin_vel_w = torch.tensor([[-1.0, 0.0, 0.0], [-0.1, 0.0, 0.0]])

        scene_dict = {"robot": robot, "dodgeball": dodgeball}
        scene = MagicMock()
        scene.__getitem__ = lambda s, k: scene_dict[k]  # noqa: ARG005
        self.env.scene = scene

    def test_relative_observations(self):
        rel_pos = ball_pos_rel_root(self.env)
        rel_vel = ball_vel_rel_root(self.env)
        torch.testing.assert_close(rel_pos, torch.tensor([[1.0, 0.0, 0.0], [0.05, 0.0, 0.0]]))
        torch.testing.assert_close(rel_vel, torch.tensor([[-1.0, 0.0, 0.0], [-0.1, 0.0, 0.0]]))

    def test_time_to_impact_is_finite(self):
        tti = ball_time_to_impact(self.env, safe_distance=0.2).squeeze(-1)
        self.assertTrue(torch.all(tti >= 0.0))
        self.assertTrue(torch.all(torch.isfinite(tti)))

    def test_rewards_are_well_behaved(self):
        clearance = ball_clearance_reward(self.env, safe_distance=0.4, distance_std=0.5)
        closing_penalty = ball_closing_speed_penalty(self.env, speed_threshold=0.2)
        self.assertEqual(clearance.shape[0], self.env.num_envs)
        self.assertEqual(closing_penalty.shape[0], self.env.num_envs)
        self.assertGreater(closing_penalty[0].item(), closing_penalty[1].item())

    def test_hit_termination(self):
        asset_cfg = SceneEntityCfg("robot")
        asset_cfg.body_ids = torch.tensor([0, 1])
        terminated = ball_hit_protected_body(
            self.env,
            object_cfg=SceneEntityCfg("dodgeball"),
            asset_cfg=asset_cfg,
            distance_threshold=0.1,
        )
        self.assertFalse(bool(terminated[0].item()))
        self.assertTrue(bool(terminated[1].item()))


if __name__ == "__main__":
    unittest.main()

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
from agile.rl_env.mdp.rewards.dodgeball_rewards import (
    ball_clearance_reward,
    ball_closing_speed_penalty,
    dodgeball_non_settle_termination_penalty,
    dodgeball_settle_failure_penalty,
    dodgeball_settle_success_reward,
    self_collision_penalty,
)
from agile.rl_env.mdp.terminations import ball_hit_protected_body, ball_rested_settle_done
from agile.rl_env.tasks.dodgeball.g1.dodgeball_env import DodgeballEnv


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
        scene.__getitem__.side_effect = lambda k: scene_dict[k]
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


class TestSelfCollisionPenalty(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock()
        self.env.device = "cpu"
        self.env.num_envs = 2
        self.env.scene = MagicMock()
        self.env.scene.sensors = {}

    def _add_sensor(self, name: str, force_matrix_w: torch.Tensor | None) -> None:
        sensor = MagicMock()
        sensor.data = MagicMock()
        sensor.data.force_matrix_w = force_matrix_w
        self.env.scene.sensors[name] = sensor

    def test_missing_or_empty_force_matrix_returns_zero(self):
        self._add_sensor("self_collision_head_link", None)
        result = self_collision_penalty(self.env, ("self_collision_head_link", "missing_sensor"))
        torch.testing.assert_close(result, torch.zeros(2))

    def test_forces_below_threshold_return_zero(self):
        force_matrix = torch.tensor(
            [
                [[[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]],
                [[[0.0, 0.0, 5.0], [1.0, 1.0, 1.0]]],
            ]
        )
        self._add_sensor("self_collision_left_elbow_link", force_matrix)

        result = self_collision_penalty(
            self.env,
            ("self_collision_left_elbow_link",),
            force_threshold=5.0,
            force_scale=100.0,
        )

        torch.testing.assert_close(result, torch.zeros(2))

    def test_force_above_threshold_is_normalized_and_clipped(self):
        force_matrix = torch.tensor(
            [
                [[[105.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
                [[[55.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
            ]
        )
        self._add_sensor("self_collision_right_elbow_link", force_matrix)

        result = self_collision_penalty(
            self.env,
            ("self_collision_right_elbow_link",),
            force_threshold=5.0,
            force_scale=100.0,
        )

        torch.testing.assert_close(result, torch.tensor([1.0, 0.5]))

    def test_multiple_sensors_sum(self):
        first = torch.tensor(
            [
                [[[25.0, 0.0, 0.0]]],
                [[[5.0, 0.0, 0.0]]],
            ]
        )
        second = torch.tensor(
            [
                [[[0.0, 0.0, 0.0]]],
                [[[45.0, 0.0, 0.0]]],
            ]
        )
        self._add_sensor("self_collision_left_knee_link", first)
        self._add_sensor("self_collision_right_knee_link", second)

        result = self_collision_penalty(
            self.env,
            ("self_collision_left_knee_link", "self_collision_right_knee_link"),
            force_threshold=5.0,
            force_scale=100.0,
        )

        torch.testing.assert_close(result, torch.tensor([0.2, 0.4]))


class TestDodgeballRestedSettle(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock()
        self.env.device = "cpu"
        self.env.num_envs = 2
        self.env.step_dt = 0.5
        self.env.curriculum_stage = 2
        self.env.episode_length_buf = torch.full((2,), 20, dtype=torch.long)

        robot = MagicMock()
        robot.data = MagicMock()
        robot.data.default_mass = torch.ones(2, 3)
        robot.data.root_pos_w = torch.tensor([[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]])
        robot.data.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
        robot.data.body_pos_w = torch.tensor(
            [
                [[0.0, 0.0, 0.8], [-0.10, 0.08, 0.0], [0.10, -0.08, 0.0]],
                [[1.0, 1.0, 0.8], [-0.10, 0.08, 0.0], [0.10, -0.08, 0.0]],
            ],
            dtype=torch.float32,
        )
        robot.data.body_quat_w = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        )

        dodgeball = MagicMock()
        dodgeball.data = MagicMock()
        dodgeball.data.root_pos_w = torch.tensor([[2.0, 0.0, 0.5], [2.0, 0.0, 0.5]])
        dodgeball.data.root_lin_vel_w = torch.tensor([[0.05, 0.0, 0.0], [0.05, 0.0, 0.0]])

        ball_sensor = MagicMock()
        ball_sensor._num_envs = 2
        ball_sensor._device = "cpu"
        ball_sensor.data = MagicMock()
        ball_sensor.data.net_forces_w_history = torch.tensor(
            [[[[3.0, 0.0, 0.0]]], [[[3.0, 0.0, 0.0]]]],
            dtype=torch.float32,
        )
        ball_sensor.data.net_forces_w = None
        ball_sensor.data.force_matrix_w_history = None
        ball_sensor.data.force_matrix_w = None

        foot_sensor = MagicMock()
        foot_sensor.data = MagicMock()
        foot_sensor.data.net_forces_w = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 20.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 20.0], [0.0, 0.0, 20.0]],
            ],
            dtype=torch.float32,
        )

        scene_dict = {"robot": robot, "dodgeball": dodgeball}
        scene = MagicMock()
        scene.__getitem__.side_effect = lambda k: scene_dict[k]
        scene.sensors = {"dodgeball_robot_contact": ball_sensor, "contact_forces": foot_sensor}
        self.env.scene = scene

        self.object_cfg = SceneEntityCfg("dodgeball")
        self.asset_cfg = SceneEntityCfg("robot")
        self.sensor_cfg = SceneEntityCfg("dodgeball_robot_contact")
        self.support_sensor_cfg = SceneEntityCfg("contact_forces")
        self.support_sensor_cfg.body_ids = torch.tensor([1, 2])
        self.foot_asset_cfg = SceneEntityCfg("robot")
        self.foot_asset_cfg.body_ids = torch.tensor([1, 2])

    def _make_term(self):
        term = object.__new__(ball_rested_settle_done)
        term.in_settle = torch.zeros(2, dtype=torch.bool)
        term.settle_timer = torch.zeros(2)
        term.success = torch.zeros(2, dtype=torch.bool)
        term.failure = torch.zeros(2, dtype=torch.bool)
        term._env = self.env
        return term

    def _call_term(self, term):
        return term(
            self.env,
            object_cfg=self.object_cfg,
            asset_cfg=self.asset_cfg,
            sensor_cfg=self.sensor_cfg,
            support_sensor_cfg=self.support_sensor_cfg,
            foot_asset_cfg=self.foot_asset_cfg,
            settle_duration_s=1.0,
            rest_contact_force_threshold=2.0,
            rest_lin_vel_threshold=0.20,
            robot_proximity_threshold=0.35,
            support_margin=0.02,
            foot_half_len=0.09,
            foot_half_width=0.045,
            foot_toe_offset=0.02,
            force_threshold=10.0,
        )

    def test_stage1_never_triggers_rested_settle(self):
        self.env.curriculum_stage = 1
        term = self._make_term()
        done = self._call_term(term)
        torch.testing.assert_close(done, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(self.env._dodgeball_settle_success, torch.zeros(2, dtype=torch.bool))

    def test_stable_com_succeeds_after_settle_duration(self):
        term = self._make_term()
        first = self._call_term(term)
        second = self._call_term(term)
        self.assertFalse(bool(first[0].item()))
        self.assertTrue(bool(second[0].item()))
        self.assertTrue(bool(self.env._dodgeball_settle_success[0].item()))
        self.assertFalse(bool(self.env._dodgeball_settle_success[1].item()))

    def test_com_outside_support_hull_fails_at_end_of_settle(self):
        term = self._make_term()
        first = self._call_term(term)
        second = self._call_term(term)
        self.assertFalse(bool(first[1].item()))
        self.assertFalse(bool(self.env._dodgeball_settle_failure[1].item()))
        self.assertTrue(bool(second[1].item()))
        self.assertTrue(bool(self.env._dodgeball_settle_failure[1].item()))
        self.assertFalse(bool(self.env._dodgeball_settle_success[1].item()))

    def test_robot_contact_does_not_start_ground_rest_settle(self):
        self.env.scene["dodgeball"].data.root_pos_w = torch.tensor([[0.05, 0.0, 0.8], [0.05, 0.0, 0.8]])
        term = self._make_term()
        done = self._call_term(term)
        torch.testing.assert_close(done, torch.zeros(2, dtype=torch.bool))
        torch.testing.assert_close(term.in_settle, torch.zeros(2, dtype=torch.bool))

    def test_settle_rewards_read_published_terminal_flags(self):
        self.env._dodgeball_settle_success = torch.tensor([True, False])
        self.env._dodgeball_settle_failure = torch.tensor([False, True])
        torch.testing.assert_close(dodgeball_settle_success_reward(self.env), torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(dodgeball_settle_failure_penalty(self.env), torch.tensor([0.0, 1.0]))

    def test_generic_termination_penalty_excludes_settle_term(self):
        termination_manager = MagicMock()
        termination_manager.terminated = torch.tensor([True, True])
        termination_manager._term_name_to_term_idx = {
            "dodgeball_passed_humanoid": 0,
            "dodgeball_hit_upper_body": 1,
        }
        termination_manager._term_dones = torch.tensor([[True, False], [False, True]])
        self.env.termination_manager = termination_manager

        result = dodgeball_non_settle_termination_penalty(self.env)
        torch.testing.assert_close(result, torch.tensor([0.0, 1.0]))

    def test_curriculum_accounting_counts_settle_success_only(self):
        env = object.__new__(DodgeballEnv)
        env._ep_had_ball_contact = torch.tensor([False, False])
        env.reset_terminated = torch.tensor([True, True])
        env._dodgeball_settle_success = torch.tensor([True, False])
        env._cur_epoch_successes = []
        env._total_episodes = 0
        env._stage_episodes = 0

        DodgeballEnv._process_finished_episodes(env, torch.tensor([0, 1]))

        self.assertEqual(env._cur_epoch_successes, [1.0, 0.0])
        self.assertEqual(env._total_episodes, 2)
        self.assertEqual(env._stage_episodes, 2)


if __name__ == "__main__":
    unittest.main()

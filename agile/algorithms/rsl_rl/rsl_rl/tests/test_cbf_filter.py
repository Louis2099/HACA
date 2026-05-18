# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import unittest

import torch

from agile.rl_env.rsl_rl.cbf import CBFActionFilter


class TestCbfFilter(unittest.TestCase):
    def test_projection_activates_for_violation(self):
        cfg = {
            "enabled": True,
            "observation_pos_key": "ball_pos_rel_root",
            "observation_vel_key": "ball_vel_rel_root",
            "safe_distance": 0.7,
            "reaction_time": 0.25,
            "projection_gain": 2.0,
            "max_projection_norm": 0.2,
        }
        cbf_filter = CBFActionFilter(cfg)
        obs = {
            "ball_pos_rel_root": torch.tensor([[0.3, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            "ball_vel_rel_root": torch.tensor([[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        }
        actions = torch.zeros((2, 6))
        safe_actions, stats = cbf_filter.filter_actions(obs, actions)

        self.assertGreater(stats["cbf_projection_norm"][0].item(), 0.0)
        self.assertEqual(stats["cbf_projection_norm"][1].item(), 0.0)
        self.assertTrue(torch.all(torch.isfinite(safe_actions)))

    def test_filter_passthrough_when_disabled(self):
        cbf_filter = CBFActionFilter({"enabled": False})
        obs = {"ball_pos_rel_root": torch.randn(3, 3), "ball_vel_rel_root": torch.randn(3, 3)}
        actions = torch.randn(3, 10)
        safe_actions, stats = cbf_filter.filter_actions(obs, actions)
        torch.testing.assert_close(safe_actions, actions)
        self.assertTrue(torch.all(stats["cbf_projection_norm"] == 0.0))


if __name__ == "__main__":
    unittest.main()

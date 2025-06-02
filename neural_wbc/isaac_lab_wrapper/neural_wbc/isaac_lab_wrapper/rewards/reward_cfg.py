# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from isaaclab.utils import configclass


@configclass
class NeuralWBCRewardCfg:
    # Reward and penalty scales
    scales = {
        "reward_track_joint_positions": 32.0,
        "reward_track_joint_velocities": 16.0,
        "reward_track_body_velocities": 8.0,
        "reward_track_body_angular_velocities": 8.0,
        "reward_track_body_position_extended": 30.0,
        "reward_track_body_position_vr_key_points": 50.0,
        "penalize_torques": -0.0001,
        "penalize_by_torque_limits": -2,
        "penalize_joint_accelerations": -0.000011,
        "penalize_joint_velocities": -0.004,
        "penalize_lower_body_action_changes": -3.0,
        "penalize_upper_body_action_changes": -0.625,
        "penalize_by_joint_pos_limits": -125.0,
        "penalize_by_joint_velocity_limits": -50.0,
        "penalize_early_termination": -250.0,
        "penalize_feet_contact_forces": -0.75,
        "penalize_stumble": -1000.0,
        "penalize_slippage": -37.5,
        "penalize_feet_orientation": -62.5,
        "penalize_feet_air_time": 1000.0,
        "penalize_both_feet_in_air": -200.0,
        "penalize_orientation": -200.0,
        "penalize_max_feet_height_before_contact": -2500.0,
    }

    # Sigmas for exponential terms
    body_pos_lower_body_sigma = 0.5
    body_pos_upper_body_sigma = 0.03
    body_pos_vr_key_points_sigma = 0.03
    body_rot_sigma = 0.1
    body_vel_sigma = 10
    body_ang_vel_sigma = 10
    joint_pos_sigma = 0.5
    joint_vel_sigma = 1.0

    # Weights for weighted sums
    body_pos_lower_body_weight = 0.5
    body_pos_upper_body_weight = 1.0

    # Limits
    torque_limits_scale = 0.85
    # The order here follows the order in cfg.joint_names
    torque_limits = [
        300.0,
        300.0,
        300.0,
        300.0,
        200.0,
        200.0,

        300.0,
        300.0,
        300.0,
        300.0,
        200.0,
        200.0,

        300.0,

        100.0,
        100.0,
        100.0,
        20.0,
        20.0,
        20.0,
        20.0,

        100.0,
        100.0,
        100.0,
        20.0,
        20.0,
        20.0,
        20.0,

    ]
    # Joint pos limits, in the form of (lower_limit, upper_limit)
    joint_pos_limits = [
        (-1.5, 1.5),
        (-0.25, 2.8),
        (-1.2, 2.6),
        (0.0, 2.27),
        (-0.34, 0.17),
        (-0.26, 0.26),

        (-1.5, 1.5),
        (-2.8, 0.25),
        (-2.6, 1.2),
        (0.0, 2.27),
        (-0.34, 0.17),
        (-0.26, 0.26),

        (-3.0, 3.0),

        (-3.0, 3.0),
        (-0.12, 2.3),
        (-1.9, 1.9),

        (-0.83, 1.57),

        (-2.3, 2.3),
        (-1.2, 1.2),
        (-1.2, 1.2),


        (-3.0, 3.0),
        (-2.3, 0.12),
        (-1.9, 1.9),

        (-0.83, 1.57),

        (-2.3, 2.3),
        (-1.2, 1.2),
        (-1.2, 1.2),

    ]
    joint_vel_limits_scale = 0.85
    joint_vel_limits = [
        12.0,
        12.0,
        12.0,
        12.0,
        20.0,
        20.0,

        12.0,
        12.0,
        12.0,
        12.0,
        20.0,
        20.0,

        12,

        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,

        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,
        20.0,

    ]
    max_contact_force = 600.0
    max_feet_height_limit_before_contact = 0.25

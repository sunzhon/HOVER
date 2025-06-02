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

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path
from neural_wbc.data.asset.lumos import Lus2_Joint27_CFG

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
#from isaaclab_assets import H1_CFG

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": [".*torso_joint.*", ".*shoulder.*joint.*", ".*elbow.*joint.*"],
        "lower_body": [".*hip.*joint.*", ".*knee.*joint.*", ".*ankle.*joint.*", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*shoulder.*link.*",
            ".*elbow.*link.*",
            ".*hand.*link.*",
        ],
        "lower_body": [".*ankle.*link.*"],
    },
    "omnih2o": {
        "upper_body": [".*hand.*link.*", ".*head.*link.*"],
    },
}


@configclass
class NeuralWBCEnvCfgLus2(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 27
    observation_space = 777 #921
    state_space = 878 #990

    # Distillation parameters:
    single_history_dim = 63
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    # OH2O mode is tracking the head and hand positions. This can be modified to train a different specialist
    # or use the full DISTILL_MASK_MODES_ALL to train a generalist policy.
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}

    # Robot geometry / actuation parameters:
    robot: ArticulationCfg = Lus2_Joint27_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        "left_wrist_roll_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        "right_wrist_roll_link",
    ]

    # Joint names by the order in the MJCF model.
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "torso_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_roll_joint",
    ]

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # hips, knees, ankles
    upper_body_joint_ids = [12,  13, 14, 15, 16, 17, 18, 19,  20,21,22,23,24,25,26]  # torso, shoulders, elbows

    base_name = 'pelvis' #"torso_link"
    root_id = body_names.index(base_name)

    feet_name = ".*_ankle_roll_link"

    extend_body_parent_names = ["left_ankle_roll_link", "right_ankle_roll_link", "torso_link"]
    extend_body_names = ["left_toe_link", "right_toe_link", "head_link"]
    extend_body_pos = torch.tensor([[0.08, 0, 0], [0.08, 0, 0], [0, 0, 0.4]])

    # These are the bodies that are tracked by the teacher. They may also contain the extended
    # bodies.

    tracked_body_names = [
            "left_elbow_link", 
            "right_elbow_link",
            "left_hip_pitch_link",
            "right_hip_pitch_link",
            "left_shoulder_roll_link",
            "right_shoulder_roll_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link", 
            "left_knee_link", 
            "right_knee_link",
            "left_wrist_roll_link",
            "right_wrist_roll_link"
            ]

    # control parameters
    stiffness = {
        "left_hip_pitch_joint": 300.0,
        "left_hip_roll_joint": 200.0,
        "left_hip_yaw_joint": 200.0,
        "left_knee_joint": 300.0,
        "left_ankle_pitch_joint": 100.0,
        "left_ankle_roll_joint": 100.0,
        "right_hip_pitch_joint": 300.0,
        "right_hip_roll_joint": 200.0,
        "right_hip_yaw_joint": 200.0,
        "right_knee_joint": 300.0,
        "left_ankle_pitch_joint": 100.0,
        "left_ankle_roll_joint": 100.0,
        "torso_joint": 300.0,
        "left_shoulder_pitch_joint": 60.0,
        "left_shoulder_roll_joint": 60.0,
        "left_shoulder_yaw_joint": 60.0,
        "left_elbow_joint": 60.0,
        "left_wrist_pitch_joint": 40.0,
        "left_wrist_roll_joint": 40.0,
        "left_wrist_yaw_joint": 40.0,
        "right_shoulder_pitch_joint": 60.0,
        "right_shoulder_roll_joint": 60.0,
        "right_shoulder_yaw_joint": 60.0,
        "right_elbow_joint": 60.0,
        "right_wrist_pitch_joint": 40.0,
        "right_wrist_roll_joint": 40.0,
        "right_wrist_yaw_joint": 40.0,
    }

    damping = {
        "left_hip_pitch_joint": 5.0,
        "left_hip_roll_joint": 5.0,
        "left_hip_yaw_joint": 5.0,
        "left_knee_joint": 5.0,
        "left_ankle_pitch_joint": 8.0,
        "left_ankle_roll_joint": 8.0,
        "right_hip_pitch_joint": 5.0,
        "right_hip_roll_joint": 5.0,
        "right_hip_yaw_joint": 5.0,
        "right_knee_joint": 5.0,
        "left_ankle_pitch_joint": 8.0,
        "left_ankle_roll_joint": 8.0,
        "torso_joint": 5.0,
        "left_shoulder_pitch_joint": 10.0,
        "left_shoulder_roll_joint": 10.0,
        "left_shoulder_yaw_joint": 10.0,
        "left_elbow_joint": 10.0,
        "left_wrist_pitch_joint": 5.0,
        "left_wrist_roll_joint": 5.0,
        "left_wrist_yaw_joint": 5.0,
        "right_shoulder_pitch_joint": 10.0,
        "right_shoulder_roll_joint": 10.0,
        "right_shoulder_yaw_joint": 10.0,
        "right_elbow_joint": 10.0,
        "right_wrist_pitch_joint": 5.0,
        "right_wrist_roll_joint": 5.0,
        "right_wrist_yaw_joint": 5.0,
    }

    mass_randomized_body_names = [
        "pelvis",
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
        "torso_link",
    ]

    undesired_contact_body_names = [
        "pelvis",
        #".*_yaw_link",
        #".*_roll_link",
        #".*_pitch_link",
        #".*_knee_link",
    ]

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        # Apply a grid pattern that is smaller than the resolution to only return one height value.
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.motion_path = get_data_path("../../../../humanoid_demo_retarget/data/lus2_joint27/fit_motion/CMU_CMU_80.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("../../../../lumos_rl_gym/resources/robots/lus2/mjcf/lus2_joint27.xml")
        
        # flat terrain
        self.terrain = flat_terrain

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "pelvis" #"torso_link"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "pelvis" #"torso_link"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            # Do not reset mask when there is only one mode.
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`UNITREE_GO1_CFG`: Unitree Go1 robot with actuator net model for the legs
* :obj:`UNITREE_GO2_CFG`: Unitree Go2 robot with DC motor model for the legs
* :obj:`Lus1_CFG`: H1 humanoid robot
* :obj:`G1_CFG`: G1 humanoid robot
* :obj:`G1_MINIMAL_CFG`: G1 humanoid robot with minimal collision bodies

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import os


##
# Configuration
##

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
usd_dir_path = os.path.join(BASE_DIR, "../../../../../lumos_rl_gym/resources")
usd_dir_path = os.path.join("/home/thomas/workspace/lumos_ws/lumos_rl_gym/resources")

Lus1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{usd_dir_path}/robots/lus1_v5/usd/lus1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.0, #-0.3,
            ".*_knee_joint": 0.0, #0.3,
            ".*_ankle_pitch_joint": 0.0, #-0.18,
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0, #0.2,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0, #-0.6, 
            ".*_wrist.*": 0.0, #-0.6, 
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=300,
            velocity_limit_sim=30.0,
            stiffness={
                ".*_hip_yaw_joint": 300.0, #150
                ".*_hip_roll_joint": 300.0,
                ".*_hip_pitch_joint": 350.0,
                ".*_knee_joint": 350.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 10.0,
                ".*_hip_roll_joint": 10.0,
                ".*_hip_pitch_joint": 15.0,
                ".*_knee_joint": 15.0,
                "torso_joint": 10.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint",".*_ankle_pitch_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={".*_ankle_.*_joint": 80.0},
            damping={".*_ankle_.*_joint": 2.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
            },
        ),

        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*_joint"],
            effort_limit_sim=20,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_wrist_.*_joint": 20.0,
            },
            damping={
                ".*_wrist_.*_joint": 5,
            },
        ),
    },
)
"""Configuration for the Kepler Lus1 Humanoid robot."""


Lus1_Joint25_CFG = Lus1_CFG.copy()
Lus1_Joint25_CFG.spawn.usd_path = f"{usd_dir_path}/robots/lus1_v5/usd/lus1_joint25.usd"



Lus2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{usd_dir_path}/robots/lus2/usd/lus2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw_joint": 0.0,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_pitch_joint": -0.0, #-0.3,
            ".*_knee_joint": 0.0, #0.3,
            ".*_ankle_pitch_joint": 0.0, #-0.18,
            ".*_ankle_roll_joint": 0.0,
            "torso_joint": 0.0,
            ".*_shoulder_pitch_joint": 0.0, #0.2,
            "left_shoulder_roll_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            ".*_shoulder_yaw_joint": 0.0,
            ".*_elbow_joint": 0.0, #-0.6, 
            ".*_wrist.*": 0.0, #-0.6, 
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=300,
            velocity_limit_sim=30.0,
            stiffness={
                ".*_hip_yaw_joint": 200.0, #150
                ".*_hip_roll_joint": 200.0,
                ".*_hip_pitch_joint": 300.0,
                ".*_knee_joint": 300.0,
                "torso_joint": 300.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
            friction=0.001,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint",".*_ankle_pitch_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={".*_ankle_.*_joint": 100.0},
            damping={".*_ankle_.*_joint": 8.0},
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=100,
            velocity_limit_sim=20.0,
            stiffness={
                ".*_shoulder_pitch_joint": 60.0,
                ".*_shoulder_roll_joint": 60.0,
                ".*_shoulder_yaw_joint": 60.0,
                ".*_elbow_joint": 60.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 10.0,
                ".*_shoulder_roll_joint": 10.0,
                ".*_shoulder_yaw_joint": 10.0,
                ".*_elbow_joint": 10.0,
            },
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
            },


        ),

        "wrist": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_.*_joint"],
            effort_limit_sim=20,
            velocity_limit_sim=15.0,
            stiffness={
                ".*_wrist_.*_joint": 40.0,
            },
            damping={
                ".*_wrist_.*_joint": 5.0,
            },
            armature=0.001,
        ),
    },
)



Lus2_Joint25_CFG = Lus2_CFG.copy()
Lus2_Joint25_CFG.spawn.usd_path = f"{usd_dir_path}/robots/lus2/usd/lus2_joint25.usd"

Lus2_Joint27_CFG = Lus2_CFG.copy()
Lus2_Joint27_CFG.spawn.usd_path = f"{usd_dir_path}/robots/lus2/usd/lus2_joint27.usd"

# NOTE 

USE_Lus2_CFG = Lus2_Joint27_CFG

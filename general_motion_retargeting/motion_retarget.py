import mink
import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Dict, List
from pydantic import BaseModel
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT


class IKConfigs(BaseModel):
    class IKMatch(BaseModel):
        human_frame_name: str
        position: List[float]
        rotation: List[float]

    class TaskWeight(BaseModel):
        position: float
        rotation: float

    robot_root_name: str
    human_root_name: str
    human_height_assumption: float
    human_scale_table: Dict[str, float]
    ground_height: float
    ik_match_table: Dict[str, IKMatch]
    task_weights: List[Dict[str, TaskWeight]]


class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR)."""

    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str = "daqp",  # change from "quadprog" to "daqp".
        damping: float = 5e-1,  # change from 1e-1 to 1e-2.
        use_velocity_limit: bool = False,
    ) -> None:

        self.max_iter = 10
        self.ground_offset = 0.0

        self.solver = solver
        self.damping = damping

        # load the robot model
        self.model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT[tgt_robot]))

        self.robot_dof_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i]): i
            for i in range(self.model.nv)
        }

        self.robot_body_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i): i
            for i in range(self.model.nbody)
        }

        self.robot_motor_names = {
            mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self.model.nu)
        }

        # Load the IK config
        ik_config = IKConfigs.model_validate_json(
            IK_CONFIG_DICT[src_human][tgt_robot].read_text()
        )

        # used for retargeting
        self.human_root_name = ik_config.human_root_name
        self.robot_root_name = ik_config.robot_root_name
        self.ik_match_table = ik_config.ik_match_table

        # compute the scale ratio based on given human height and the assumption in the IK config
        ratio = (
            actual_human_height / ik_config.human_height_assumption
            if actual_human_height is not None
            else 1.0
        )

        self.human_scale_table = {
            frame: scale * ratio for frame, scale in ik_config.human_scale_table.items()
        }
        self.ground = ik_config.ground_height * np.array([0, 0, 1])

        self.ik_limits = [mink.ConfigurationLimit(self.model)]

        if use_velocity_limit:
            VELOCITY_LIMITS = {k: 3 * np.pi for k in self.robot_motor_names}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))

        self.configuration = mink.Configuration(self.model)

        self.problems = [
            [
                mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=weights.position,
                    orientation_cost=weights.rotation,
                    lm_damping=1,
                )
                for frame_name, weights in problem.items()
            ]
            for problem in ik_config.task_weights
        ]

        self.frame_offsets = {
            ik_match.human_frame_name: mink.SE3(
                np.hstack([ik_match.rotation, ik_match.position])
            )
            for ik_match in ik_config.ik_match_table.values()
        }

        self.robot_to_human_map = {
            frame_name: ik_match.human_frame_name
            for frame_name, ik_match in self.ik_match_table.items()
        }

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        human_data = self.scale_human_data(human_data)
        human_data = self.offset_human_data(human_data)
        human_data = self.apply_ground_offset(human_data)

        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)

        self.scaled_human_data = human_data

        for problem in self.problems:
            for task in problem:
                pos, rot = human_data[self.robot_to_human_map[task.frame_name]]
                task.set_target(
                    mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos)
                )

    def solve(self, tasks):
        dt = self.configuration.model.opt.timestep
        prev_error = np.inf

        for _ in range(self.max_iter):
            vel = mink.solve_ik(
                self.configuration,
                tasks,
                dt,
                self.solver,
                self.damping,
                False,
                self.ik_limits,
            )
            self.configuration.integrate_inplace(vel, dt)

            curr_error = np.linalg.norm(
                np.concatenate(
                    [task.compute_error(self.configuration) for task in tasks]
                )
            )

            if abs(prev_error - curr_error) < 0.001:
                break

            prev_error = curr_error

    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)

        for problem in self.problems:
            self.solve(problem)

        return self.configuration.data.qpos.copy()

    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [
                np.asarray(human_data[body_name][0]),
                np.asarray(human_data[body_name][1]),
            ]
        return human_data

    def scale_human_data(self, human_data):
        human_data_local = {}
        root_pos, root_quat = human_data[self.human_root_name]

        # scale root
        scaled_root_pos = self.human_scale_table[self.human_root_name] * root_pos

        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in self.human_scale_table:
                continue
            if body_name == self.human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (
                    human_data[body_name][0] - root_pos
                ) * self.human_scale_table[body_name]

        # transform the human data back to the global frame
        human_data_global = {self.human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (
                human_data_local[body_name] + scaled_root_pos,
                human_data[body_name][1],
            )

        return human_data_global

    def offset_human_data(self, human_data):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data:
            pos, quat = human_data[body_name]
            body_pose = mink.SE3(np.hstack([quat, pos]))
            body_pose = body_pose.multiply(self.frame_offsets[body_name])
            offset_human_data[body_name] = [
                body_pose.translation(),
                body_pose.rotation().wxyz,
            ]

        return offset_human_data

    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = (
                pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
            )
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data

import numpy as np

from libero.libero.envs.libero_sawyer_pose import (
    sawyer_init_qpos_mounted,
    sawyer_study_table_x_offset,
    sawyer_table_x_offset,
)
from libero.libero.envs.libero_sawyer_xml import patched_sawyer_robot_xml
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel


class MountedSawyer(ManipulatorModel):
    """Sawyer on Rethink mount — LIBERO table scenes (kitchen / study)."""

    def __init__(self, idn=0):
        super().__init__(patched_sawyer_robot_xml(), idn=idn)

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return sawyer_init_qpos_mounted()

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": sawyer_table_x_offset,
            "study_table": sawyer_study_table_x_offset,
            "kitchen_table": sawyer_table_x_offset,
            "living_room_table": sawyer_table_x_offset,
            "coffee_table": sawyer_table_x_offset,
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"

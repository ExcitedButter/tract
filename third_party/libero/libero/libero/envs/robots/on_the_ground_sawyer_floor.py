"""Sawyer for LIBERO floor scenes (libero_object, etc.) — home matches Franka on floor."""

import numpy as np

from libero.libero.envs.libero_sawyer_pose import sawyer_init_qpos_floor
from libero.libero.envs.libero_sawyer_xml import patched_sawyer_robot_xml
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel


class OnTheGroundSawyerFloor(ManipulatorModel):
    """Sawyer without mount — ``arena_type=floor`` (base offset empty)."""

    def __init__(self, idn=0):
        super().__init__(patched_sawyer_robot_xml(), idn=idn)

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_sawyer"

    @property
    def init_qpos(self):
        return sawyer_init_qpos_floor()

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.42),
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

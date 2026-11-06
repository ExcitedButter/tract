import numpy as np

from libero.libero.envs.libero_sawyer_pose import (
    sawyer_init_qpos_living_room,
    sawyer_living_room_x_offset,
    sawyer_table_x_offset,
)
from libero.libero.envs.libero_sawyer_xml import patched_sawyer_robot_xml
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel


class OnTheGroundSawyer(ManipulatorModel):
    """Sawyer without mount — LIBERO living-room / coffee-table scenes."""

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
        return sawyer_init_qpos_living_room()

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": sawyer_table_x_offset,
            "coffee_table": lambda table_length: (
                sawyer_table_x_offset(table_length)[0],
                0.0,
                0.41,
            ),
            "living_room_table": sawyer_living_room_x_offset,
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

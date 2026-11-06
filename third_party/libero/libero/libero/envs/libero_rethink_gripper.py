"""Rethink gripper with MuJoCo 3.6-compatible MJCF."""

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.models.grippers.rethink_gripper import RethinkGripper

from libero.libero.envs.libero_rethink_gripper_xml import patched_rethink_gripper_xml


class LiberoRethinkGripper(RethinkGripper):
    def __init__(self, idn=0):
        GripperModel.__init__(self, patched_rethink_gripper_xml(), idn=idn)

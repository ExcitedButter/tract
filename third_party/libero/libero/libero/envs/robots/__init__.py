from .mounted_panda import MountedPanda
from . import libero_plus_mounted_panda_variants  # noqa: F401 — MountedPanda1..500 for libero_robot
from .on_the_ground_panda import OnTheGroundPanda
from . import libero_plus_on_the_ground_panda_variants  # noqa: F401 — OnTheGroundPanda1..500
from .mounted_ur5e import MountedUR5e
from .on_the_ground_ur5e import OnTheGroundUR5e
from .mounted_sawyer import MountedSawyer
from .on_the_ground_sawyer import OnTheGroundSawyer
from .on_the_ground_sawyer_floor import OnTheGroundSawyerFloor

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.models.grippers import GRIPPER_MAPPING

from libero.libero.envs.libero_rethink_gripper import LiberoRethinkGripper

GRIPPER_MAPPING["RethinkGripper"] = LiberoRethinkGripper

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedUR5e": SingleArm,
        "OnTheGroundUR5e": SingleArm,
        "MountedSawyer": SingleArm,
        "OnTheGroundSawyer": SingleArm,
        "OnTheGroundSawyerFloor": SingleArm,
    }
)

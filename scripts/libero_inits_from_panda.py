"""Build non-Franka LIBERO inits from Franka (or cross UR5) pruned_init tensors.

Used by the evaluation scripts for the cross-robot (``*_cross`` / ``*_sawyer``) suites:
copy object/scene qpos, then IK the target arm to match source EEF pose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from libero.libero.envs import OffScreenRenderEnv


def _joint_qpos_size(model, joint_id: int) -> int:
    jtype = model.jnt_type[joint_id]
    if jtype == 0:
        return 7
    if jtype == 3:
        return 1
    if jtype == 2:
        return 4
    return int(model.jnt_dof[joint_id])


def _is_robot_joint(name: str) -> bool:
    return name.startswith("robot") or "gripper" in name


def _collect_gripper_qpos_flat(model, data) -> np.ndarray:
    specs: list[tuple[int, int]] = []
    for j in range(model.njnt):
        name = model.joint(j).name
        if "gripper" not in name:
            continue
        adr = model.jnt_qposadr[j]
        specs.append((adr, _joint_qpos_size(model, j)))
    specs.sort(key=lambda x: x[0])
    parts = [data.qpos[adr : adr + sz] for adr, sz in specs]
    return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=np.float64)


# UR5e Robotiq home on libero_10_cross ↔ Rethink open pose on Sawyer (reset reference).
_UR5_ROBOTIQ_REF = (0.026, 0.267)
_RETHINK_GRIPPER_OPEN = 0.020833


def _robotiq_opening_to_rethink(gq: np.ndarray) -> np.ndarray:
    """Map Robotiq85 (6) qpos → RethinkGripper (2) using paired finger joints 0 and 3."""
    gq = np.asarray(gq, dtype=np.float64).reshape(-1)
    if gq.size < 4:
        spread = float(np.mean(np.abs(gq))) if gq.size else _RETHINK_GRIPPER_OPEN
        return np.array([spread, -spread], dtype=np.float64)
    t_a = abs(float(gq[0])) / _UR5_ROBOTIQ_REF[0]
    t_b = abs(float(gq[3])) / _UR5_ROBOTIQ_REF[1]
    spread = _RETHINK_GRIPPER_OPEN * 0.5 * (t_a + t_b)
    spread = float(np.clip(spread, 0.0, 0.04))
    return np.array([spread, -spread], dtype=np.float64)


def copy_gripper_qpos(sim_src, sim_dst) -> int:
    """Copy gripper qpos source→target; Robotiq85 (6) ↔ Rethink (2) uses paired-finger mapping."""
    model_s, model_d = sim_src.model, sim_dst.model
    data_s, data_d = sim_src.data, sim_dst.data

    def _gripper_specs(model):
        specs: list[tuple[str, int, int]] = []
        for j in range(model.njnt):
            name = model.joint(j).name
            if "gripper" not in name:
                continue
            adr = model.jnt_qposadr[j]
            specs.append((name, adr, _joint_qpos_size(model, j)))
        return sorted(specs, key=lambda x: x[0])

    src_flat = _collect_gripper_qpos_flat(model_s, data_s)
    dst_specs = _gripper_specs(model_d)
    if src_flat.size >= 6 and len(dst_specs) == 2:
        rethink = _robotiq_opening_to_rethink(src_flat)
        for i, (_, adr_d, sz_d) in enumerate(dst_specs):
            data_d.qpos[adr_d : adr_d + sz_d] = rethink[i : i + 1]
        return len(dst_specs)
    if src_flat.size == 2 and len(dst_specs) >= 6:
        spread = 0.5 * (abs(float(src_flat[0])) + abs(float(src_flat[1])))
        for _, adr_d, sz_d in dst_specs:
            data_d.qpos[adr_d : adr_d + min(sz_d, 1)] = -spread
        return len(dst_specs)

    src_specs = _gripper_specs(model_s)
    n = min(len(src_specs), len(dst_specs))
    for (_, adr_s, sz_s), (_, adr_d, sz_d) in zip(src_specs[:n], dst_specs[:n]):
        size = min(sz_s, sz_d)
        data_d.qpos[adr_d : adr_d + size] = data_s.qpos[adr_s : adr_s + size]
    return n


def copy_scene_qpos(sim_src, sim_dst) -> int:
    model_s, model_d = sim_src.model, sim_dst.model
    data_s, data_d = sim_src.data, sim_dst.data
    names_src = {model_s.joint(i).name for i in range(model_s.njnt)}
    n = 0
    for j in range(model_d.njnt):
        name = model_d.joint(j).name
        if _is_robot_joint(name) or name not in names_src:
            continue
        jid_s = model_s.joint_name2id(name)
        jid_d = model_d.joint_name2id(name)
        adr_s = model_s.jnt_qposadr[jid_s]
        adr_d = model_d.jnt_qposadr[jid_d]
        size_s = _joint_qpos_size(model_s, jid_s)
        size_d = _joint_qpos_size(model_d, jid_d)
        size = min(size_s, size_d)
        data_d.qpos[adr_d : adr_d + size] = data_s.qpos[adr_s : adr_s + size]
        n += 1
    return n


def ik_arm_to_eef(
    sim,
    robot,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    x0: np.ndarray,
    *,
    orientation_weight: float = 8.0,
    posture_weight: float = 0.01,
    maxiter: int = 200,
) -> np.ndarray:
    idx = robot._ref_joint_pos_indexes
    eef = robot.robot_model.eef_name

    def loss(q: np.ndarray) -> float:
        sim.data.qpos[idx] = q
        sim.forward()
        pos = sim.data.get_body_xpos(eef)
        quat = sim.data.get_body_xquat(eef)
        pos_cost = float(np.sum((pos - target_pos) ** 2))
        ori_cost = 1.0 - abs(float(np.dot(quat, target_quat)))
        posture_cost = float(np.sum((q - x0) ** 2))
        return pos_cost + orientation_weight * ori_cost + posture_weight * posture_cost

    res = minimize(
        loss,
        x0,
        method="L-BFGS-B",
        bounds=[(-3.14, 3.14)] * len(x0),
        options={"maxiter": maxiter},
    )
    return res.x


def target_init_from_source_vec(
    source_vec: np.ndarray,
    *,
    bddl_source: Path,
    bddl_target: Path,
    resolution: int = 128,
    render_gpu_device_id: int = -1,
) -> np.ndarray:
    """Map a flattened init from *source* sim → *target* sim (same scene + matched EEF)."""
    render_kwargs = {"render_gpu_device_id": int(render_gpu_device_id)}
    env_src = OffScreenRenderEnv(
        bddl_file_name=str(bddl_source),
        camera_heights=resolution,
        camera_widths=resolution,
        **render_kwargs,
    )
    env_dst = OffScreenRenderEnv(
        bddl_file_name=str(bddl_target),
        camera_heights=resolution,
        camera_widths=resolution,
        **render_kwargs,
    )
    env_src.reset()
    env_dst.reset()
    env_src.set_init_state(source_vec)

    sim_src, sim_dst = env_src.env.sim, env_dst.env.sim
    robot_src, robot_dst = env_src.env.robots[0], env_dst.env.robots[0]
    target_eef = sim_src.data.get_body_xpos(robot_src.robot_model.eef_name).copy()
    target_quat = sim_src.data.get_body_xquat(robot_src.robot_model.eef_name).copy()

    n_copy = copy_scene_qpos(sim_src, sim_dst)
    if n_copy == 0:
        env_src.close()
        env_dst.close()
        raise RuntimeError(
            f"No shared object joints copied between {bddl_source.name} and {bddl_target.name}"
        )

    copy_gripper_qpos(sim_src, sim_dst)
    sim_dst.forward()

    q_arm = ik_arm_to_eef(
        sim_dst,
        robot_dst,
        target_eef,
        target_quat,
        robot_dst.robot_model.init_qpos,
    )
    sim_dst.data.qpos[robot_dst._ref_joint_pos_indexes] = q_arm
    sim_dst.forward()

    flat = np.array(sim_dst.get_state().flatten(), dtype=np.float64)
    env_src.close()
    env_dst.close()
    return flat

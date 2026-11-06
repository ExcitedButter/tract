"""Dynamic 7-point gripper mesh vertex IDs (Franka / UR5e / Sawyer) for LIBERO eval.

Mirrors ``lerobot_preprocess_cotracker._select_dynamic_ids_for_camera``: pick vertices
from gripper geoms (finger1/finger2/base) visible in each camera, not hardcoded Panda IDs.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def gripper_geom_group(geom_name: str) -> str:
    """Map MuJoCo geom name → finger1 | finger2 | base | other."""
    n = (geom_name or "").lower()
    if not n:
        return "other"
    # Panda: finger1_collision; UR5e: robotiq; Sawyer Rethink: l_finger / r_finger
    if "finger1" in n or "l_finger" in n or ("gripper" in n and "left" in n):
        return "finger1"
    if "finger2" in n or "r_finger" in n or ("gripper" in n and "right" in n):
        return "finger2"
    if "hand" in n or ("gripper" in n and "finger" not in n and "knuckle" not in n):
        return "base"
    if "gripper" in n or "finger" in n or "robotiq" in n:
        return "base"
    return "other"


def select_gripper_vertex_ids_for_camera(
    tracker_env: Any,
    *,
    camera_name: str,
    fallback_ids: list[int],
    img_hw: tuple[int, int],
    num_points: int = 7,
) -> list[int]:
    """Return ``num_points`` global mesh vertex IDs on the gripper for ``camera_name``."""
    if not hasattr(tracker_env, "vertex_mapping"):
        return list(fallback_ids)

    base_env = tracker_env
    while hasattr(base_env, "env") and not hasattr(base_env, "sim"):
        base_env = base_env.env
    if not hasattr(base_env, "sim"):
        return list(fallback_ids)
    model = base_env.sim.model

    candidates: dict[int, str] = {}
    for vid, mapping in tracker_env.vertex_mapping.items():
        geom_id = int(mapping.get("geom_id", -1))
        if geom_id < 0:
            continue
        geom_name = model.geom_id2name(geom_id)
        group = gripper_geom_group(geom_name)
        if group != "other":
            candidates[int(vid)] = group
    if not candidates:
        return list(fallback_ids)

    vids = list(candidates.keys())
    if hasattr(tracker_env, "_update_vertex_positions"):
        tracker_env._update_vertex_positions()
    tracked = tracker_env._get_tracked_vertices(camera_name, vids)
    id_to_xy = {
        int(i): p for i, p in zip(tracked["vertex_ids"].tolist(), tracked["vertices_2d"])
    }
    h, w = float(img_hw[0]), float(img_hw[1])
    rows = []
    for vid in vids:
        if vid not in id_to_xy:
            continue
        x = float(id_to_xy[vid][0]) / w
        y = float(id_to_xy[vid][1]) / h
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            rows.append({"vid": vid, "group": candidates[vid], "x": x, "y": y})

    def _pick_three(group_rows: list[dict]) -> list[dict]:
        if len(group_rows) <= 3:
            return group_rows
        ys = np.array([r["y"] for r in group_rows], dtype=np.float32)
        lo = int(np.argmin(ys))
        hi = int(np.argmax(ys))
        mid_target = float((ys[lo] + ys[hi]) * 0.5)
        mid = min(
            range(len(group_rows)),
            key=lambda i: abs(float(group_rows[i]["y"]) - mid_target),
        )
        idx: list[int] = []
        for i in (lo, mid, hi):
            if i not in idx:
                idx.append(i)
        for i in range(len(group_rows)):
            if len(idx) >= 3:
                break
            if i not in idx:
                idx.append(i)
        return [group_rows[i] for i in idx[:3]]

    f1 = _pick_three([r for r in rows if r["group"] == "finger1"])
    f2 = _pick_three([r for r in rows if r["group"] == "finger2"])
    b = [r for r in rows if r["group"] == "base"]
    if len(f1) < 2 or len(f2) < 2 or not b:
        return list(fallback_ids)
    mx = (float(np.mean([r["x"] for r in f1])) + float(np.mean([r["x"] for r in f2]))) / 2.0
    my = (float(np.mean([r["y"] for r in f1])) + float(np.mean([r["y"] for r in f2]))) / 2.0
    base = sorted(b, key=lambda r: abs(r["x"] - mx) + 0.5 * abs(r["y"] - my))[0]
    picked = (f1 + f2 + [base])[:num_points]
    picked_ids = [int(r["vid"]) for r in picked]
    return picked_ids if len(picked_ids) == num_points else list(fallback_ids)


def find_mesh_vertex_tracker(env: Any) -> Any | None:
    cur = env
    while cur is not None:
        if hasattr(cur, "agentview_vertex_ids") and hasattr(cur, "vertex_mapping"):
            return cur
        cur = getattr(cur, "env", None)
    return None


def apply_dynamic_gripper_vertex_ids(
    env: Any,
    *,
    fallback_agent_ids: list[int],
    fallback_wrist_ids: list[int],
    img_hw: tuple[int, int],
) -> tuple[list[int], list[int]]:
    """Update MeshVertexTracker IDs from gripper geoms; return (agent_ids, wrist_ids)."""
    tracker = find_mesh_vertex_tracker(env)
    if tracker is None:
        return list(fallback_agent_ids), list(fallback_wrist_ids)

    agent_ids = select_gripper_vertex_ids_for_camera(
        tracker,
        camera_name="agentview",
        fallback_ids=fallback_agent_ids,
        img_hw=img_hw,
    )
    wrist_ids = select_gripper_vertex_ids_for_camera(
        tracker,
        camera_name="robot0_eye_in_hand",
        fallback_ids=fallback_wrist_ids,
        img_hw=img_hw,
    )
    tracker.agentview_vertex_ids = agent_ids
    tracker.eye_in_hand_vertex_ids = wrist_ids
    return agent_ids, wrist_ids


def refresh_tracked_vertices_obs(env: Any, obs: dict[str, Any]) -> dict[str, Any]:
    """Re-fill tracked vertex 2D fields in ``obs`` after vertex ID update."""
    tracker = find_mesh_vertex_tracker(env)
    if tracker is None:
        return obs
    tracker._update_vertex_positions()
    if tracker.agentview_vertex_ids:
        d = tracker._get_tracked_vertices("agentview", tracker.agentview_vertex_ids)
        obs["agentview_tracked_vertices_2d"] = d["vertices_2d"]
        obs["agentview_tracked_vertices_3d"] = d["vertices_3d"]
        obs["agentview_tracked_vertex_ids"] = d["vertex_ids"]
    if tracker.eye_in_hand_vertex_ids:
        d = tracker._get_tracked_vertices("robot0_eye_in_hand", tracker.eye_in_hand_vertex_ids)
        obs["eye_in_hand_tracked_vertices_2d"] = d["vertices_2d"]
        obs["eye_in_hand_tracked_vertices_3d"] = d["vertices_3d"]
        obs["eye_in_hand_tracked_vertex_ids"] = d["vertex_ids"]
    return obs

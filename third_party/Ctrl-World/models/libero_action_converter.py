"""
Convert LIBERO OSC_POSE delta actions to absolute EE states for Ctrl-World.

Two modes:
  1. Analytical (approximate): uses OSC scaling rules directly
  2. Learned MLP adapter (accurate): trained on 271K transitions, <1.6cm max error over 16 steps

Usage:
    from models.libero_action_converter import LiberoActionConverter

    converter = LiberoActionConverter()

    # From RLDS 8D observation → 7D initial state
    state_7d = converter.initial_state_from_obs(state_8d)

    # Load trained MLP adapter for accurate conversion
    converter.load_adapter("checkpoints/ctrl_world/libero_action_adapter.pt")  # repo-level checkpoints/

    # Single step: policy delta → absolute state (uses MLP if loaded, else analytical)
    next_state = converter.step(current_state, delta_action)

    # Full trajectory: initial state + action sequence → state sequence
    states = converter.trajectory(initial_state, actions)
"""

import os
import numpy as np
from scipy.spatial.transform import Rotation

import torch
import torch.nn as nn


class LiberoActionConverter:
    """Convert LIBERO OSC_POSE delta actions to absolute EE states.

    The learned adapter predicts residuals: delta = MLP(state, action),
    then next_state = state + delta. Trained on all 4 LIBERO suites
    (271K transitions), achieving 2.7mm mean / 1.6cm max error over
    16-step rollouts.
    """

    GRIPPER_MAX = 0.08

    def __init__(self, device="cpu"):
        self._adapter = None
        self._norm = None
        self._device = torch.device(device)

    def initial_state_from_obs(self, state_8d: np.ndarray) -> np.ndarray:
        """Extract 7D Ctrl-World state from RLDS 8D observation.

        RLDS state: [ee_pos(3), ee_ori_axis_angle(3), gripper_L(1), gripper_R(1)]
        Output:     [ee_pos(3), ee_ori_axis_angle(3), gripper_width(1)]
        """
        state_8d = np.asarray(state_8d, dtype=np.float64)
        gripper_width = float(state_8d[6] - state_8d[7])
        return np.concatenate([state_8d[:6], [gripper_width]])

    def load_adapter(self, path: str, device: str = None):
        """Load trained MLP adapter weights (LIBERO converter checkpoint)."""
        if device is not None:
            self._device = torch.device(device)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if not isinstance(ckpt, dict):
            raise ValueError(f"Expected dict checkpoint, got {type(ckpt)} from {path}")

        # Droid joint-vel adapter (models/action_adapter/model2_15_9.pth) is a different format.
        if "model_state_dict" not in ckpt and any(k.startswith("net.") for k in ckpt):
            raise ValueError(
                f"{path} looks like a Droid action_adapter checkpoint (net.* keys), not a "
                "LIBERO converter. Use checkpoints/ctrl_world/libero_action_adapter.pt"
            )

        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint {path} missing 'model_state_dict'. Keys: {sorted(ckpt.keys())}"
            )

        state_dict = ckpt["model_state_dict"]
        in_dim = int(ckpt["in_dim"])
        out_dim = int(ckpt["out_dim"])
        hidden = int(ckpt["hidden_dim"])

        if "n_layers" in ckpt:
            n_layers = int(ckpt["n_layers"])
        else:
            weight_keys = sorted(k for k in state_dict if k.endswith(".weight"))
            n_layers = len(weight_keys)
            if n_layers < 1:
                raise ValueError(f"Cannot infer MLP depth from {path}")

        layers = []
        d = in_dim
        for i in range(n_layers):
            out_d = hidden if i < n_layers - 1 else out_dim
            layers.append(nn.Linear(d, out_d))
            if i < n_layers - 1:
                layers.append(nn.ReLU())
            d = out_d
        self._adapter = nn.Sequential(*layers).to(self._device)
        self._adapter.load_state_dict(state_dict)
        self._adapter.eval()

        for key in ("X_mean", "X_std", "Y_mean", "Y_std"):
            if key not in ckpt:
                raise KeyError(f"Checkpoint {path} missing normalization key '{key}'")

        self._norm = {
            "X_mean": torch.from_numpy(np.asarray(ckpt["X_mean"], dtype=np.float32)).to(self._device),
            "X_std": torch.from_numpy(np.asarray(ckpt["X_std"], dtype=np.float32)).to(self._device),
            "Y_mean": torch.from_numpy(np.asarray(ckpt["Y_mean"], dtype=np.float32)).to(self._device),
            "Y_std": torch.from_numpy(np.asarray(ckpt["Y_std"], dtype=np.float32)).to(self._device),
        }

    @property
    def has_adapter(self) -> bool:
        return self._adapter is not None

    def step(self, current_state_7d: np.ndarray, delta_action_7d: np.ndarray) -> np.ndarray:
        """Single step: current_state + delta → next_state.

        Uses MLP adapter if loaded, otherwise falls back to analytical.
        """
        if self._adapter is not None:
            return self._step_mlp(current_state_7d, delta_action_7d)
        return self._step_analytical(current_state_7d, delta_action_7d)

    def trajectory(self, initial_state_7d: np.ndarray, delta_actions: np.ndarray) -> np.ndarray:
        """Convert action sequence to absolute state sequence.

        Returns: (T+1, 7) states where states[0] = initial_state.
        """
        initial_state_7d = np.asarray(initial_state_7d, dtype=np.float64)
        delta_actions = np.asarray(delta_actions, dtype=np.float64)

        states = np.zeros((len(delta_actions) + 1, 7), dtype=np.float64)
        states[0] = initial_state_7d
        for t in range(len(delta_actions)):
            states[t + 1] = self.step(states[t], delta_actions[t])
        return states

    def _step_mlp(self, current_state_7d: np.ndarray, delta_action_7d: np.ndarray) -> np.ndarray:
        inp = np.concatenate([
            np.asarray(current_state_7d, dtype=np.float32),
            np.asarray(delta_action_7d, dtype=np.float32),
        ])
        inp_t = (torch.from_numpy(inp).to(self._device) - self._norm["X_mean"]) / self._norm["X_std"]
        with torch.no_grad():
            delta = self._adapter(inp_t.unsqueeze(0)).squeeze(0)
        delta_np = (delta * self._norm["Y_std"] + self._norm["Y_mean"]).cpu().numpy()
        return np.asarray(current_state_7d, dtype=np.float64) + delta_np.astype(np.float64)

    def _step_analytical(self, current_state_7d: np.ndarray, delta_action_7d: np.ndarray) -> np.ndarray:
        """Analytical fallback using calibrated OSC scaling (less accurate)."""
        POS_SCALE = np.array([0.01052, 0.01209, 0.01122])
        ORI_SCALE = np.array([0.5, 0.5, 0.5])

        current_state_7d = np.asarray(current_state_7d, dtype=np.float64)
        delta_action_7d = np.asarray(delta_action_7d, dtype=np.float64)

        next_pos = current_state_7d[:3] + delta_action_7d[:3] * POS_SCALE
        current_rot = Rotation.from_rotvec(current_state_7d[3:6])
        delta_rot = Rotation.from_rotvec(delta_action_7d[3:6] * ORI_SCALE)
        next_ori = (delta_rot * current_rot).as_rotvec()
        gripper_width = (1.0 - delta_action_7d[6]) / 2.0 * self.GRIPPER_MAX

        return np.array([*next_pos, *next_ori, gripper_width])

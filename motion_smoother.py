"""
motion_smoother.py
------------------
Per-person, per-joint Exponential Moving Average smoother with
confidence-weighted blending and missing-joint interpolation.

Each person has an isolated state, so multi-person tracking stays stable
when people enter or leave the frame.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class EMASmoother:
    """Confidence-weighted EMA over (N, 4) landmark arrays."""

    def __init__(
        self,
        base_alpha: float = 0.5,
        min_visibility: float = 0.5,
    ) -> None:
        self.base_alpha = base_alpha
        self.min_visibility = min_visibility
        self._state: Dict[int, np.ndarray] = {}

    def reset(self, person_id: Optional[int] = None) -> None:
        if person_id is None:
            self._state.clear()
        else:
            self._state.pop(person_id, None)

    def smooth(self, person_id: int, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply per-joint EMA. For each joint:
          - if its visibility is below the threshold and we have history,
            carry forward the previous smoothed position;
          - otherwise blend new vs. previous with alpha proportional to confidence.
        """
        prev = self._state.get(person_id)
        if prev is None or prev.shape != landmarks.shape:
            self._state[person_id] = landmarks.copy()
            return landmarks.copy()

        out = landmarks.copy()
        for i in range(landmarks.shape[0]):
            vis = float(landmarks[i, 3])
            if vis < self.min_visibility:
                # Carry forward; decay confidence so we know it's interpolated
                out[i, :3] = prev[i, :3]
                out[i, 3] = max(vis, prev[i, 3] * 0.9)
            else:
                # Higher visibility => trust the new sample more
                conf = max(0.0, min(1.0, vis))
                alpha = self.base_alpha * (0.4 + 0.6 * conf)
                out[i, :3] = (1.0 - alpha) * prev[i, :3] + alpha * landmarks[i, :3]
                out[i, 3] = 0.5 * (prev[i, 3] + vis)

        self._state[person_id] = out
        return out


class VelocityCalculator:
    """Computes per-joint velocity (units / s) from successive smoothed frames."""

    def __init__(self) -> None:
        self._prev: Dict[int, np.ndarray] = {}
        self._prev_t: Dict[int, float] = {}

    def reset(self, person_id: Optional[int] = None) -> None:
        if person_id is None:
            self._prev.clear()
            self._prev_t.clear()
        else:
            self._prev.pop(person_id, None)
            self._prev_t.pop(person_id, None)

    def update(self, person_id: int, landmarks: np.ndarray, t: float) -> np.ndarray:
        """Return a (N, 3) velocity array. First call returns zeros."""
        prev = self._prev.get(person_id)
        prev_t = self._prev_t.get(person_id)
        self._prev[person_id] = landmarks.copy()
        self._prev_t[person_id] = t

        if prev is None or prev_t is None:
            return np.zeros((landmarks.shape[0], 3), dtype=np.float32)

        dt = max(1e-3, t - prev_t)
        return ((landmarks[:, :3] - prev[:, :3]) / dt).astype(np.float32)

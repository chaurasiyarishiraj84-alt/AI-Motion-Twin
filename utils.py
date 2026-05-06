"""
utils.py
--------
Small helper functions shared across the AI Motion Twin pipeline:
geometric math, normalization, and timing utilities.
"""

from __future__ import annotations

import time
from typing import Iterable, Tuple

import numpy as np


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the midpoint between two 2D or 3D points."""
    return (a + b) * 0.5


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points (any dimensionality)."""
    return float(np.linalg.norm(a - b))


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Interior angle at point B formed by the polyline A-B-C, in degrees.
    Returns 180 when the three points are collinear, 0 when they overlap.
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom == 0:
        return 180.0
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def normalize_landmarks(
    landmarks: np.ndarray,
    reference_indices: Tuple[int, int],
    target_size: float = 0.25,
) -> np.ndarray:
    """
    Scale a landmark array so the distance between the two reference indices
    (typically left/right shoulder) equals ``target_size`` in normalized
    coordinates. The center is preserved. Useful for camera-distance
    invariance when comparing skeletons.

    Parameters
    ----------
    landmarks : (N, D) array of normalized landmark positions (0..1).
    reference_indices : (i, j) indices of the two reference joints.
    target_size : desired distance between the reference joints after scaling.
    """
    if landmarks.size == 0:
        return landmarks
    i, j = reference_indices
    ref_dist = distance(landmarks[i, :2], landmarks[j, :2])
    if ref_dist <= 1e-3:
        return landmarks
    scale = target_size / ref_dist
    center = (landmarks[i] + landmarks[j]) * 0.5
    out = landmarks.copy()
    out[:, : center.shape[0]] = (landmarks[:, : center.shape[0]] - center) * scale + center
    return out


class FpsCounter:
    """Tiny rolling FPS counter that updates once per second."""

    def __init__(self) -> None:
        self._last = time.perf_counter()
        self._frames = 0
        self._fps = 0.0

    def tick(self) -> float:
        self._frames += 1
        now = time.perf_counter()
        dt = now - self._last
        if dt >= 1.0:
            self._fps = self._frames / dt
            self._frames = 0
            self._last = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps


def chunk_iterable(items: Iterable, size: int) -> Iterable:
    """Yield successive ``size``-sized chunks from ``items``."""
    bucket = []
    for it in items:
        bucket.append(it)
        if len(bucket) == size:
            yield bucket
            bucket = []
    if bucket:
        yield bucket

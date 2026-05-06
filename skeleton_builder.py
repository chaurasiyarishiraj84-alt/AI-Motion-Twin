"""
skeleton_builder.py
-------------------
Turns a (33, 4) MediaPipe landmark array into a richer 36-point skeleton:
the 33 native joints plus three derived joints (NECK, MID_HIP, SPINE) and
the bone topology used by the renderer.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils import midpoint

# ---------- Index map ------------------------------------------------------

# Native MediaPipe indices (0..32)
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Derived indices (appended)
NECK = 33
MID_HIP = 34
SPINE = 35

TOTAL_LANDMARKS = 36

# ---------- Bone topology --------------------------------------------------

# Native MediaPipe pose connections
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Shoulders + arms
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso sides + hips
    (11, 23), (12, 24), (23, 24),
    # Legs
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]

# Bones added by the derived joints — gives the skeleton a real spine
DERIVED_CONNECTIONS: List[Tuple[int, int]] = [
    (NOSE, NECK),
    (NECK, SPINE),
    (SPINE, MID_HIP),
    (NECK, LEFT_SHOULDER),
    (NECK, RIGHT_SHOULDER),
    (MID_HIP, LEFT_HIP),
    (MID_HIP, RIGHT_HIP),
]

ALL_CONNECTIONS: List[Tuple[int, int]] = POSE_CONNECTIONS + DERIVED_CONNECTIONS


def derive_joints(landmarks_33: np.ndarray) -> np.ndarray:
    """
    Append NECK, MID_HIP and SPINE to a (33, 4) landmark array.
    Returns a (36, 4) array. Visibility for each derived joint is the
    minimum visibility of its parents (a conservative estimate).
    """
    if landmarks_33.shape[0] != 33:
        raise ValueError(f"Expected 33 landmarks, got {landmarks_33.shape[0]}")

    out = np.zeros((TOTAL_LANDMARKS, landmarks_33.shape[1]), dtype=landmarks_33.dtype)
    out[:33] = landmarks_33

    ls, rs = landmarks_33[LEFT_SHOULDER], landmarks_33[RIGHT_SHOULDER]
    lh, rh = landmarks_33[LEFT_HIP], landmarks_33[RIGHT_HIP]

    neck_xyz = midpoint(ls[:3], rs[:3])
    midhip_xyz = midpoint(lh[:3], rh[:3])
    spine_xyz = midpoint(neck_xyz, midhip_xyz)

    out[NECK, :3] = neck_xyz
    out[NECK, 3] = min(ls[3], rs[3])

    out[MID_HIP, :3] = midhip_xyz
    out[MID_HIP, 3] = min(lh[3], rh[3])

    out[SPINE, :3] = spine_xyz
    out[SPINE, 3] = min(out[NECK, 3], out[MID_HIP, 3])

    return out


def body_center(landmarks: np.ndarray) -> np.ndarray:
    """Center of mass proxy: average of shoulders + hips midpoint (xy only)."""
    ref = landmarks[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], :2]
    return ref.mean(axis=0)


def shoulder_width(landmarks: np.ndarray) -> float:
    """Pixel distance between left and right shoulders."""
    return float(np.linalg.norm(landmarks[LEFT_SHOULDER, :2] - landmarks[RIGHT_SHOULDER, :2]))

"""
renderer.py
-----------
Draws the stick figure skeleton(s) and composes the side-by-side view
(original video on the left, stick figure canvas on the right).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from skeleton_builder import ALL_CONNECTIONS, TOTAL_LANDMARKS

# A short, distinct color palette per person ID (BGR)
PERSON_COLORS: List[Tuple[int, int, int]] = [
    (255, 220, 0),     # cyan-ish
    (0, 200, 255),     # amber
    (180, 80, 255),    # magenta
    (80, 255, 120),    # green
    (255, 120, 80),    # blue
]

JOINT_RADIUS = 5
DERIVED_JOINT_RADIUS = 7
BONE_THICKNESS = 3
MIN_VISIBILITY = 0.5


def color_for(person_id: int) -> Tuple[int, int, int]:
    return PERSON_COLORS[(person_id - 1) % len(PERSON_COLORS)]


def draw_skeleton(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    person_id: int = 1,
    show_label: bool = True,
) -> None:
    """
    Draw a single person's skeleton onto ``canvas`` in place.

    landmarks : (36, 4) array of (x_px, y_px, z, visibility).
    """
    if landmarks.shape[0] != TOTAL_LANDMARKS:
        return

    color = color_for(person_id)
    # Bones first
    for a, b in ALL_CONNECTIONS:
        pa = landmarks[a]
        pb = landmarks[b]
        if pa[3] < MIN_VISIBILITY or pb[3] < MIN_VISIBILITY:
            continue
        cv2.line(
            canvas,
            (int(pa[0]), int(pa[1])),
            (int(pb[0]), int(pb[1])),
            color,
            BONE_THICKNESS,
            lineType=cv2.LINE_AA,
        )

    # Joints
    for i in range(TOTAL_LANDMARKS):
        pt = landmarks[i]
        if pt[3] < MIN_VISIBILITY:
            continue
        r = DERIVED_JOINT_RADIUS if i >= 33 else JOINT_RADIUS
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), r, color, -1, lineType=cv2.LINE_AA)

    if show_label:
        # Label near the head (NOSE)
        nose = landmarks[0]
        if nose[3] >= MIN_VISIBILITY:
            cv2.putText(
                canvas,
                f"P{person_id}",
                (int(nose[0]) + 10, int(nose[1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                lineType=cv2.LINE_AA,
            )


def draw_heatmap(
    canvas: np.ndarray,
    landmarks: np.ndarray,
    velocities: np.ndarray,
    intensity: float = 1.0,
) -> None:
    """Splat motion-energy blobs at each joint scaled by velocity magnitude."""
    overlay = np.zeros_like(canvas)
    for i in range(min(landmarks.shape[0], velocities.shape[0])):
        if landmarks[i, 3] < MIN_VISIBILITY:
            continue
        speed = float(np.linalg.norm(velocities[i, :2]))
        if speed < 5:
            continue
        radius = int(min(80, 8 + speed * 0.3))
        cv2.circle(
            overlay,
            (int(landmarks[i, 0]), int(landmarks[i, 1])),
            radius,
            (0, 80, 255),
            -1,
            lineType=cv2.LINE_AA,
        )
    cv2.addWeighted(canvas, 1.0, overlay, intensity * 0.4, 0, dst=canvas)


def make_blank_canvas(width: int, height: int) -> np.ndarray:
    """Dark canvas with a faint grid for the stick figure side."""
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    grid_color = (28, 28, 28)
    step = 40
    for x in range(0, width, step):
        cv2.line(canvas, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, step):
        cv2.line(canvas, (0, y), (width, y), grid_color, 1)
    return canvas


def compose_side_by_side(
    original: np.ndarray,
    stick_canvas: np.ndarray,
    fps: float = 0.0,
    person_count: int = 0,
    action: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Concatenate the original frame and the stick canvas horizontally,
    matching their heights, and draw a small HUD on top.
    """
    h = max(original.shape[0], stick_canvas.shape[0])

    def fit(img: np.ndarray) -> np.ndarray:
        if img.shape[0] == h:
            return img
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    left = fit(original)
    right = fit(stick_canvas)
    composite = np.hstack([left, right])

    # HUD
    pad = 12
    cv2.rectangle(composite, (0, 0), (composite.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        composite,
        f"AI MOTION TWIN  |  FPS {fps:5.1f}  |  PEOPLE {person_count}",
        (pad, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 220, 0),
        1,
        cv2.LINE_AA,
    )
    if action:
        cv2.putText(
            composite,
            f"ACTION: {action}",
            (composite.shape[1] - 280, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    if metrics:
        y = 56
        for k, v in metrics.items():
            cv2.putText(
                composite,
                f"{k.upper():<10s} {v:6.1f}",
                (pad, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            y += 20

    # Divider
    cv2.line(
        composite,
        (left.shape[1], 0),
        (left.shape[1], composite.shape[0]),
        (60, 60, 60),
        1,
    )
    return composite


def draw_skeletons(
    canvas: np.ndarray,
    skeletons: Sequence[np.ndarray],
    person_ids: Sequence[int],
) -> None:
    for lm, pid in zip(skeletons, person_ids):
        draw_skeleton(canvas, lm, person_id=pid)

"""
pose_detector.py
----------------
Wraps MediaPipe Pose for single- and multi-person detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mediapipe as mp
import numpy as np


mp_pose: Any = mp.solutions.pose

POSE_LANDMARK_COUNT = 33


@dataclass
class PoseResult:
    """One detected person."""
    landmarks: np.ndarray
    person_id: int = 0


class PoseDetector:
    """High-confidence single-person pose detector."""

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        model_complexity: int = 1,
    ) -> None:
        
        self._pose: Any = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, rgb_frame: np.ndarray) -> Optional[np.ndarray]:
        """Return a (33, 4) landmark array or None."""
        results = self._pose.process(rgb_frame)
        if not results.pose_landmarks:
            return None
        h, w = rgb_frame.shape[:2]
        out = np.zeros((POSE_LANDMARK_COUNT, 4), dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            out[i, 0] = lm.x * w
            out[i, 1] = lm.y * h
            out[i, 2] = lm.z * w
            out[i, 3] = lm.visibility
        return out

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MultiPersonPoseDetector:
    """Naive multi-person pose detector via successive masking."""

    def __init__(self, max_persons: int = 3, min_confidence: float = 0.6) -> None:
        self.max_persons    = max(1, max_persons)
        self.min_confidence = min_confidence
        self._detectors: List[PoseDetector] = [
            PoseDetector(
                min_detection_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
            for _ in range(self.max_persons)
        ]

    def detect(self, rgb_frame: np.ndarray) -> List[np.ndarray]:
        """Return up to max_persons (33, 4) landmark arrays."""
        results: List[np.ndarray] = []
        working = rgb_frame.copy()
        h, w = working.shape[:2]

        for det in self._detectors:
            lm = det.detect(working)
            if lm is None:
                break
            if float(lm[:, 3].mean()) < self.min_confidence:
                break
            results.append(lm)
            x1 = max(0, int(lm[:, 0].min() - 20))
            y1 = max(0, int(lm[:, 1].min() - 20))
            x2 = min(w, int(lm[:, 0].max() + 20))
            y2 = min(h, int(lm[:, 1].max() + 20))
            working[y1:y2, x1:x2] = 0

        return results

    def close(self) -> None:
        for det in self._detectors:
            det.close()

    def __enter__(self) -> "MultiPersonPoseDetector":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class PersonTracker:
    """Greedy centroid-distance tracker that assigns stable integer IDs."""

    def __init__(self, match_radius_px: float = 200.0) -> None:
        self._next_id      = 1
        # FIX 2 — Dict[int, ...] from typing works on Python 3.7+.
        # Lowercase dict[int, ...] requires Python 3.9+.
        self._tracks: Dict[int, np.ndarray] = {}
        self._match_radius = match_radius_px

    @staticmethod
    def _centroid(lm: np.ndarray) -> np.ndarray:
        return lm[[11, 12, 23, 24], :2].mean(axis=0)

    def assign(self, detections: Sequence[np.ndarray]) -> List[int]:
        ids: List[Optional[int]] = [None] * len(detections)
        used: set = set()

        cands: List[Tuple[float, int, int]] = []
        for di, lm in enumerate(detections):
            c = self._centroid(lm)
            for tid, prev_c in self._tracks.items():
                d = float(np.linalg.norm(c - prev_c))
                if d <= self._match_radius:
                    cands.append((d, di, tid))
        cands.sort()

        for _d, di, tid in cands:
            if ids[di] is not None or tid in used:
                continue
            ids[di] = tid
            used.add(tid)

        for di in range(len(detections)):
            if ids[di] is None:
                ids[di] = self._next_id
                self._next_id += 1

        new_tracks: Dict[int, np.ndarray] = {}
        for di, lm in enumerate(detections):
            pid = ids[di]           # pid: Optional[int]
            if pid is not None:     # Pylance narrows pid → int here
                new_tracks[pid] = self._centroid(lm)
        self._tracks = new_tracks

        # Same narrowing pattern for the return list
        out: List[int] = []
        for pid in ids:
            if pid is not None:
                out.append(pid)
        return out

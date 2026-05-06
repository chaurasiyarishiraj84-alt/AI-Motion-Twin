"""
input_handler.py
----------------
Wraps OpenCV's VideoCapture for both webcam and file input.
Handles graceful open/close, optional resize, and exposes a simple
iterator interface so the rest of the pipeline doesn't have to care
about the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameSource:
    """A unified frame iterator over a webcam index or a video file path."""

    source: int | str
    width: Optional[int] = 1280
    height: Optional[int] = 720

    def __post_init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        self._cap = cap

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 0.0
        return float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def size(self) -> Tuple[int, int]:
        if self._cap is None:
            return (0, 0)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def frames(self, mirror: bool = False) -> Iterator[np.ndarray]:
        """Yield BGR frames until the source ends or open() hasn't been called."""
        if self._cap is None:
            raise RuntimeError("FrameSource not opened. Call open() or use 'with'.")
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            if mirror:
                frame = cv2.flip(frame, 1)
            yield frame


def resize_keep_aspect(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale a frame to ``max_width`` preserving aspect ratio."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

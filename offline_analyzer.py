"""
offline_analyzer.py
-------------------
Batch-process a folder of videos and emit an ML-ready dataset.

For every video in the input folder this module produces:
  dataset/<video_name>.csv     one row per frame, all features below
  dataset/metadata.json        per-video summary (fps, frames, source path, ...)

Per-frame columns
-----------------
  frame_id, timestamp, video_id
  <joint>_x, <joint>_y, <joint>_z, <joint>_v        (33 native + 3 derived = 36 joints)
  <joint>_norm_x, <joint>_norm_y                    (hip-centered, shoulder-scaled)
  <joint>_vx, <joint>_vy                            (per-joint velocity, px/s and norm/s)
  <joint>_ax, <joint>_ay                            (per-joint acceleration)
  left_elbow_angle, right_elbow_angle,
  left_knee_angle, right_knee_angle,
  left_shoulder_angle, right_shoulder_angle         (degrees)
  energy_score, arm_symmetry_score, leg_symmetry_score
  action_label, action_confidence

Reuses the project's existing pose detector, skeleton builder and EMA smoother
so dataset rows match what the realtime app sees frame-for-frame.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - friendly error
    raise SystemExit(
        "pandas is required for offline analysis. Install with `pip install pandas`."
    ) from exc

from input_handler import FrameSource, resize_keep_aspect
from main import ActionClassifier  # reuse the same rule-based classifier
from motion_smoother import EMASmoother
from pose_detector import PoseDetector
from skeleton_builder import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    MID_HIP,
    NECK,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    SPINE,
    TOTAL_LANDMARKS,
    derive_joints,
    shoulder_width,
)
from utils import angle_deg

# --------------------------------------------------------------------------
# Joint naming — index → snake_case label used in the CSV header
# --------------------------------------------------------------------------

JOINT_NAMES: List[str] = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
    "neck", "mid_hip", "spine",
]
assert len(JOINT_NAMES) == TOTAL_LANDMARKS

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

def normalize_to_hip(landmarks: np.ndarray) -> np.ndarray:
    """
    Hip-centered, shoulder-scaled normalization.

    Returns a (N, 2) array of x/y in body-relative units (1 unit ~ shoulder width).
    """
    out = np.zeros((landmarks.shape[0], 2), dtype=np.float32)
    sw = shoulder_width(landmarks)
    if sw < 1e-3:
        return out
    origin = landmarks[MID_HIP, :2]
    out[:] = (landmarks[:, :2] - origin) / sw
    return out


def joint_angles_deg(lm: np.ndarray) -> Dict[str, float]:
    """Six standard joint angles in degrees, 180 if not computable."""
    a = lambda p, q, r: angle_deg(lm[p, :2], lm[q, :2], lm[r, :2])
    return {
        "left_elbow_angle":     a(LEFT_SHOULDER, LEFT_ELBOW,    LEFT_WRIST),
        "right_elbow_angle":    a(RIGHT_SHOULDER, RIGHT_ELBOW,  RIGHT_WRIST),
        "left_knee_angle":      a(LEFT_HIP,      LEFT_KNEE,     LEFT_ANKLE),
        "right_knee_angle":     a(RIGHT_HIP,     RIGHT_KNEE,    RIGHT_ANKLE),
        "left_shoulder_angle":  a(LEFT_ELBOW,    LEFT_SHOULDER, LEFT_HIP),
        "right_shoulder_angle": a(RIGHT_ELBOW,   RIGHT_SHOULDER, RIGHT_HIP),
    }


def symmetry_scores(angles: Dict[str, float]) -> Dict[str, float]:
    arm_diff = abs(angles["left_elbow_angle"] - angles["right_elbow_angle"])
    leg_diff = abs(angles["left_knee_angle"] - angles["right_knee_angle"])
    return {
        "arm_symmetry_score": max(0.0, 100.0 - arm_diff),
        "leg_symmetry_score": max(0.0, 100.0 - leg_diff),
    }


def energy_score(velocity: np.ndarray) -> float:
    """Sum of squared 2D speeds across all joints."""
    speeds = np.linalg.norm(velocity[:, :2], axis=1)
    return float(np.sum(speeds ** 2))


# --------------------------------------------------------------------------
# Per-video pipeline
# --------------------------------------------------------------------------

@dataclass
class VideoStats:
    video_id: str
    source_path: str
    fps: float
    frame_count: int
    detected_frames: int
    duration_s: float


def _build_row(
    frame_id: int,
    timestamp: float,
    video_id: str,
    smoothed: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    norm: np.ndarray,
    angles: Dict[str, float],
    sym: Dict[str, float],
    energy: float,
    action_label: str,
    action_conf: float,
) -> Dict[str, float | str | int]:
    row: Dict[str, float | str | int] = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "video_id": video_id,
    }
    for i, name in enumerate(JOINT_NAMES):
        row[f"{name}_x"] = float(smoothed[i, 0])
        row[f"{name}_y"] = float(smoothed[i, 1])
        row[f"{name}_z"] = float(smoothed[i, 2])
        row[f"{name}_v"] = float(smoothed[i, 3])
        row[f"{name}_norm_x"] = float(norm[i, 0])
        row[f"{name}_norm_y"] = float(norm[i, 1])
        row[f"{name}_vx"] = float(velocity[i, 0])
        row[f"{name}_vy"] = float(velocity[i, 1])
        row[f"{name}_ax"] = float(acceleration[i, 0])
        row[f"{name}_ay"] = float(acceleration[i, 1])
    row.update({k: float(v) for k, v in angles.items()})
    row.update({k: float(v) for k, v in sym.items()})
    row["energy_score"] = float(energy)
    row["action_label"] = action_label
    row["action_confidence"] = float(action_conf)
    return row


def analyze_video(
    video_path: Path,
    output_csv: Path,
    max_width: int = 960,
    min_visibility: float = 0.5,
    smoother_alpha: float = 0.5,
    progress: bool = True,
) -> VideoStats:
    """Run pose + feature extraction over one video and write a CSV."""
    video_id = video_path.stem
    rows: List[Dict[str, float | str | int]] = []
    detected = 0

    smoother = EMASmoother(base_alpha=smoother_alpha, min_visibility=min_visibility)
    classifier = ActionClassifier()

    prev_smoothed: Optional[np.ndarray] = None
    prev_velocity: Optional[np.ndarray] = None
    prev_t: Optional[float] = None

    with FrameSource(str(video_path), width=None, height=None) as src, PoseDetector() as pose:
        src_fps = src.fps if src.fps > 0 else 30.0
        frame_id = 0

        for frame_bgr in src.frames(mirror=False):
            frame_bgr = resize_keep_aspect(frame_bgr, max_width)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            timestamp = frame_id / src_fps

            lm33 = pose.detect(rgb)
            if lm33 is None:
                frame_id += 1
                if progress and frame_id % 30 == 0:
                    print(f"  [{video_id}] frame {frame_id}  (no detection)", end="\r")
                continue

            derived = derive_joints(lm33)
            smoothed = smoother.smooth(0, derived)

            # Velocity & acceleration in pixels (and pixels/s²) using real dt
            if prev_smoothed is not None and prev_t is not None:
                dt = max(1e-3, timestamp - prev_t)
                velocity = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)
                velocity[:, :3] = (smoothed[:, :3] - prev_smoothed[:, :3]) / dt
            else:
                velocity = np.zeros((TOTAL_LANDMARKS, 3), dtype=np.float32)

            if prev_velocity is not None and prev_t is not None:
                dt = max(1e-3, timestamp - prev_t)
                acceleration = (velocity - prev_velocity) / dt
            else:
                acceleration = np.zeros_like(velocity)

            angles = joint_angles_deg(smoothed)
            sym = symmetry_scores(angles)
            energy = energy_score(velocity)
            norm = normalize_to_hip(smoothed)
            action = classifier.classify(smoothed, velocity)
            # Smooth the buffer is internal; emit a confidence proxy from buffer mode share
            mode_share = (
                sum(1 for x in classifier._buf if x == action) / max(1, len(classifier._buf))
            )

            rows.append(
                _build_row(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    video_id=video_id,
                    smoothed=smoothed,
                    velocity=velocity,
                    acceleration=acceleration,
                    norm=norm,
                    angles=angles,
                    sym=sym,
                    energy=energy,
                    action_label=action,
                    action_conf=mode_share,
                )
            )

            prev_smoothed = smoothed
            prev_velocity = velocity
            prev_t = timestamp
            detected += 1
            frame_id += 1

            if progress and frame_id % 30 == 0:
                print(f"  [{video_id}] frame {frame_id}  detected={detected}", end="\r")

    if progress:
        print()  # end the in-place line

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    return VideoStats(
        video_id=video_id,
        source_path=str(video_path),
        fps=float(src_fps),
        frame_count=int(frame_id),
        detected_frames=int(detected),
        duration_s=float(frame_id / src_fps),
    )


# --------------------------------------------------------------------------
# Folder driver
# --------------------------------------------------------------------------

def iter_videos(folder: Path) -> Iterable[Path]:
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def analyze_folder(
    input_dir: Path,
    output_dir: Path,
    max_width: int = 960,
    merge: bool = False,
) -> List[VideoStats]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    videos = list(iter_videos(input_dir))
    if not videos:
        print(f"No videos found in {input_dir} (looked for {sorted(VIDEO_EXTS)}).")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    stats: List[VideoStats] = []
    print(f"Processing {len(videos)} video(s) → {output_dir}")
    for v in videos:
        out_csv = output_dir / f"{v.stem}.csv"
        print(f"• {v.name}")
        s = analyze_video(v, out_csv, max_width=max_width)
        stats.append(s)
        print(f"  saved {out_csv.name}  ({s.detected_frames}/{s.frame_count} frames @ {s.fps:.1f} fps)")

    metadata = {
        "input_folder": str(input_dir),
        "output_folder": str(output_dir),
        "video_count": len(stats),
        "videos": [s.__dict__ for s in stats],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote metadata.json")

    if merge and stats:
        merged_path = output_dir / "merged.csv"
        first = True
        for s in stats:
            df = pd.read_csv(output_dir / f"{s.video_id}.csv")
            df.to_csv(merged_path, mode="w" if first else "a", header=first, index=False)
            first = False
        print(f"Wrote merged dataset → {merged_path.name}")

    return stats


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI Motion Twin — offline batch analyzer "
                    "(folder of videos → ML-ready CSV dataset)."
    )
    p.add_argument("--input", required=True, help="Folder containing input videos.")
    p.add_argument("--output", default="dataset", help="Output dataset folder. Default: ./dataset")
    p.add_argument("--max-width", type=int, default=960,
                   help="Resize frames so width ≤ this many pixels.")
    p.add_argument("--merge", action="store_true",
                   help="Also write a single merged.csv across all videos.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        analyze_folder(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            max_width=args.max_width,
            merge=args.merge,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

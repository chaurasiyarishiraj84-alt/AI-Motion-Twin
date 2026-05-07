from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np

from input_handler import FrameSource, resize_keep_aspect
from motion_smoother import EMASmoother, VelocityCalculator
from pose_detector import MultiPersonPoseDetector, PersonTracker, PoseDetector
from renderer import (
    compose_side_by_side,
    draw_heatmap,
    draw_skeletons,
    make_blank_canvas,
)
from skeleton_builder import (
    LEFT_ANKLE,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_ANKLE,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    derive_joints,
)
from utils import FpsCounter, angle_deg


#  Action classifier 

class ActionClassifier:
    def __init__(self, window: int = 8) -> None:
        self._buf: Deque[str] = deque(maxlen=window)
        self._ankle_y_hist: Deque[float] = deque(maxlen=15)

    def classify(self, lm: np.ndarray, vel: np.ndarray) -> str:
        if lm.shape[0] < 33:
            return "Idle"

        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        lw, rw = lm[LEFT_WRIST],    lm[RIGHT_WRIST]
        lh, rh = lm[LEFT_HIP],      lm[RIGHT_HIP]
        lk, rk = lm[LEFT_KNEE],     lm[RIGHT_KNEE]
        la, ra = lm[LEFT_ANKLE],    lm[RIGHT_ANKLE]

        l_knee_angle = angle_deg(lh[:2], lk[:2], la[:2])
        r_knee_angle = angle_deg(rh[:2], rk[:2], ra[:2])
        squatting = l_knee_angle < 110 and r_knee_angle < 110
        arms_horizontal = (
            abs(lw[1] - ls[1]) < 30 and abs(rw[1] - rs[1]) < 30
            and lw[0] < ls[0] and rw[0] > rs[0]
        )
        wrist_above_head = lw[1] < ls[1] - 50 or rw[1] < rs[1] - 50
        hip_y_vel = float(vel[LEFT_HIP, 1]) if vel.shape[0] > LEFT_HIP else 0.0
        ankle_avg_y = (la[1] + ra[1]) * 0.5
        self._ankle_y_hist.append(ankle_avg_y)
        ankle_var = float(np.var(self._ankle_y_hist)) if len(self._ankle_y_hist) > 4 else 0.0
        wrist_speed = (float(np.linalg.norm(vel[LEFT_WRIST, :2]))
                     + float(np.linalg.norm(vel[RIGHT_WRIST, :2])))

        if   arms_horizontal:                      label = "T-Pose"
        elif squatting:                            label = "Squatting"
        elif hip_y_vel < -200:                     label = "Jumping"
        elif wrist_above_head and wrist_speed > 200: label = "Waving"
        elif ankle_var > 400:                      label = "Walking"
        elif wrist_speed > 400:                    label = "Dancing"
        else:                                      label = "Idle"

        self._buf.append(label)
        return max(set(self._buf), key=self._buf.count)


#  Metrics 

def compute_metrics(lm: np.ndarray, vel: np.ndarray) -> Dict[str, float]:
    speed = float(np.linalg.norm(vel[:33, :2], axis=1).mean())
    energy_joints = [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]
    energy = float(np.sum(np.linalg.norm(vel[energy_joints, :2], axis=1) ** 2))
    l_elbow_a = angle_deg(lm[LEFT_SHOULDER, :2], lm[13, :2], lm[LEFT_WRIST, :2])
    r_elbow_a = angle_deg(lm[RIGHT_SHOULDER, :2], lm[14, :2], lm[RIGHT_WRIST, :2])
    l_knee_a  = angle_deg(lm[LEFT_HIP, :2], lm[LEFT_KNEE, :2], lm[LEFT_ANKLE, :2])
    r_knee_a  = angle_deg(lm[RIGHT_HIP, :2], lm[RIGHT_KNEE, :2], lm[RIGHT_ANKLE, :2])
    arm_diff = abs(l_elbow_a - r_elbow_a)
    leg_diff = abs(l_knee_a  - r_knee_a)
    symmetry = max(0.0, 100.0 - (arm_diff + leg_diff) * 0.5)
    return {
        "speed":    speed,
        "energy":   min(100.0, energy / 1000.0),
        "symmetry": symmetry,
    }


# Main run loop 

def run(
    source: int | str,
    multi: bool,
    overlay: bool,
    show_heatmap: bool,
    record_path: Optional[str],
    max_width: int = 960,
) -> None:

    smoother    = EMASmoother(base_alpha=0.5)
    vel_calc    = VelocityCalculator()
    tracker     = PersonTracker(match_radius_px=250)
    classifier  = ActionClassifier()
    fps         = FpsCounter()


    single_pose: Optional[PoseDetector]      = None
    multi_pose:  Optional[MultiPersonPoseDetector] = None
    if multi:
        multi_pose  = MultiPersonPoseDetector(max_persons=3)
    else:
        single_pose = PoseDetector()

    writer: Optional[cv2.VideoWriter] = None
    window_name = "AI Motion Twin"
    print("Press 'q' to quit, 'h' to toggle heatmap, 'o' to toggle overlay mode.")

    # Choose whichever detector is active as the context manager
    pose_ctx = multi_pose if multi else single_pose

    with FrameSource(source) as src, pose_ctx:  # type: ignore[union-attr]
        for frame_bgr in src.frames(mirror=isinstance(source, int)):
            frame_bgr = resize_keep_aspect(frame_bgr, max_width)
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # FIX 1 — each branch calls a concretely-typed variable; no union
            detections: List[np.ndarray]
            if multi and multi_pose is not None:
                detections = multi_pose.detect(rgb)          
            elif single_pose is not None:
                raw = single_pose.detect(rgb)               
                detections = [raw] if raw is not None else []  
            else:
                detections = []

            person_ids = tracker.assign(detections) if detections else []

            now = time.perf_counter()
            skeletons: List[np.ndarray] = []
            primary_metrics: Dict[str, float] = {}
            primary_action: Optional[str] = None

            for det, pid in zip(detections, person_ids):
                derived  = derive_joints(det)
                smoothed = smoother.smooth(pid, derived)
                vel      = vel_calc.update(pid, smoothed, now)
                skeletons.append(smoothed)
                if pid == person_ids[0]:
                    primary_metrics = compute_metrics(smoothed, vel)
                    primary_action  = classifier.classify(smoothed, vel)

            stick_canvas = (
                frame_bgr.copy()
                if overlay
                else make_blank_canvas(*frame_bgr.shape[1::-1])
            )
            if show_heatmap and skeletons:
                draw_heatmap(stick_canvas, skeletons[0],
                             vel_calc.update(person_ids[0], skeletons[0], now))
            draw_skeletons(stick_canvas, skeletons, person_ids)

            current_fps = fps.tick()
            composite = compose_side_by_side(
                frame_bgr, stick_canvas,
                fps=current_fps,
                person_count=len(skeletons),
                action=primary_action,
                metrics=primary_metrics,
            )

            if record_path:
                if writer is None:
                    h, w = composite.shape[:2]
                   
                    fourcc = getattr(cv2, "VideoWriter_fourcc")(*"mp4v")
                    writer = cv2.VideoWriter(record_path, fourcc, 24.0, (w, h))
                writer.write(composite)

            cv2.imshow(window_name, composite)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("h"):
                show_heatmap = not show_heatmap
            elif key == ord("o"):
                overlay = not overlay

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


# CLI 

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Motion Twin")
    sub = parser.add_subparsers(dest="command")

    live = sub.add_parser("live", help="Run the realtime viewer (default).")
    live.add_argument("--source",    default="0")
    live.add_argument("--multi",     action="store_true")
    live.add_argument("--overlay",   action="store_true")
    live.add_argument("--heatmap",   action="store_true")
    live.add_argument("--record",    default=None)
    live.add_argument("--max-width", type=int, default=960)

    ana = sub.add_parser("analyze", help="Batch-process videos into a CSV dataset.")
    ana.add_argument("--input",     required=True)
    ana.add_argument("--output",    default="dataset")
    ana.add_argument("--max-width", type=int, default=960)
    ana.add_argument("--merge",     action="store_true")

    raw = list(argv) if argv is not None else sys.argv[1:]
    if not raw or raw[0] not in {"live", "analyze", "-h", "--help"}:
        raw = ["live", *raw]
    return parser.parse_args(raw)


def _run_live(args: argparse.Namespace) -> int:
    raw_source = args.source if args.source is not None else "0"
    try:
        source: int | str = int(raw_source)
    except ValueError:
        source = raw_source
    try:
        run(
            source=source,
            multi=args.multi,
            overlay=args.overlay,
            show_heatmap=args.heatmap,
            record_path=args.record,
            max_width=args.max_width or 960,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    from pathlib import Path
    from offline_analyzer import analyze_folder
    try:
        analyze_folder(
            input_dir=Path(args.input),
            output_dir=Path(args.output),
            max_width=args.max_width or 960,
            merge=args.merge,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "analyze":
        return _run_analyze(args)
    return _run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())

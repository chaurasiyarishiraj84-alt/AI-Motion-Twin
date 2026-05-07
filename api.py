from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

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
    LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, LEFT_SHOULDER, LEFT_WRIST,
    RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_SHOULDER, RIGHT_WRIST,
    derive_joints,
)
from utils import FpsCounter, angle_deg

#  App 
app = FastAPI(title="AI Motion Twin")

TEMPLATES_DIR = Path(__file__).parent / "templates"
UPLOAD_DIR    = Path(tempfile.mkdtemp(prefix="motion_twin_"))

#  Shared state (guarded by state_lock)

state: Dict = {
    "overlay":      False,
    "heatmap":      False,
    "multi":        False,
    "source":       0,          # int = webcam index, str = file path
    "fps":          0.0,
    "person_count": 0,
    "action":       "Idle",
    "metrics":      {},
}
state_lock = threading.Lock()

_latest_frame: bytes = b""
_frame_lock   = threading.Lock()
_frame_event  = threading.Event()   # set whenever a new frame is ready


# Action classifier 

class ActionClassifier:
    def __init__(self, window: int = 8) -> None:
        self._buf:          deque = deque(maxlen=window)
        self._ankle_y_hist: deque = deque(maxlen=15)

    def classify(self, lm: np.ndarray, vel: np.ndarray) -> str:
        if lm.shape[0] < 33:
            return "Idle"
        ls, rs = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        lw, rw = lm[LEFT_WRIST],    lm[RIGHT_WRIST]
        lh, rh = lm[LEFT_HIP],      lm[RIGHT_HIP]
        lk, rk = lm[LEFT_KNEE],     lm[RIGHT_KNEE]
        la, ra = lm[LEFT_ANKLE],    lm[RIGHT_ANKLE]

        l_knee_a = angle_deg(lh[:2], lk[:2], la[:2])
        r_knee_a = angle_deg(rh[:2], rk[:2], ra[:2])
        squatting       = l_knee_a < 110 and r_knee_a < 110
        arms_horizontal = (
            abs(lw[1] - ls[1]) < 30 and abs(rw[1] - rs[1]) < 30
            and lw[0] < ls[0] and rw[0] > rs[0]
        )
        wrist_above = lw[1] < ls[1] - 50 or rw[1] < rs[1] - 50
        hip_y_vel   = float(vel[LEFT_HIP, 1]) if vel.shape[0] > LEFT_HIP else 0.0

        ankle_avg_y = (la[1] + ra[1]) * 0.5
        self._ankle_y_hist.append(ankle_avg_y)
        ankle_var = float(np.var(self._ankle_y_hist)) if len(self._ankle_y_hist) > 4 else 0.0

        wrist_speed = (float(np.linalg.norm(vel[LEFT_WRIST,  :2]))
                     + float(np.linalg.norm(vel[RIGHT_WRIST, :2])))

        if   arms_horizontal:                  label = "T-Pose"
        elif squatting:                        label = "Squatting"
        elif hip_y_vel < -200:                 label = "Jumping"
        elif wrist_above and wrist_speed > 200: label = "Waving"
        elif ankle_var > 400:                  label = "Walking"
        elif wrist_speed > 400:                label = "Dancing"
        else:                                  label = "Idle"

        self._buf.append(label)
        return max(set(self._buf), key=self._buf.count)


def compute_metrics(lm: np.ndarray, vel: np.ndarray) -> Dict:
    speed         = float(np.linalg.norm(vel[:33, :2], axis=1).mean())
    energy_joints = [LEFT_WRIST, RIGHT_WRIST, LEFT_ANKLE, RIGHT_ANKLE]
    energy        = float(np.sum(np.linalg.norm(vel[energy_joints, :2], axis=1) ** 2))
    l_elbow_a     = angle_deg(lm[LEFT_SHOULDER, :2], lm[13, :2], lm[LEFT_WRIST, :2])
    r_elbow_a     = angle_deg(lm[RIGHT_SHOULDER, :2], lm[14, :2], lm[RIGHT_WRIST, :2])
    l_knee_a      = angle_deg(lm[LEFT_HIP, :2], lm[LEFT_KNEE, :2], lm[LEFT_ANKLE, :2])
    r_knee_a      = angle_deg(lm[RIGHT_HIP, :2], lm[RIGHT_KNEE, :2], lm[RIGHT_ANKLE, :2])
    symmetry      = max(0.0, 100.0 - (abs(l_elbow_a - r_elbow_a)
                                     + abs(l_knee_a  - r_knee_a)) * 0.5)
    return {
        "speed":    round(speed,    1),
        "energy":   round(min(100.0, energy / 1000.0), 1),
        "symmetry": round(symmetry, 1),
    }


#  Pipeline thread (singleton) 
def _pipeline_loop() -> None:
    """Background daemon that owns capture + pose and feeds _latest_frame."""
    global _latest_frame

    smoother    = EMASmoother(base_alpha=0.5)
    vel_calc    = VelocityCalculator()
    tracker     = PersonTracker(match_radius_px=250)
    classifier  = ActionClassifier()
    fps_counter = FpsCounter()

    # Explicit types so Pylance never infers these as plain None
    current_source: Optional[Union[int, str]]                              = None
    current_multi:  Optional[bool]                                         = None
    pose_ctx: Any = None  # PoseDetector | MultiPersonPoseDetector — Any avoids
                          # Pylance failing to resolve .detect() on the union
    src:            Optional[FrameSource]                                  = None

    while True:
        with state_lock:
            source       = state["source"]
            overlay      = state["overlay"]
            show_heatmap = state["heatmap"]
            multi        = state["multi"]

        #  Re-init pipeline when source or mode changes 
        if source != current_source or multi != current_multi:
            if pose_ctx is not None:
                pose_ctx.close()
            if src is not None:
                src.close()
            smoother.reset()
            vel_calc.reset()

            current_source = None
            current_multi  = multi

            pose_ctx = (MultiPersonPoseDetector(max_persons=3)
                        if multi else PoseDetector())
            src = FrameSource(source, width=1280, height=720)
            try:
                src.open()
                current_source = source   # only set after successful open
            except RuntimeError:
                # Emit a "source unavailable" placeholder and keep retrying
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Source unavailable", (60, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 80, 255), 2)
                _, buf = cv2.imencode(".jpg", placeholder)
                with _frame_lock:
                    _latest_frame = buf.tobytes()
                _frame_event.set()
                time.sleep(1)
                continue

           
        assert src is not None
        cap = src._cap
        assert cap is not None
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            if isinstance(source, str):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video file
            else:
                time.sleep(0.01)
            continue

        if isinstance(source, int):
            frame_bgr = cv2.flip(frame_bgr, 1)
        frame_bgr = resize_keep_aspect(frame_bgr, 960)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Pose detection 
        assert pose_ctx is not None
        if multi:
            detections = pose_ctx.detect(rgb)
        else:
            lm = pose_ctx.detect(rgb)
            detections = [lm] if lm is not None else []

        person_ids = tracker.assign(detections) if detections else []

        now = time.perf_counter()
        skeletons: list        = []
        primary_metrics: Dict  = {}
        primary_action: str    = "Idle"
        primary_vel: np.ndarray = np.zeros((36, 3), dtype=np.float32)

        for det, pid in zip(detections, person_ids):
            derived  = derive_joints(det)
            smoothed = smoother.smooth(pid, derived)
            vel      = vel_calc.update(pid, smoothed, now)   # called ONCE per person
            skeletons.append(smoothed)
            if pid == person_ids[0]:
                primary_metrics = compute_metrics(smoothed, vel)
                primary_action  = classifier.classify(smoothed, vel)
                primary_vel     = vel   # cache for heatmap — BUG 1 FIX

        #  Build composite frame 
        stick_canvas = (
            frame_bgr.copy()
            if overlay
            else make_blank_canvas(*frame_bgr.shape[1::-1])
        )

       
        if show_heatmap and skeletons:
            draw_heatmap(stick_canvas, skeletons[0], primary_vel)

        draw_skeletons(stick_canvas, skeletons, person_ids)

        current_fps = fps_counter.tick()
        composite = compose_side_by_side(
            frame_bgr, stick_canvas,
            fps=current_fps,
            person_count=len(skeletons),
            action=primary_action,
            metrics=primary_metrics,
        )

        # Update shared state
        with state_lock:
            state["fps"]          = round(current_fps, 1)
            state["person_count"] = len(skeletons)
            state["action"]       = primary_action
            state["metrics"]      = primary_metrics

        # Write JPEG to shared buffer
        _, buf = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 82])
        with _frame_lock:
            _latest_frame = buf.tobytes()
        _frame_event.set()


# Start the singleton pipeline thread at import time
_pipeline_thread = threading.Thread(target=_pipeline_loop, daemon=True, name="pipeline")
_pipeline_thread.start()


# MJPEG generator (reads from shared buffer, safe for multiple clients) 

def _mjpeg_generator():
    """Yield MJPEG boundary frames. Multiple /video clients share one pipeline."""
    BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    while True:
        _frame_event.wait(timeout=2.0)
        _frame_event.clear()
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield BOUNDARY + frame + b"\r\n"


# HTTP routes 

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse((TEMPLATES_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/video")
def video():
    
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/toggle-overlay")
def toggle_overlay():
    with state_lock:
        state["overlay"] = not state["overlay"]
        return {"overlay": state["overlay"]}


@app.get("/toggle-heatmap")
def toggle_heatmap_route():
    with state_lock:
        state["heatmap"] = not state["heatmap"]
        return {"heatmap": state["heatmap"]}


@app.get("/toggle-multi")
def toggle_multi():
    with state_lock:
        state["multi"] = not state["multi"]
        return {"multi": state["multi"]}


@app.get("/status")
def status():
    with state_lock:
        return JSONResponse({
            "overlay":      state["overlay"],
            "heatmap":      state["heatmap"],
            "multi":        state["multi"],
            "fps":          state["fps"],
            "person_count": state["person_count"],
            "action":       state["action"],
            "metrics":      state["metrics"],
            "source":       "webcam" if isinstance(state["source"], int) else "video",
        })


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename or ""   
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        return JSONResponse({"error": f"Unsupported format: {suffix}"}, status_code=400)
    dest = UPLOAD_DIR / f"upload{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    with state_lock:
        state["source"] = str(dest)
    return {"status": "ok", "filename": filename}


@app.post("/use-webcam")
def use_webcam():
    with state_lock:
        state["source"] = 0
    return {"status": "ok", "source": "webcam"}

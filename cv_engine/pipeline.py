"""
CV pipeline: MediaPipe Holistic (legacy) or Pose+Face Landmarkers (tasks API), kinesic entropy,
gaze, raw landmark extraction with Z normalization, and optional Biometric Heatmap.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from core.state import TelemetryState
from cv_engine.capture import CameraCapture
from cv_engine.gaze_tracker import GazeTracker
from cv_engine.heatmap import draw_heatmap
from cv_engine.mediapipe_backend import (
    create_face_landmarker,
    create_holistic,
    create_pose_landmarker,
    has_holistic,
    has_tasks,
    process_frame_tasks,
)
from cv_engine.pose_entropy import PoseEntropyTracker
from backend_signalprocessing_unit.fidget_detector import FidgetDetector as BSPFidgetDetector
from backend_signalprocessing_unit.gaze_tracker import GazeTracker as BSPGazeTracker

# Landmark indices for raw (x,y,z) mapping
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
FACE_LEFT_IRIS_CENTER = 468
FACE_RIGHT_IRIS_CENTER = 473
FACE_REFINED_MIN_LEN = 478
FACE_PNP_INDICES = (1, 152, 33, 263, 61, 291)
CONFIDENCE_GATE_MIN_LEN = 468


def _get_config(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    d = config
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def _lm_xyz(landmarks: Any, index: int) -> tuple[float, float, float] | None:
    if index >= len(landmarks):
        return None
    lm = landmarks[index]
    x = getattr(lm, "x", None) or (lm[0] if hasattr(lm, "__getitem__") else None)
    y = getattr(lm, "y", None) or (lm[1] if hasattr(lm, "__getitem__") else None)
    z = getattr(lm, "z", None) if hasattr(lm, "z") else (lm[2] if hasattr(lm, "__getitem__") and len(lm) > 2 else 0.0)
    if x is None or y is None or z is None:
        return None
    try:
        x, y, z = float(x), float(y), float(z)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return (x, y, z)


def _face_confidence_gate_passed(face_landmarks: Any) -> bool:
    if face_landmarks is None:
        return False
    lms = getattr(face_landmarks, "landmark", None)
    if lms is None or len(lms) < CONFIDENCE_GATE_MIN_LEN:
        return False
    for idx in FACE_PNP_INDICES:
        if _lm_xyz(lms, idx) is None:
            return False
    if len(lms) >= FACE_REFINED_MIN_LEN:
        for idx in (FACE_LEFT_IRIS_CENTER, FACE_RIGHT_IRIS_CENTER):
            if _lm_xyz(lms, idx) is None:
                return False
    return True


def _extract_raw_landmarks(
    pose_landmarks: Any,
    face_landmarks: Any,
) -> dict[str, tuple[float, float, float]]:
    out: dict[str, tuple[float, float, float]] = {}
    default = (0.0, 0.0, 0.0)
    if pose_landmarks is None:
        for k in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist"):
            out[k] = default
    else:
        plm = getattr(pose_landmarks, "landmark", None) or []
        ls = _lm_xyz(plm, POSE_LEFT_SHOULDER)
        rs = _lm_xyz(plm, POSE_RIGHT_SHOULDER)
        mid_z = (ls[2] + rs[2]) / 2.0 if (ls is not None and rs is not None) else 0.0
        for key, idx in (
            ("left_shoulder", POSE_LEFT_SHOULDER),
            ("right_shoulder", POSE_RIGHT_SHOULDER),
            ("left_wrist", POSE_LEFT_WRIST),
            ("right_wrist", POSE_RIGHT_WRIST),
        ):
            xyz = _lm_xyz(plm, idx)
            out[key] = (xyz[0], xyz[1], xyz[2] - mid_z) if xyz else default
    if face_landmarks is None:
        out["left_iris"] = default
        out["right_iris"] = default
    else:
        flm = getattr(face_landmarks, "landmark", None) or []
        if len(flm) < FACE_REFINED_MIN_LEN:
            out["left_iris"] = default
            out["right_iris"] = default
        else:
            out["left_iris"] = _lm_xyz(flm, FACE_LEFT_IRIS_CENTER) or default
            out["right_iris"] = _lm_xyz(flm, FACE_RIGHT_IRIS_CENTER) or default
    return out


class CVPipeline:
    """
    Single-threaded CV loop: Holistic or Pose+Face -> entropy, gaze, raw_landmarks, optional heatmap.
    """

    def __init__(
        self,
        state: TelemetryState,
        config: dict[str, Any] | None = None,
        camera_index: int | None = None,
        heatmap_queue: queue.Queue[Any] | None = None,
    ) -> None:
        self.state = state
        self.config = config or {}
        self._heatmap_queue = heatmap_queue
        cam_cfg = self.config.get("camera", {})
        w = _get_config(self.config, "camera", "width", default=640)
        h = _get_config(self.config, "camera", "height", default=480)
        fps = _get_config(self.config, "camera", "fps", default=15.0)
        idx = camera_index if camera_index is not None else cam_cfg.get("index", 0)
        self.capture = CameraCapture(camera_index=idx, width=w, height=h, target_fps=fps)
        kinesic_cfg = self.config.get("kinesic", {})
        self.pose_entropy = PoseEntropyTracker(
            landmark_indices=kinesic_cfg.get("landmark_indices", [11, 12, 13, 14, 15, 16]),
            window_seconds=kinesic_cfg.get("window_seconds", 1.5),
            calibration_seconds=kinesic_cfg.get("calibration_seconds", 5.0),
            target_fps=fps,
            threshold_sigma=kinesic_cfg.get("threshold_sigma", 2.0),
        )
        gaze_cfg = self.config.get("gaze", {})
        self.gaze_tracker = GazeTracker(
            frame_width=w,
            frame_height=h,
            pitch_min=gaze_cfg.get("golden_zone_pitch_min", -15),
            pitch_max=gaze_cfg.get("golden_zone_pitch_max", 15),
            yaw_min=gaze_cfg.get("golden_zone_yaw_min", -20),
            yaw_max=gaze_cfg.get("golden_zone_yaw_max", 20),
            gaze_lost_seconds=gaze_cfg.get("gaze_lost_seconds", 1.5),
        )
        sensory_cfg = self.config.get("sensory", {}) or self.config.get("holistic", {})
        model_complexity = sensory_cfg.get("model_complexity", 0)
        refine_face = sensory_cfg.get("refine_face_landmarks", True)
        heatmap_cfg = self.config.get("heatmap", {})
        self._heatmap_enabled = heatmap_cfg.get("enabled", False) or (heatmap_queue is not None)
        self._heatmap_window_title = heatmap_cfg.get("window_title", "Lumina Biometric Heatmap")
        self._use_holistic = has_holistic()
        self._holistic = None
        self._pose_landmarker = None
        self._face_landmarker = None
        if self._use_holistic:
            self._holistic = create_holistic(model_complexity=model_complexity, refine_face_landmarks=refine_face)
        elif has_tasks():
            project_root = Path(__file__).resolve().parent.parent
            default_pose = project_root / "assets" / "models" / "pose_landmarker_lite.task"
            default_face = project_root / "assets" / "models" / "face_landmarker.task"
            pose_path = sensory_cfg.get("pose_model_path") or str(default_pose)
            face_path = sensory_cfg.get("face_model_path") or str(default_face)
            if not Path(pose_path).exists() or not Path(face_path).exists():
                raise RuntimeError(
                    "MediaPipe 0.10+ requires model files. Run: python scripts/download_models.py"
                )
            self._pose_landmarker = create_pose_landmarker(pose_path, model_complexity)
            self._face_landmarker = create_face_landmarker(face_path)
        else:
            raise RuntimeError("No MediaPipe backend available. Install mediapipe with tasks or legacy solutions.")
        # Backend signal-processing unit (BSP) detectors
        bsp_cfg = self.config.get("bsp", {})
        self.bsp_fidget = BSPFidgetDetector(
            threshold=bsp_cfg.get("fidget_threshold", 0.0008),
            window_size=bsp_cfg.get("fidget_window_size", 45),
            smoothing_window=bsp_cfg.get("fidget_smoothing_window", 3),
        )
        self.bsp_gaze = BSPGazeTracker(
            deviation_threshold=bsp_cfg.get("gaze_deviation_threshold", 0.12),
            time_threshold=bsp_cfg.get("gaze_time_threshold", 3.0),
            smoothing_window=bsp_cfg.get("gaze_smoothing_window", 12),
        )
        self._stop = threading.Event()
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0

    def _process_frame(self, frame: cv2.Mat, timestamp: float) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._use_holistic:
            results = self._holistic.process(rgb)
        else:
            results = process_frame_tasks(self._pose_landmarker, self._face_landmarker, rgb)
        pose_lm = results.pose_landmarks
        pose_list = getattr(pose_lm, "landmark", None) if pose_lm else None
        if pose_list:
            stability, high_entropy = self.pose_entropy.update(pose_list, timestamp)
            self.state.update(
                stability_score=stability,
                high_entropy=high_entropy,
                calibration_done=self.pose_entropy.calibration_done,
            )
            # BSP fidget detector
            bsp_fidget, bsp_var = self.bsp_fidget.update(pose_list)
            self.state.update(
                bsp_fidgeting=bsp_fidget,
                bsp_fidget_variance=bsp_var,
            )
        else:
            self.state.update(last_frame_ok=True)
        if not _face_confidence_gate_passed(results.face_landmarks):
            self.state.update(
                gaze_lost=True,
                time_in_zone_seconds=0.0,
                yaw_degrees=0.0,
                pitch_degrees=0.0,
                last_frame_ok=True,
                bsp_gaze_status="GAZE LOST",
                bsp_gaze_distance=-1.0,
            )
        elif results.face_landmarks:
            lms = getattr(results.face_landmarks, "landmark", None) or []
            snap = self.state.snapshot()
            current_ref = snap.get("reference_frame", "digital")
            gaze_lost, time_in_zone, yaw, pitch = self.gaze_tracker.update(
                lms, timestamp, current_ref
            )
            self.state.update(
                gaze_lost=gaze_lost,
                time_in_zone_seconds=time_in_zone,
                yaw_degrees=yaw,
                pitch_degrees=pitch,
                last_frame_ok=True,
            )
            # BSP iris-based gaze tracker
            bsp_status, bsp_dist = self.bsp_gaze.update(lms)
            self.state.update(
                bsp_gaze_status=bsp_status,
                bsp_gaze_distance=bsp_dist,
            )
        else:
            self.state.update(
                gaze_lost=True,
                last_frame_ok=True,
                bsp_gaze_status="GAZE LOST",
                bsp_gaze_distance=-1.0,
            )
        raw = _extract_raw_landmarks(results.pose_landmarks, results.face_landmarks)
        self.state.update(raw_landmarks=raw)
        
        # Always generate the heatmap for the HUD
        draw_face = self.state.snapshot().get("show_face_heatmap", True)
        annotated = draw_heatmap(
            frame, results,
            window_title=self._heatmap_window_title,
            draw_face=draw_face,
        )
        self.state.update(latest_frame=annotated)
        
        if self._heatmap_enabled and self._heatmap_queue is not None:
            try:
                if self._heatmap_queue.full():
                    try:
                        self._heatmap_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._heatmap_queue.put_nowait(annotated)
            except (queue.Full, Exception):
                pass
        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now
        self.state.update(fps=self._fps)

    def run(self) -> None:
        with self.capture:
            while not self._stop.is_set():
                ok, frame, t = self.capture.read()
                if not ok or frame is None:
                    self.state.update(last_frame_ok=False)
                    break
                self._process_frame(frame, t)
        if self._holistic is not None:
            self._holistic.close()
        if self._pose_landmarker is not None:
            self._pose_landmarker.close()
        if self._face_landmarker is not None:
            self._face_landmarker.close()

    def stop(self) -> None:
        self._stop.set()

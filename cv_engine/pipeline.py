"""
CV pipeline: runs MediaPipe Pose and Face Mesh on each frame, computes
kinesic entropy and gaze state, and updates shared TelemetryState.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from core.state import TelemetryState
from cv_engine.capture import CameraCapture
from cv_engine.gaze_tracker import GazeTracker
from cv_engine.pose_entropy import PoseEntropyTracker


def _get_config(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Navigate nested config dict with default."""
    d = config
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


class CVPipeline:
    """
    Single-threaded CV loop: capture frame -> Pose + Face Mesh -> entropy + gaze -> state.
    Intended to run in a dedicated thread; state is updated under its lock.
    """

    def __init__(
        self,
        state: TelemetryState,
        config: dict[str, Any] | None = None,
        camera_index: int | None = None,
    ) -> None:
        self.state = state
        self.config = config or {}
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
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._stop = threading.Event()
        self._frame_count = 0
        self._last_fps_time = time.perf_counter()
        self._fps = 0.0

    def _process_frame(self, frame: cv2.Mat, timestamp: float) -> None:
        """Run Pose + Face Mesh on frame and update state."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        # Pose
        pose_result = self._pose.process(rgb)
        if pose_result.pose_landmarks:
            stability, high_entropy = self.pose_entropy.update(
                pose_result.pose_landmarks.landmark,
                timestamp,
            )
            self.state.update(
                stability_score=stability,
                high_entropy=high_entropy,
                calibration_done=self.pose_entropy.calibration_done,
            )
        else:
            self.state.update(last_frame_ok=True)
        # Face Mesh -> gaze
        face_result = self._face_mesh.process(rgb)
        if face_result.multi_face_landmarks:
            lms = face_result.multi_face_landmarks[0].landmark
            gaze_lost, time_in_zone, yaw, pitch = self.gaze_tracker.update(lms, timestamp)
            self.state.update(
                gaze_lost=gaze_lost,
                time_in_zone_seconds=time_in_zone,
                yaw_degrees=yaw,
                pitch_degrees=pitch,
                last_frame_ok=True,
            )
        else:
            self.state.update(gaze_lost=True, last_frame_ok=True)
        # FPS
        self._frame_count += 1
        now = time.perf_counter()
        if now - self._last_fps_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_fps_time)
            self._frame_count = 0
            self._last_fps_time = now
        self.state.update(fps=self._fps)

    def run(self) -> None:
        """Run the capture loop until stop is set. Call from a worker thread."""
        with self.capture:
            while not self._stop.is_set():
                ok, frame, t = self.capture.read()
                if not ok or frame is None:
                    self.state.update(last_frame_ok=False)
                    break
                self._process_frame(frame, t)
        self._pose.close()
        self._face_mesh.close()

    def stop(self) -> None:
        """Signal the pipeline to stop after the current frame."""
        self._stop.set()

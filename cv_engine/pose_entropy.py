"""
Monocular kinesic entropy analysis ("Fidget" engine).

Uses MediaPipe Pose 33-landmark output to compute micro-movement velocity
at wrists, elbows, and shoulders. When velocity exceeds a calibrated baseline,
the tracker flags high kinesic entropy (fidgeting/anxiety) for HUD alerts.
"""

from __future__ import annotations

from collections import deque
from typing import List, Sequence

import numpy as np


# Default landmark indices: shoulders 11,12; elbows 13,14; wrists 15,16
DEFAULT_LANDMARK_INDICES = (11, 12, 13, 14, 15, 16)


class PoseEntropyTracker:
    """
    Tracks temporal variance of upper-body landmarks and flags high entropy.
    Call set_baseline_from_calibration() after calibration_seconds of low movement,
    then update() each frame with normalized landmark coordinates (x, y, z).
    """

    def __init__(
        self,
        landmark_indices: Sequence[int] = DEFAULT_LANDMARK_INDICES,
        window_seconds: float = 1.5,
        calibration_seconds: float = 5.0,
        target_fps: float = 15.0,
        threshold_sigma: float = 2.0,
    ) -> None:
        self.landmark_indices = list(landmark_indices)
        self.window_seconds = window_seconds
        self.calibration_seconds = calibration_seconds
        self.target_fps = target_fps
        self.threshold_sigma = threshold_sigma
        self._max_len = max(2, int(window_seconds * target_fps))
        self._calib_len = max(2, int(calibration_seconds * target_fps))
        # Queue of per-frame "micro-movement velocity" scalars (mean over landmarks)
        self._velocity_queue: deque[float] = deque(maxlen=self._max_len)
        self._calibration_samples: list[float] = []
        self._baseline_mean: float = 0.0
        self._baseline_std: float = 1e-6
        self._calibration_done = False
        self._prev_points: np.ndarray | None = None
        self._prev_time: float | None = None

    def _landmarks_to_points(self, landmarks: Sequence[object]) -> np.ndarray | None:
        """Extract (x,y,z) for our indices. landmarks can be MediaPipe NormalizedLandmark."""
        points = []
        for i in self.landmark_indices:
            if i >= len(landmarks):
                return None
            lm = landmarks[i]
            x = getattr(lm, "x", None) or lm[0]
            y = getattr(lm, "y", None) or lm[1]
            z = getattr(lm, "z", None) if hasattr(lm, "z") else (lm[2] if len(lm) > 2 else 0)
            points.append([x, y, z])
        return np.array(points, dtype=np.float64)

    def set_baseline_from_calibration(self) -> None:
        """Use collected calibration samples to set baseline mean and std."""
        if len(self._calibration_samples) < 2:
            self._baseline_mean = 0.0
            self._baseline_std = 1e-6
        else:
            arr = np.array(self._calibration_samples)
            self._baseline_mean = float(np.mean(arr))
            self._baseline_std = float(np.std(arr)) or 1e-6
        self._calibration_done = True

    def update(
        self,
        landmarks: Sequence[object],
        timestamp: float,
    ) -> tuple[float, bool]:
        """
        Update with pose landmarks for the current frame.
        landmarks: sequence of objects with .x, .y, .z (e.g. MediaPipe pose_landmarks).
        timestamp: seconds (used for dt).
        Returns (stability_score, high_entropy).
        stability_score in [0, 1]; 1 = stable. high_entropy True when fidgeting.
        """
        pts = self._landmarks_to_points(landmarks)
        if pts is None:
            return 1.0, False
        if self._prev_time is None:
            self._prev_time = timestamp
            self._prev_points = pts
            return 1.0, False
        dt = timestamp - self._prev_time
        if dt <= 0:
            dt = 1.0 / self.target_fps
        self._prev_time = timestamp
        # Velocity = magnitude of displacement / dt per landmark; then mean over landmarks
        disp = np.linalg.norm(pts - self._prev_points, axis=1)
        vel = float(np.mean(disp) / dt)
        self._prev_points = pts
        self._velocity_queue.append(vel)
        if not self._calibration_done:
            self._calibration_samples.append(vel)
            if len(self._calibration_samples) >= self._calib_len:
                self.set_baseline_from_calibration()
        if len(self._velocity_queue) < 2:
            return 1.0, False
        current = np.mean(self._velocity_queue)
        threshold = self._baseline_mean + self.threshold_sigma * self._baseline_std
        high_entropy = current > threshold
        # Map to stability score: 0 at threshold and above, 1 at baseline
        if threshold <= self._baseline_mean:
            stability = 1.0 if not high_entropy else 0.0
        else:
            stability = max(0.0, min(1.0, 1.0 - (current - self._baseline_mean) / (threshold - self._baseline_mean)))
        return stability, high_entropy

    @property
    def calibration_done(self) -> bool:
        return self._calibration_done

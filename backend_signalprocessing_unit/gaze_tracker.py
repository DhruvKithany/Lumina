"""
Iris-based gaze tracker for the signal-processing backend.

Accepts a MediaPipe Face Mesh landmark sequence (with refine_landmarks=True),
extracts the left and right iris centers (indices 468-472 and 473-477),
computes the smoothed mean iris position, and checks how far it deviates
from the frame center.

Returns a status string ("LOCKED", "WARNING", "GAZE LOST") and the
numeric deviation distance for telemetry.

This is a simpler, complementary approach to the solvePnP head-pose
estimation in cv_engine/gaze_tracker.py.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from backend_signalprocessing_unit.smoothing_buffer import SmoothingBuffer

# MediaPipe iris landmark indices (available when refine_landmarks=True)
# Left iris: 468 (center), 469-472 (ring)
# Right iris: 473 (center), 474-477 (ring)
LEFT_IRIS_INDICES = (468, 469, 470, 471, 472)
RIGHT_IRIS_INDICES = (473, 474, 475, 476, 477)


class GazeTracker:
    """
    Tracks iris position from Face Mesh landmarks and determines
    whether the presenter's gaze is locked on camera.

    Parameters
    ----------
    deviation_threshold : float
        Maximum normalized Euclidean distance from frame center
        before gaze is considered "off".
    time_threshold : float
        Seconds of continuous off-gaze before status becomes "GAZE LOST".
    smoothing_window : int
        Number of frames for the iris-position smoothing buffer.
    """

    def __init__(
        self,
        deviation_threshold: float = 0.05,
        time_threshold: float = 1.5,
        smoothing_window: int = 8,
    ) -> None:
        self.deviation_threshold = deviation_threshold
        self.time_threshold = time_threshold
        self.lost_since: float | None = None
        self._smoother = SmoothingBuffer(window_size=smoothing_window)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_iris_center(
        landmarks: Sequence[object],
    ) -> tuple[float, float] | None:
        """
        Compute the mean (x, y) of left and right iris landmarks.
        Returns None if the landmarks sequence is too short (no iris data).
        """
        required = max(max(LEFT_IRIS_INDICES), max(RIGHT_IRIS_INDICES)) + 1
        if len(landmarks) < required:
            return None

        xs, ys = [], []
        for idx in (*LEFT_IRIS_INDICES, *RIGHT_IRIS_INDICES):
            lm = landmarks[idx]
            x = getattr(lm, "x", None)
            y = getattr(lm, "y", None)
            if x is None:
                x = lm[0]
            if y is None:
                y = lm[1]
            xs.append(x)
            ys.append(y)

        return float(np.mean(xs)), float(np.mean(ys))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        landmarks: Sequence[object],
        center_x: float = 0.5,
        center_y: float = 0.5,
    ) -> tuple[str, float]:
        """
        Update gaze state from a Face Mesh landmark sequence.

        Parameters
        ----------
        landmarks : Sequence
            MediaPipe Face Mesh landmarks (must include refined iris points).
        center_x, center_y : float
            Normalized frame center (default 0.5, 0.5).

        Returns
        -------
        status : str
            One of "LOCKED", "WARNING", or "GAZE LOST".
        distance : float
            Smoothed Euclidean distance from iris center to frame center.
        """
        iris = self._extract_iris_center(landmarks)
        if iris is None:
            # No iris data — treat as gaze lost immediately
            return "GAZE LOST", -1.0

        iris_x, iris_y = iris

        # Smooth the iris position across frames
        smoothed = self._smoother.update([iris_x, iris_y])
        sx, sy = float(smoothed[0]), float(smoothed[1])

        distance = float(np.sqrt((sx - center_x) ** 2 + (sy - center_y) ** 2))
        gaze_is_good = distance < self.deviation_threshold

        if gaze_is_good:
            self.lost_since = None
            return "LOCKED", distance

        # Gaze is off-center
        if self.lost_since is None:
            self.lost_since = time.time()

        elapsed = time.time() - self.lost_since
        if elapsed > self.time_threshold:
            return "GAZE LOST", distance

        return "WARNING", distance
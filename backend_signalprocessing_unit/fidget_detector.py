"""
Variance-based fidget detector for the signal-processing backend.

Accepts a MediaPipe Pose landmark sequence (33 landmarks), extracts
the left and right wrist positions (indices 15 and 16), buffers them
over a sliding window, and flags fidgeting when the positional variance
exceeds a configurable threshold.

This is a simpler, complementary approach to the velocity-based entropy
tracking in cv_engine/pose_entropy.py.
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np

from backend_signalprocessing_unit.smoothing_buffer import SmoothingBuffer

# MediaPipe Pose landmark indices for wrists
LEFT_WRIST_INDEX = 15
RIGHT_WRIST_INDEX = 16


class FidgetDetector:
    """
    Detects fidgeting by tracking wrist position variance over a window.

    Parameters
    ----------
    threshold : float
        Variance above this value is flagged as fidgeting.
    window_size : int
        Number of frames in the sliding variance window (~1 s at 30 fps).
    smoothing_window : int
        Number of frames for per-wrist position smoothing.
    min_samples : int
        Minimum buffered frames before detection activates.
    """

    def __init__(
        self,
        threshold: float = 0.002,
        window_size: int = 30,
        smoothing_window: int = 5,
        min_samples: int = 10,
    ) -> None:
        self.threshold = threshold
        self.min_samples = min_samples
        self.buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self._left_smoother = SmoothingBuffer(window_size=smoothing_window)
        self._right_smoother = SmoothingBuffer(window_size=smoothing_window)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_wrist_coords(
        landmarks: Sequence[object],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Extract (x, y) for left and right wrists from a Pose landmark sequence.
        Returns None if the sequence is too short.
        """
        required = max(LEFT_WRIST_INDEX, RIGHT_WRIST_INDEX) + 1
        if len(landmarks) < required:
            return None

        def _xy(lm: object) -> np.ndarray:
            x = getattr(lm, "x", None)
            y = getattr(lm, "y", None)
            if x is None:
                x = lm[0]  # type: ignore[index]
            if y is None:
                y = lm[1]  # type: ignore[index]
            return np.array([x, y], dtype=np.float64)

        left = _xy(landmarks[LEFT_WRIST_INDEX])
        right = _xy(landmarks[RIGHT_WRIST_INDEX])
        return left, right

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        landmarks: Sequence[object],
    ) -> tuple[bool, float]:
        """
        Update with a Pose landmark sequence for the current frame.

        Parameters
        ----------
        landmarks : Sequence
            MediaPipe Pose landmarks (33 points with .x, .y attributes).

        Returns
        -------
        is_fidgeting : bool
            True when wrist variance exceeds the threshold.
        variance : float
            The computed positional variance (useful for telemetry / HUD).
        """
        wrists = self._extract_wrist_coords(landmarks)
        if wrists is None:
            return False, 0.0

        left, right = wrists

        # Smooth each wrist independently
        left_smooth = self._left_smoother.update(left)
        right_smooth = self._right_smoother.update(right)

        # Buffer the concatenated smoothed positions [lx, ly, rx, ry]
        combined = np.concatenate([left_smooth, right_smooth])
        self.buffer.append(combined)

        if len(self.buffer) < self.min_samples:
            return False, 0.0

        variance = float(np.var(self.buffer))
        return variance > self.threshold, variance
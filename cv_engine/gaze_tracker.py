"""
Sub-pixel iris tracking and gaze-vector locking via head pose estimation.

Uses MediaPipe Face Mesh landmarks with OpenCV solvePnP to estimate head pose
(yaw, pitch). The "golden zone" is defined by allowed yaw/pitch ranges (looking
at camera). If the head leaves the zone for > gaze_lost_seconds, a "Gaze Lost"
warning is triggered for the HUD.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import cv2
import numpy as np


# 3D model points (generic head model, same order as image point indices)
# Order: nose tip, chin, left eye left corner, right eye right corner, left mouth, right mouth
MODEL_POINTS_3D = np.array(
    [
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0),  # Right mouth corner
    ],
    dtype=np.float64,
)

# MediaPipe Face Mesh landmark indices for the 6 points above
# https://developers.google.com/mediapipe/solutions/vision/face_landmarker
FACE_MESH_INDICES = (1, 152, 33, 263, 61, 291)


def _rotation_vector_to_euler(rvec: np.ndarray) -> tuple[float, float, float]:
    """Convert OpenCV rotation vector to (pitch_deg, yaw_deg, roll_deg)."""
    rmat, _ = cv2.Rodrigues(rvec)
    # Extract Euler angles (one common convention: pitch, yaw, roll)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0
    return (
        math.degrees(pitch),
        math.degrees(yaw),
        math.degrees(roll),
    )


class GazeTracker:
    """
    Tracks head pose from face landmarks and determines if the user is in the
    "golden zone" (looking at camera). Triggers gaze_lost when out-of-zone
    for longer than gaze_lost_seconds.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        pitch_min: float = -15,
        pitch_max: float = 15,
        yaw_min: float = -20,
        yaw_max: float = 20,
        gaze_lost_seconds: float = 1.5,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pitch_min = pitch_min
        self.pitch_max = pitch_max
        self.yaw_min = yaw_min
        self.yaw_max = yaw_max
        self.gaze_lost_seconds = gaze_lost_seconds
        # Camera matrix: focal length ~ width, principal point at center
        fx = float(frame_width)
        fy = float(frame_width)
        cx = frame_width / 2.0
        cy = frame_height / 2.0
        self._camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            dtype=np.float64,
        )
        self._dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        self._out_of_zone_since: float | None = None
        self._in_zone_since: float | None = None
        self._time_in_zone_seconds: float = 0.0

    def _landmarks_to_image_points(self, landmarks: Sequence[object]) -> np.ndarray | None:
        """Extract 2D image points for MODEL_POINTS_3D order. landmarks are normalized [0,1]."""
        points = []
        for i in FACE_MESH_INDICES:
            if i >= len(landmarks):
                return None
            lm = landmarks[i]
            x = getattr(lm, "x", None) or lm[0]
            y = getattr(lm, "y", None) or lm[1]
            # Convert normalized to pixel coordinates
            px = x * self.frame_width
            py = y * self.frame_height
            points.append([px, py])
        return np.array(points, dtype=np.float64)

    def update(
        self,
        landmarks: Sequence[object],
        timestamp: float,
    ) -> tuple[bool, float, float, float]:
        """
        Update head pose from face mesh landmarks.
        Returns (gaze_lost, time_in_zone_seconds, yaw_degrees, pitch_degrees).
        """
        img_pts = self._landmarks_to_image_points(landmarks)
        if img_pts is None:
            return True, self._time_in_zone_seconds, 0.0, 0.0
        success, rvec, tvec = cv2.solvePnP(
            MODEL_POINTS_3D,
            img_pts,
            self._camera_matrix,
            self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return True, self._time_in_zone_seconds, 0.0, 0.0
        pitch_deg, yaw_deg, _roll = _rotation_vector_to_euler(rvec)
        in_zone = (
            self.pitch_min <= pitch_deg <= self.pitch_max
            and self.yaw_min <= yaw_deg <= self.yaw_max
        )
        if in_zone:
            if self._in_zone_since is None:
                self._in_zone_since = timestamp
            self._out_of_zone_since = None
            self._time_in_zone_seconds = timestamp - (self._in_zone_since or timestamp)
            return False, self._time_in_zone_seconds, yaw_deg, pitch_deg
        else:
            if self._out_of_zone_since is None:
                self._out_of_zone_since = timestamp
            self._in_zone_since = None
            out_duration = timestamp - (self._out_of_zone_since or timestamp)
            gaze_lost = out_duration >= self.gaze_lost_seconds
            return gaze_lost, self._time_in_zone_seconds, yaw_deg, pitch_deg

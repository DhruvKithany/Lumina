"""
Biometric Heatmap: high-contrast HUD-style overlay of pose and face landmarks.

Draws skeletal lines and face mesh in Neon Green/Cyan for demo WOW factor.
Used with OpenCV imshow in a separate thread; does not touch TelemetryState.
Supports both legacy MediaPipe solutions and tasks API.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cv_engine.mediapipe_backend import (
    get_face_connections_for_draw,
    get_pose_connections,
    get_tasks_drawing_utils,
    has_holistic,
    has_tasks,
)

# HUD-style high-contrast colors (BGR for OpenCV)
CYAN_BGR = (255, 255, 0)
NEON_GREEN_BGR = (0, 255, 100)


def _get_drawing_specs():
    """Return (landmark_spec, connection_spec) for the active backend."""
    if has_holistic():
        import mediapipe as mp
        return (
            mp.solutions.drawing_utils.DrawingSpec(
                color=NEON_GREEN_BGR,
                thickness=1,
                circle_radius=2,
            ),
            mp.solutions.drawing_utils.DrawingSpec(
                color=CYAN_BGR,
                thickness=2,
                circle_radius=0,
            ),
        )
    if has_tasks():
        du = get_tasks_drawing_utils()
        return (
            du.DrawingSpec(color=NEON_GREEN_BGR, thickness=1, circle_radius=2),
            du.DrawingSpec(color=CYAN_BGR, thickness=2, circle_radius=0),
        )
    class DummySpec:
        color = NEON_GREEN_BGR
        thickness = 2
        circle_radius = 2
    return DummySpec(), DummySpec()


LANDMARK_SPEC, CONNECTION_SPEC = _get_drawing_specs()


def draw_heatmap(
    frame_bgr: np.ndarray,
    results: Any,
    window_title: str = "Lumina Biometric Heatmap",
) -> np.ndarray:
    """
    Draw pose and face landmarks on a copy of the frame in HUD style.

    Args:
        frame_bgr: BGR image from camera.
        results: Holistic or tasks result with pose_landmarks and/or face_landmarks (each with .landmark).
        window_title: Unused; kept for API consistency.

    Returns:
        New BGR image with landmarks drawn (original unchanged).
    """
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    if has_holistic():
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=LANDMARK_SPEC,
                connection_drawing_spec=CONNECTION_SPEC,
            )
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=LANDMARK_SPEC,
                connection_drawing_spec=CONNECTION_SPEC,
            )
            lms = getattr(results.face_landmarks, "landmark", [])
            if len(lms) >= 478:
                for idx in (468, 473):
                    lm = lms[idx]
                    px = int(getattr(lm, "x", 0) * w)
                    py = int(getattr(lm, "y", 0) * h)
                    cv2.circle(annotated, (px, py), 4, NEON_GREEN_BGR, -1)
    else:
        du = get_tasks_drawing_utils()
        pose_conns = get_pose_connections()
        face_conns = get_face_connections_for_draw()
        if results.pose_landmarks:
            plm = getattr(results.pose_landmarks, "landmark", None)
            if plm:
                du.draw_landmarks(
                    annotated,
                    plm,
                    pose_conns,
                    landmark_drawing_spec=LANDMARK_SPEC,
                    connection_drawing_spec=CONNECTION_SPEC,
                )
        if results.face_landmarks:
            flm = getattr(results.face_landmarks, "landmark", None)
            if flm:
                du.draw_landmarks(
                    annotated,
                    flm,
                    face_conns,
                    landmark_drawing_spec=LANDMARK_SPEC,
                    connection_drawing_spec=CONNECTION_SPEC,
                )
                if len(flm) >= 478:
                    for idx in (468, 473):
                        lm = flm[idx]
                        px = int(getattr(lm, "x", 0) * w)
                        py = int(getattr(lm, "y", 0) * h)
                        cv2.circle(annotated, (px, py), 4, NEON_GREEN_BGR, -1)

    return annotated

"""
MediaPipe backend: use legacy Holistic when available, else PoseLandmarker + FaceLandmarker (tasks API).

MediaPipe 0.10+ removed mp.solutions; this module allows the pipeline to run with either API.
"""

from __future__ import annotations

from typing import Any, NamedTuple

# Optional legacy Holistic (mediapipe < 0.10 or fork with solutions)
_holistic = None
try:
    import mediapipe as mp
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
        _holistic = mp.solutions.holistic
except (ImportError, AttributeError):
    pass

# Tasks API (mediapipe 0.10+)
_tasks_vision = None
_PoseLandmarksConnections = None
_FaceLandmarksConnections = None
_tasks_drawing_utils = None
_vision_task_running_mode = None
_mp_image = None
_base_options = None

try:
    from mediapipe.tasks.python.vision import (
        PoseLandmarker,
        PoseLandmarkerOptions,
        FaceLandmarker,
        FaceLandmarkerOptions,
        PoseLandmarksConnections,
        FaceLandmarksConnections,
        drawing_utils as _tasks_drawing_utils,
        core as vision_core,
    )
    from mediapipe.tasks.python.vision.core import image as _mp_image
    from mediapipe.tasks.python.core import base_options as _base_options
    _PoseLandmarksConnections = PoseLandmarksConnections
    _FaceLandmarksConnections = FaceLandmarksConnections
    _RunningMode = vision_core.vision_task_running_mode.VisionTaskRunningMode
    _tasks_vision = True
except ImportError:
    _RunningMode = None


def has_holistic() -> bool:
    return _holistic is not None


def has_tasks() -> bool:
    return _tasks_vision is True


class LandmarkWrapper(NamedTuple):
    """Wrapper so pose_landmarks.landmark and face_landmarks.landmark work like legacy."""
    landmark: list


def create_holistic(model_complexity: int = 0, refine_face_landmarks: bool = True) -> Any:
    if _holistic is None:
        raise RuntimeError("MediaPipe legacy solutions (Holistic) not available. Install mediapipe<0.10 or use tasks API.")
    return _holistic.Holistic(
        static_image_mode=False,
        model_complexity=model_complexity,
        refine_face_landmarks=refine_face_landmarks,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def create_pose_landmarker(model_path: str, model_complexity: int = 0) -> Any:
    if not _tasks_vision:
        raise RuntimeError("MediaPipe tasks vision not available.")
    opts = PoseLandmarkerOptions(
        base_options=_base_options.BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(opts)


def create_face_landmarker(model_path: str) -> Any:
    if not _tasks_vision:
        raise RuntimeError("MediaPipe tasks vision not available.")
    opts = FaceLandmarkerOptions(
        base_options=_base_options.BaseOptions(model_asset_path=model_path),
        running_mode=_RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(opts)


def process_frame_tasks(
    pose_landmarker: Any,
    face_landmarker: Any,
    rgb_image: Any,
) -> Any:
    """Run pose and face detection; return a result-like object with .pose_landmarks and .face_landmarks."""
    mp_img = _mp_image.Image(image_format=_mp_image.ImageFormat.SRGB, data=rgb_image)
    pose_result = pose_landmarker.detect(mp_img)
    face_result = face_landmarker.detect(mp_img)
    pose_list = pose_result.pose_landmarks[0] if pose_result.pose_landmarks else None
    face_list = face_result.face_landmarks[0] if face_result.face_landmarks else None
    class Result:
        pose_landmarks: Any = None
        face_landmarks: Any = None
    r = Result()
    r.pose_landmarks = LandmarkWrapper(pose_list) if pose_list else None
    r.face_landmarks = LandmarkWrapper(face_list) if face_list else None
    return r


def get_tasks_drawing_utils() -> Any:
    return _tasks_drawing_utils


def get_pose_connections() -> Any:
    return _PoseLandmarksConnections.POSE_LANDMARKS


def get_face_connections_for_draw() -> list:
    """All face connections for drawing (contours + iris)."""
    conns = []
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS)
    conns.extend(_FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS)
    return conns

"""
Camera capture and frame iteration for the CV pipeline.

Provides a context manager or iterator that yields BGR frames at a configured
resolution and FPS. Used by the pipeline to feed MediaPipe Pose and Face Mesh.
"""

from __future__ import annotations

import time
from typing import Generator

import cv2


class CameraCapture:
    """
    OpenCV VideoCapture wrapper with configurable size and FPS.
    Yields (frame_bgr, timestamp) for each read; timestamp is seconds since first frame.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        target_fps: float = 15.0,
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture | None = None
        self._start_time: float | None = None

    def __enter__(self) -> CameraCapture:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self.camera_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> tuple[bool, cv2.Mat | None, float]:
        """
        Read one frame. Returns (success, frame_bgr, timestamp_seconds).
        timestamp is relative to first frame.
        """
        if self._cap is None:
            return False, None, 0.0
        ok, frame = self._cap.read()
        t = (time.perf_counter() - (self._start_time or 0)) if self._start_time else 0.0
        return ok, frame, t

    def frames(self) -> Generator[tuple[cv2.Mat, float], None, None]:
        """Yield (frame_bgr, timestamp_seconds) until read fails."""
        while True:
            ok, frame, t = self.read()
            if not ok or frame is None:
                break
            yield frame, t

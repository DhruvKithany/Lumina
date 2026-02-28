"""
Computer vision pipeline: capture, pose-based kinesic entropy, and gaze tracking.

Modules:
- capture: OpenCV video capture and frame iteration
- pose_entropy: Micro-movement velocity and "fidget" detection from MediaPipe Pose
- gaze_tracker: Head pose (PnP) and golden-zone gaze-lost logic from Face Mesh
- pipeline: Runs Pose + Face on each frame and updates shared TelemetryState
"""

from cv_engine.capture import CameraCapture
from cv_engine.gaze_tracker import GazeTracker
from cv_engine.pipeline import CVPipeline
from cv_engine.pose_entropy import PoseEntropyTracker

__all__ = ["CameraCapture", "CVPipeline", "GazeTracker", "PoseEntropyTracker"]

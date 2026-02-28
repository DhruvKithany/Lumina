"""
Backend signal-processing unit: lightweight, variance/distance-based
detectors for gaze tracking and fidget detection.

These complement the more advanced cv_engine implementations with simpler
signal-processing approaches (iris distance vs. solvePnP, variance vs.
velocity entropy).
"""

from backend_signalprocessing_unit.fidget_detector import FidgetDetector
from backend_signalprocessing_unit.gaze_tracker import GazeTracker
from backend_signalprocessing_unit.smoothing_buffer import SmoothingBuffer

__all__ = ["FidgetDetector", "GazeTracker", "SmoothingBuffer"]

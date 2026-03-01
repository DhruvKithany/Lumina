"""
Thread-safe telemetry state and config loading.

This module defines TelemetryState (consumed by HUD, produced by CV/probes)
and load_config() for YAML settings. See core/__init__.py for public API.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TelemetryState:
    """
    Real-time performance telemetry produced by the CV engine and probes.
    All fields are updated by worker threads; readers must hold the lock.
    """

    # Kinesic entropy (pose)
    stability_score: float = 1.0  # 0 = high fidgeting, 1 = stable
    high_entropy: bool = False
    calibration_done: bool = False

    # Gaze
    gaze_lost: bool = False
    time_in_zone_seconds: float = 0.0
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    presentation_mode: str = "pitch"  # "pitch", "q&a", or "interview"
    reference_frame: str = "digital"  # "digital" (camera) or "irl" (audience above)

    # Backend signal-processing unit (BSP) outputs
    bsp_gaze_status: str = "LOCKED"  # "LOCKED", "WARNING", or "GAZE LOST"
    bsp_gaze_distance: float = 0.0  # iris deviation from center
    bsp_fidgeting: bool = False  # variance-based fidget flag
    bsp_fidget_variance: float = 0.0  # raw wrist-position variance

    # Knowledge probe
    probe_text: str = ""
    probe_visible: bool = False

    # Pipeline health
    last_frame_ok: bool = False
    fps: float = 0.0

    # Raw (x,y,z) landmark coordinates for irises, shoulders, wrists.
    raw_landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    # Heatmap overlay: when True, face landmarks are drawn on the heatmap window.
    show_face_heatmap: bool = True

    # Latest camera frame for HUD overlay
    latest_frame: Any = None

    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, **kwargs: Any) -> None:
        """Update one or more fields under lock."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k) and not k.startswith("_"):
                    setattr(self, k, v)

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of telemetry fields for the HUD (no lock held after return)."""
        with self._lock:
            return {
                "stability_score": self.stability_score,
                "high_entropy": self.high_entropy,
                "calibration_done": self.calibration_done,
                "gaze_lost": self.gaze_lost,
                "time_in_zone_seconds": self.time_in_zone_seconds,
                "yaw_degrees": self.yaw_degrees,
                "pitch_degrees": self.pitch_degrees,
                "presentation_mode": self.presentation_mode,
                "reference_frame": self.reference_frame,
                "bsp_gaze_status": self.bsp_gaze_status,
                "bsp_gaze_distance": self.bsp_gaze_distance,
                "bsp_fidgeting": self.bsp_fidgeting,
                "bsp_fidget_variance": self.bsp_fidget_variance,
                "probe_text": self.probe_text,
                "probe_visible": self.probe_visible,
                "last_frame_ok": self.last_frame_ok,
                "fps": self.fps,
                "raw_landmarks": dict(self.raw_landmarks),
                "show_face_heatmap": self.show_face_heatmap,
                "latest_frame": self.latest_frame,
            }


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """
    Load YAML config from path. If path is None, use config.yaml in project root.
    Returns a nested dict; missing keys should use defaults in each module.
    """
    if path is None:
        path = Path(__file__).resolve().parent.parent / "config.yaml"
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

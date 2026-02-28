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

    # Knowledge probe
    probe_text: str = ""
    probe_visible: bool = False

    # Pipeline health
    last_frame_ok: bool = False
    fps: float = 0.0

    # Raw (x,y,z) landmark coordinates for irises, shoulders, wrists.
    raw_landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)

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
                "probe_text": self.probe_text,
                "probe_visible": self.probe_visible,
                "last_frame_ok": self.last_frame_ok,
                "fps": self.fps,
                "raw_landmarks": dict(self.raw_landmarks),
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

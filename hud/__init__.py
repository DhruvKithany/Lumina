"""
Temporal HUD overlay ("Smart-Pilot" interface).

PyQt6 transparent, always-on-top window that displays performance telemetry:
Stability Meter and Gaze Duration Tracker. Reads from shared TelemetryState.
"""

from hud.overlay import HUDOverlay

__all__ = ["HUDOverlay"]

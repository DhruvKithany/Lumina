"""
PyQt6 transparent HUD overlay for Lumina-Presenter telemetry.

Displays:
- Stability meter (0–1, from kinesic entropy)
- Gaze duration and "Gaze Lost" indicator
- Optional probe text when cognitive stall triggers an injection

Window is frameless, transparent background, always on top, and optionally
click-through so the presenter can use PowerPoint/Zoom underneath.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from core.state import TelemetryState

if TYPE_CHECKING:
    pass


def _default_position_rect(
    width: int,
    height: int,
    position: str = "top_right",
    margin: int = 16,
    screen_geometry: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Return (x, y, width, height) for the HUD window."""
    if screen_geometry is None:
        screen_geometry = (0, 0, 1920, 1080)
    sx, sy, sw, sh = screen_geometry
    x = sx + sw - width - margin if "right" in position else sx + margin
    y = sy + margin if "top" in position else sy + sh - height - margin
    return x, y, width, height


class HUDOverlay(QWidget):
    """
    Transparent overlay showing stability meter, gaze duration, and probe text.
    Call start_update_timer(interval_ms) to periodically refresh from state.
    """

    def __init__(
        self,
        state: TelemetryState,
        width: int = 280,
        height: int = 200,
        position: str = "top_right",
        margin: int = 16,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self._width = width
        self._height = height
        self._position = position
        self._margin = margin
        self.setWindowTitle("Lumina-Presenter HUD")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setFixedSize(width, height)
        self._place_window()
        # Layout and widgets
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._stability_label = QLabel("Stability")
        self._stability_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self._stability_label)
        self._stability_bar = QProgressBar()
        self._stability_bar.setRange(0, 100)
        self._stability_bar.setValue(100)
        self._stability_bar.setStyleSheet(
            """
            QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center; }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #4ade80, stop:1 #22c55e); }
            """
        )
        layout.addWidget(self._stability_bar)
        self._gaze_label = QLabel("Gaze: 0.0s")
        self._gaze_label.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(self._gaze_label)
        self._gaze_lost_label = QLabel("")
        self._gaze_lost_label.setStyleSheet("color: #f87171; font-weight: bold;")
        layout.addWidget(self._gaze_lost_label)
        self._probe_label = QLabel("")
        self._probe_label.setWordWrap(True)
        self._probe_label.setStyleSheet("color: #fbbf24; font-size: 11px;")
        self._probe_label.setMaximumHeight(48)
        layout.addWidget(self._probe_label)
        layout.addStretch()
        self.setStyleSheet(
            "background-color: rgba(30, 30, 40, 220); border-radius: 8px; border: 1px solid #444;"
        )

    def _place_window(self) -> None:
        """Position window on screen."""
        screen = QApplication.primaryScreen()
        if screen:
            geom = screen.availableGeometry()
            rect = _default_position_rect(
                self._width,
                self._height,
                self._position,
                self._margin,
                (geom.x(), geom.y(), geom.width(), geom.height()),
            )
            self.setGeometry(rect[0], rect[1], rect[2], rect[3])

    def _refresh_from_state(self) -> None:
        """Read state and update widgets."""
        snap = self.state.snapshot()
        stability = int(snap["stability_score"] * 100)
        self._stability_bar.setValue(stability)
        if snap["high_entropy"]:
            self._stability_label.setText("Stability (fidgeting)")
        else:
            self._stability_label.setText("Stability")
        self._gaze_label.setText(f"Gaze: {snap['time_in_zone_seconds']:.1f}s")
        if snap["gaze_lost"]:
            self._gaze_lost_label.setText("⚠ Gaze Tracking Lost")
        else:
            self._gaze_lost_label.setText("")
        if snap["probe_visible"] and snap["probe_text"]:
            self._probe_label.setText(snap["probe_text"])
            self._probe_label.show()
        else:
            self._probe_label.setText("")
            self._probe_label.hide()

    def start_update_timer(self, interval_ms: int = 100) -> None:
        """Start a QTimer that refreshes the HUD from state every interval_ms."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_from_state)
        self._timer.start(interval_ms)

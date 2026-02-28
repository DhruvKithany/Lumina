"""
PyQt6 transparent HUD overlay for Lumina-Presenter telemetry.

Displays:
- Custom StabilityGauge: circular arc (0–100%) with high-entropy warning and FPS.
- GazeTracker: 2D radar of head yaw/pitch; shows "WARN: HOLD" when gaze is lost (Confidence Gate).
- Optional probe text with fade-in/fade-out animation.
- MODE toggle (digital vs IRL) and close button.

Window is frameless, transparent background, always on top. Data is read from
TelemetryState.snapshot() on a timer; the CV pipeline and probe injector update state.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QTimer, Qt, QPropertyAnimation, QRectF, QPointF, QEasingCurve
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect,
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


class StabilityGauge(QWidget):
    """Custom circular glow gauge for Stability Score."""
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(120, 120)
        self.value = 1.0
        self.high_entropy = False
        self.fps = 0.0
        self.last_frame_ok = False
        
    def set_value(
        self,
        value: float,
        high_entropy: bool,
        fps: float,
        last_frame_ok: bool,
    ) -> None:
        """Update gauge from telemetry: stability [0,1], fidget flag, FPS, and pipeline health."""
        self.value = value
        self.high_entropy = high_entropy
        self.fps = fps
        self.last_frame_ok = last_frame_ok
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = QRectF(10, 10, 100, 100)
        
        # Draw background track
        painter.setPen(QPen(QColor(40, 45, 55, 180), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 135 * 16, -270 * 16)
        
        # Active Value Color
        if self.high_entropy:
            color = QColor(255, 60, 60) # Red
        elif self.value > 0.7:
            color = QColor(60, 255, 120) # Green/Cyan
        else:
            color = QColor(255, 180, 60) # Amber
            
        # Draw active value arc
        painter.setPen(QPen(color, 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        span_angle = int(-270 * self.value * 16)
        painter.drawArc(rect, 135 * 16, span_angle)
        
        # Draw percentage in center
        painter.setPen(color)
        font = QFont("Consolas", 16, QFont.Weight.Bold)
        painter.setFont(font)
        text = f"{int(self.value * 100)}%"
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        
        # Draw 'STABILITY' label
        font_small = QFont("Consolas", 7, QFont.Weight.Bold)
        painter.setFont(font_small)
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(QRectF(10, 80, 100, 20), Qt.AlignmentFlag.AlignCenter, "STABILITY")
        
        # Draw core system status dots on bottom left/right
        ind_size = 4
        # System health indicator
        health_color = QColor(60, 255, 120) if self.last_frame_ok else QColor(255, 60, 60)
        painter.setBrush(QBrush(health_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(20, 100), ind_size, ind_size)
        
        # FPS readout next to dot
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(QRectF(30, 95, 40, 10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{int(self.fps)}hz")


class GazeTracker(QWidget):
    """Custom 2D Radar view of Gaze Pitch/Yaw."""
    
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(140, 120)
        self.yaw = 0.0
        self.pitch = 0.0
        self.gaze_lost = False
        self.duration = 0.0
        self.mode = "digital"
        
    def set_gaze(
        self,
        yaw: float,
        pitch: float,
        gaze_lost: bool,
        duration: float,
        mode: str,
    ) -> None:
        """
        Update radar from telemetry. When gaze_lost is True, shows "WARN: HOLD" and red indicator.
        mode is "digital" (camera) or "irl" (audience above); affects target box offset.
        """
        self.yaw = yaw
        self.pitch = pitch
        self.gaze_lost = gaze_lost
        self.duration = duration
        self.mode = mode
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = QRectF(20, 10, 100, 80)
        color = QColor(255, 60, 60) if self.gaze_lost else QColor(100, 200, 255)
        
        # Draw background grid
        painter.setPen(QPen(QColor(40, 45, 55, 150), 1))
        painter.setBrush(QBrush(QColor(20, 25, 30, 200)))
        painter.drawRect(rect)
        
        # Crosshairs
        painter.setPen(QPen(QColor(100, 200, 255, 50), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(rect.center().x(), rect.top()), QPointF(rect.center().x(), rect.bottom()))
        painter.drawLine(QPointF(rect.left(), rect.center().y()), QPointF(rect.right(), rect.center().y()))
        
        # Target Box
        painter.setPen(QPen(QColor(100, 200, 255, 100), 1))
        
        # Shift target box up if in IRL mode (looking above camera)
        target_y_offset = -15 if self.mode == "irl" else 0
        target_rect = QRectF(rect.center().x() - 25, rect.center().y() - 20 + target_y_offset, 50, 40)
        painter.drawRect(target_rect)
        
        # Clamp pitch and yaw for radar dot
        # Assume max yaw is ~30 deg, max pitch is ~20 deg
        dx = max(-1.0, min(1.0, self.yaw / 30.0))
        dy = max(-1.0, min(1.0, self.pitch / 20.0))
        
        # Dot position
        cx = rect.center().x() + dx * (rect.width() / 2)
        cy = rect.center().y() + dy * (rect.height() / 2)
        
        # Draw glow
        glow_radius = 8 if self.gaze_lost else 6
        glow_color = QColor(color)
        glow_color.setAlpha(100)
        painter.setBrush(QBrush(glow_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), glow_radius, glow_radius)
        
        # Draw dot
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(cx, cy), 3, 3)
        
        # Draw duration label and lock status
        painter.setPen(color)
        font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font)
        status = "WARN: HOLD" if self.gaze_lost else "TRK: LOCK"
        painter.drawText(QRectF(20, 95, 100, 15), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{status}")
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(QRectF(20, 95, 100, 15), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{self.duration:.1f}s")


class HUDOverlay(QWidget):
    """
    Transparent overlay showing StabilityGauge, GazeTracker, and knowledge probe.

    Refreshes from TelemetryState.snapshot() on a timer. Use start_update_timer(interval_ms)
    after show() to begin updates.
    """

    def __init__(
        self,
        state: TelemetryState,
        width: int = 340,
        height: int = 240,
        position: str = "top_right",
        margin: int = 24,
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
        
        # Layouts
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)
        
        # Top Row: Gauges
        gauge_layout = QHBoxLayout()
        gauge_layout.setSpacing(10)
        
        self._stability_gauge = StabilityGauge()
        self._gaze_tracker = GazeTracker()
        
        gauge_layout.addWidget(self._stability_gauge)
        gauge_layout.addWidget(self._gaze_tracker)
        gauge_layout.addStretch()
        
        # Add a neon glow to the entire top row
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(25)
        glow.setColor(QColor(56, 189, 248, 80)) # Light blue glow
        glow.setOffset(0, 0)
        
        # We need a wrapper widget to apply the effect to the layout
        gauge_wrapper = QWidget()
        gauge_wrapper.setLayout(gauge_layout)
        gauge_wrapper.setGraphicsEffect(glow)
        main_layout.addWidget(gauge_wrapper)
        
        # Status/Controls Row
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        
        # Close Button
        self._close_btn = QPushButton("X")
        self._close_btn.setFixedSize(20, 20)
        self._close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close_btn.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #94a3b8;
                border: 1px solid transparent;
                border-radius: 4px;
                font-family: Consolas;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: rgba(248, 113, 113, 100);
                color: white;
            }
            """
        )
        self._close_btn.clicked.connect(QApplication.instance().quit)
        
        controls_layout.addWidget(self._close_btn)
        
        self._mode_btn = QPushButton("MODE: DIGITAL")
        self._mode_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(40, 45, 55, 180);
                color: #38bdf8;
                border: 1px solid #38bdf8;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: Consolas;
                font-weight: bold;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 50);
            }
            QPushButton:pressed {
                background-color: rgba(56, 189, 248, 100);
            }
            """
        )
        self._mode_btn.clicked.connect(self._toggle_mode)
        
        controls_layout.addWidget(self._mode_btn)
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # BSP Signal Processing Row
        bsp_layout = QHBoxLayout()
        bsp_layout.setSpacing(6)
        
        self._bsp_iris_label = QLabel("IRIS: LOCK")
        self._bsp_iris_label.setStyleSheet(
            "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        bsp_layout.addWidget(self._bsp_iris_label)
        
        self._bsp_fidget_label = QLabel("● CALM")
        self._bsp_fidget_label.setStyleSheet(
            "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        bsp_layout.addWidget(self._bsp_fidget_label)
        bsp_layout.addStretch()
        main_layout.addLayout(bsp_layout)
        
        # Spacer
        main_layout.addStretch()
        
        # Bottom Row: Knowledge Probe
        self._probe_label = QLabel("")
        self._probe_label.setWordWrap(True)
        self._probe_label.setStyleSheet(
            "color: #fbbf24; background-color: rgba(30, 30, 30, 150); "
            "border: 1px solid #fbbf24; border-radius: 4px; padding: 8px; font-family: Consolas; font-size: 11px;"
        )
        self._probe_label.setMinimumHeight(48)
        self._probe_label.hide()
        
        # Setup fade animation for probe
        self._probe_opacity = QGraphicsOpacityEffect(self._probe_label)
        self._probe_label.setGraphicsEffect(self._probe_opacity)
        self._probe_animation = QPropertyAnimation(self._probe_opacity, b"opacity")
        self._probe_animation.setDuration(400)
        self._probe_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        
        main_layout.addWidget(self._probe_label)
        
        # Set overall background to HUD
        self.setStyleSheet(
            """
            HUDOverlay { 
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 rgba(10, 15, 25, 230), stop:1 rgba(20, 25, 35, 240)); 
                border-radius: 12px; 
                border: 1px solid rgba(56, 189, 248, 100); 
            }
            """
        )

    def paintEvent(self, event):
        # Allow stylesheet background to be drawn for QWidget subclasses
        import PyQt6.QtWidgets as QtWidgets
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        p = QPainter(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PrimitiveElement.PE_Widget, opt, p, self)

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
        """
        Read current telemetry from shared state and update HUD widgets.

        Drives the custom StabilityGauge (arc + percentage) and GazeTracker (2D radar)
        from snapshot(). When gaze_lost is True, GazeTracker shows "WARN: HOLD" and red
        indicator. Probe text is shown/hidden with fade animation.
        """
        snap = self.state.snapshot()

        self._stability_gauge.set_value(
            snap["stability_score"],
            snap["high_entropy"],
            snap["fps"],
            snap["last_frame_ok"],
        )

        self._gaze_tracker.set_gaze(
            snap["yaw_degrees"],
            snap["pitch_degrees"],
            snap["gaze_lost"],
            snap["time_in_zone_seconds"],
            snap.get("presentation_mode", "digital"),
        )

        # BSP indicators
        bsp_status = snap.get("bsp_gaze_status", "LOCKED")
        if bsp_status == "LOCKED":
            self._bsp_iris_label.setText("IRIS: LOCK")
            self._bsp_iris_label.setStyleSheet(
                "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        elif bsp_status == "WARNING":
            self._bsp_iris_label.setText("IRIS: WARN")
            self._bsp_iris_label.setStyleSheet(
                "color: #fbbf24; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        else:
            self._bsp_iris_label.setText("IRIS: LOST")
            self._bsp_iris_label.setStyleSheet(
                "color: #f87171; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        
        if snap.get("bsp_fidgeting", False):
            self._bsp_fidget_label.setText("● FIDGET")
            self._bsp_fidget_label.setStyleSheet(
                "color: #f87171; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        else:
            self._bsp_fidget_label.setText("● CALM")
            self._bsp_fidget_label.setStyleSheet(
                "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )

        # Handle probe visibility with fade-in / fade-out animation
        if snap["probe_visible"] and snap["probe_text"]:
            if self._probe_label.text() != snap["probe_text"] or self._probe_label.isHidden():
                self._probe_label.setText(snap["probe_text"])
                self._probe_label.show()
                # Fade in
                self._probe_animation.stop()
                self._probe_animation.setStartValue(self._probe_opacity.opacity())
                self._probe_animation.setEndValue(1.0)
                self._probe_animation.start()
        else:
            if not self._probe_label.isHidden() and self._probe_opacity.opacity() == 1.0:
                # Fade out
                self._probe_animation.stop()
                self._probe_animation.setStartValue(1.0)
                self._probe_animation.setEndValue(0.0)
                self._probe_animation.finished.connect(self._hide_probe_after_fade)
                self._probe_animation.start()

    def _toggle_mode(self) -> None:
        snap = self.state.snapshot()
        new_mode = "irl" if snap["presentation_mode"] == "digital" else "digital"
        self.state.update(presentation_mode=new_mode)
        
        if new_mode == "irl":
            self._mode_btn.setText("MODE: IRL")
            self._mode_btn.setStyleSheet(self._mode_btn.styleSheet().replace("#38bdf8", "#a78bfa")) # Purple for IRL
        else:
            self._mode_btn.setText("MODE: DIGITAL")
            self._mode_btn.setStyleSheet(self._mode_btn.styleSheet().replace("#a78bfa", "#38bdf8")) # Blue for Digital

    def _hide_probe_after_fade(self):
        # Only hide if opacity reached 0 (meaning animation didn't get reversed)
        if self._probe_opacity.opacity() == 0.0:
            self._probe_label.hide()
            try:
                self._probe_animation.finished.disconnect(self._hide_probe_after_fade)
            except Exception:
                pass # Already disconnected

    def start_update_timer(self, interval_ms: int = 100) -> None:
        """Start a QTimer that refreshes the HUD from state every interval_ms."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_from_state)
        self._timer.start(interval_ms)

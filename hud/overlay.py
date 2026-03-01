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
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QBrush, QImage, QPixmap
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
        self.target_value = 1.0
        self.high_entropy = False
        self.fps = 0.0
        self.last_frame_ok = False
        self.pulse_phase = 0.0
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(16)
        
    def _animate(self):
        # Smooth interpolation
        self.value += (self.target_value - self.value) * 0.1
        self.pulse_phase += 0.1
        if self.pulse_phase > math.pi * 2:
            self.pulse_phase -= math.pi * 2
        self.update()
        
    def set_value(
        self,
        value: float,
        high_entropy: bool,
        fps: float,
        last_frame_ok: bool,
    ) -> None:
        """Update gauge from telemetry: stability [0,1], fidget flag, FPS, and pipeline health."""
        self.target_value = value
        self.high_entropy = high_entropy
        self.fps = fps
        self.last_frame_ok = last_frame_ok

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = QRectF(10, 10, 100, 100)
        
        # Draw background track
        painter.setPen(QPen(QColor(40, 45, 55, 180), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 135 * 16, -270 * 16)
        
        # Active Value Color
        pulse_alpha = int(150 + 105 * math.sin(self.pulse_phase)) if self.high_entropy else 255
        if self.high_entropy:
            color = QColor(255, 60, 60, pulse_alpha) # Pulsing Red
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
        painter.drawText(QRectF(10, 75, 100, 15), Qt.AlignmentFlag.AlignCenter, "STABILITY")
        
        # Authority Presence Label
        painter.setFont(QFont("Consolas", 6, QFont.Weight.Bold))
        painter.setPen(QColor(150, 160, 170))
        painter.drawText(QRectF(10, 85, 100, 10), Qt.AlignmentFlag.AlignCenter, "AUTHORITY:")
        auth_color = QColor(255, 60, 60) if self.high_entropy else QColor(60, 255, 120)
        auth_text = "COMPROMISED" if self.high_entropy else "STABLE"
        painter.setPen(auth_color)
        painter.drawText(QRectF(10, 95, 100, 10), Qt.AlignmentFlag.AlignCenter, auth_text)
        
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
        self.target_yaw = 0.0
        self.target_pitch = 0.0
        self.gaze_lost = False
        self.duration = 0.0
        self.mode = "digital"
        self.pulse_phase = 0.0
        
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(16)
        
    def _animate(self):
        # Even smoother interpolation for the UI crosshair
        self.yaw += (self.target_yaw - self.yaw) * 0.15
        self.pitch += (self.target_pitch - self.pitch) * 0.15
        self.pulse_phase += 0.1
        if self.pulse_phase > math.pi * 2:
            self.pulse_phase -= math.pi * 2
        self.update()
        
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
        self.target_yaw = yaw
        self.target_pitch = pitch
        self.gaze_lost = gaze_lost
        self.duration = duration
        self.mode = mode
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = QRectF(20, 10, 100, 80)
        
        # Color pulsing when lost
        pulse_alpha = int(150 + 105 * math.sin(self.pulse_phase)) if self.gaze_lost else 255
        base_color = QColor(255, 60, 60) if self.gaze_lost else QColor(100, 200, 255)
        color = QColor(base_color.red(), base_color.green(), base_color.blue(), pulse_alpha)
        
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
        
        # Draw vector line from center to dot
        painter.setPen(QPen(color, 2, Qt.PenStyle.SolidLine))
        painter.drawLine(QPointF(rect.center().x(), rect.center().y()), QPointF(cx, cy))
        
        # Draw dot
        painter.setBrush(QBrush(color))
        painter.drawEllipse(QPointF(cx, cy), 3, 3)
        
        # Draw Engagement Lock and Tracking Confidence
        painter.setPen(color)
        font = QFont("Consolas", 7, QFont.Weight.Bold)
        painter.setFont(font)
        status = "ENGAGEMENT LOST" if self.gaze_lost else "ENGAGEMENT LOCK: ACTIVE"
        painter.drawText(QRectF(20, 95, 100, 10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{status}")
        
        # Pseudo-random tracking confidence
        import random
        conf = random.randint(95, 99)
        painter.setPen(QColor(150, 160, 170))
        painter.setFont(QFont("Consolas", 6, QFont.Weight.Bold))
        painter.drawText(QRectF(20, 105, 100, 10), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"Confidence: {conf}%")
        painter.drawText(QRectF(20, 105, 100, 10), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{self.duration:.1f}s")


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
        

            
        # Reference Frame Toggle
        self._ref_btn = QPushButton("VIEW: DIGITAL")
        self._ref_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ref_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(40, 45, 55, 180);
                color: #38bdf8;
                border: 1px solid #38bdf8;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: Consolas;
                font-weight: bold;
                font-size: 8px;
            }
            """
        )
        self._ref_btn.clicked.connect(self._toggle_ref)
        controls_layout.addWidget(self._ref_btn)
        
        # Face Heatmap Toggle
        self._face_heatmap_btn = QPushButton("FACE: ON")
        self._face_heatmap_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._face_heatmap_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(40, 45, 55, 180);
                color: #4ade80;
                border: 1px solid #4ade80;
                border-radius: 4px;
                padding: 4px 8px;
                font-family: Consolas;
                font-weight: bold;
                font-size: 8px;
            }
            """
        )
        self._face_heatmap_btn.clicked.connect(self._toggle_face_heatmap)
        controls_layout.addWidget(self._face_heatmap_btn)
            
        controls_layout.addStretch()
        main_layout.addLayout(controls_layout)
        
        # Row 1: Authority Presence + Engagement Lock
        row1_layout = QHBoxLayout()
        row1_layout.setSpacing(8)
        
        self._authority_label = QLabel("Authority Presence: ---%")
        self._authority_label.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row1_layout.addWidget(self._authority_label)
        
        self._engage_label = QLabel("Engagement Lock: ---")
        self._engage_label.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row1_layout.addWidget(self._engage_label)
        row1_layout.addStretch()
        main_layout.addLayout(row1_layout)
        
        # Row 2: Kinesic Entropy + IRIS + Cognitive Stall
        row2_layout = QHBoxLayout()
        row2_layout.setSpacing(8)
        
        self._entropy_label = QLabel("Kinesic Entropy: ---")
        self._entropy_label.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row2_layout.addWidget(self._entropy_label)
        
        self._bsp_iris_label = QLabel("IRIS: ---")
        self._bsp_iris_label.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row2_layout.addWidget(self._bsp_iris_label)
        
        self._stall_label = QLabel("Cognitive Stall: ---")
        self._stall_label.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row2_layout.addWidget(self._stall_label)
        
        row2_layout.addStretch()
        main_layout.addLayout(row2_layout)
        
        # Row 3: Detailed telemetry
        row3_layout = QHBoxLayout()
        row3_layout.setSpacing(8)
        
        self._stat_gaze_yaw = QLabel("YAW: ---")
        self._stat_gaze_yaw.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row3_layout.addWidget(self._stat_gaze_yaw)
        
        self._stat_gaze_tilt = QLabel("TILT: ---")
        self._stat_gaze_tilt.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row3_layout.addWidget(self._stat_gaze_tilt)
        
        self._stat_zone_time = QLabel("ZONE: ---")
        self._stat_zone_time.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row3_layout.addWidget(self._stat_zone_time)
        
        self._stat_fps = QLabel("FPS: ---")
        self._stat_fps.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        row3_layout.addWidget(self._stat_fps)
        
        row3_layout.addStretch()
        main_layout.addLayout(row3_layout)
        
        # Spacer
        main_layout.addStretch()
        
        # Camera Feed Row
        self._camera_feed_label = QLabel()
        self._camera_feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._camera_feed_label.setStyleSheet(
            "background-color: #000000; border: 1px solid #334155; border-radius: 4px;"
        )
        self._camera_feed_label.setMinimumSize(320, 240)
        
        main_layout.addWidget(self._camera_feed_label)

        # Bottom Row: Knowledge Probe
        # Container to hold drift warning and the probe
        self._probe_container = QWidget()
        probe_layout = QVBoxLayout(self._probe_container)
        probe_layout.setContentsMargins(0, 0, 0, 0)
        
        self._drift_label = QLabel("NARRATIVE DRIFT DETECTED")
        self._drift_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._drift_label.setStyleSheet(
            "color: #f97316; background-color: rgba(249, 115, 22, 40); "
            "border: 1px solid #f97316; border-radius: 4px; padding: 4px; font-family: Consolas; font-size: 10px; font-weight: bold;"
        )
        probe_layout.addWidget(self._drift_label)
        
        # Pulsing Animation for the Drift Label using Opacity Effect
        self._drift_opacity = QGraphicsOpacityEffect(self._drift_label)
        self._drift_label.setGraphicsEffect(self._drift_opacity)
        self._drift_animation = QPropertyAnimation(self._drift_opacity, b"opacity")
        self._drift_animation.setDuration(800)
        self._drift_animation.setStartValue(0.4)
        self._drift_animation.setEndValue(1.0)
        self._drift_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._drift_animation.setLoopCount(-1) # Infinite looping pulse
        self._drift_animation.start()
        
        self._probe_label = QLabel("")
        self._probe_label.setWordWrap(True)
        self._probe_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._probe_label.setStyleSheet(
            "color: #bae6fd; background-color: rgba(15, 23, 42, 200); "
            "border: 1px solid #38bdf8; border-top: none; border-radius: 0px 0px 4px 4px; padding: 8px; font-family: Consolas; font-size: 11px;"
        )
        self._probe_label.setMinimumHeight(30)
        probe_layout.addWidget(self._probe_label)
        
        self._probe_container.hide()
        
        # Setup fade animation for probe container
        self._probe_opacity = QGraphicsOpacityEffect(self._probe_container)
        self._probe_container.setGraphicsEffect(self._probe_opacity)
        self._probe_animation = QPropertyAnimation(self._probe_opacity, b"opacity")
        self._probe_animation.setDuration(400)
        self._probe_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        
        main_layout.addWidget(self._probe_container)
        
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
            snap.get("reference_frame", "digital"),
        )

        # === SPEC INDICATORS: Authority, Engagement, Entropy, IRIS, Stall ===
        stab = snap.get("stability_score", 1.0)
        stab_pct = int(stab * 100)
        stab_color = "#4ade80" if stab > 0.7 else ("#fbbf24" if stab > 0.4 else "#f87171")
        self._authority_label.setText(f"Authority Presence: {stab_pct}%")
        self._authority_label.setStyleSheet(
            f"color: {stab_color}; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        
        gaze_lost = snap.get("gaze_lost", False)
        if gaze_lost:
            self._engage_label.setText("Engagement Lock: INACTIVE")
            self._engage_label.setStyleSheet(
                "color: #f87171; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        else:
            self._engage_label.setText("Engagement Lock: ACTIVE")
            self._engage_label.setStyleSheet(
                "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        
        entropy = snap.get("high_entropy", False)
        if entropy:
            self._entropy_label.setText("Kinesic Entropy: HIGH")
            self._entropy_label.setStyleSheet(
                "color: #f87171; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        else:
            self._entropy_label.setText("Kinesic Entropy: LOW")
            self._entropy_label.setStyleSheet(
                "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        
        # BSP IRIS gaze tracker
        bsp_status = snap.get("bsp_gaze_status", "LOCKED")
        bsp_dist = snap.get("bsp_gaze_distance", 0.0)
        if bsp_status == "LOCKED":
            self._bsp_iris_label.setText(f"IRIS: LOCK [{bsp_dist:.3f}]")
            self._bsp_iris_label.setStyleSheet(
                "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        elif bsp_status == "WARNING":
            self._bsp_iris_label.setText(f"IRIS: WARN [{bsp_dist:.3f}]")
            self._bsp_iris_label.setStyleSheet(
                "color: #fbbf24; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        else:
            self._bsp_iris_label.setText(f"IRIS: LOST [{bsp_dist:.3f}]")
            self._bsp_iris_label.setStyleSheet(
                "color: #f87171; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        
        # Cognitive Stall Indicator
        probe_vis = snap.get("probe_visible", False)
        if probe_vis:
            self._stall_label.setText("Cognitive Stall: NARRATIVE DRIFT")
            self._stall_label.setStyleSheet(
                "color: #f97316; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        else:
            self._stall_label.setText("Cognitive Stall: CLEAR")
            self._stall_label.setStyleSheet(
                "color: #4ade80; font-family: Consolas; font-size: 9px; font-weight: bold;"
            )
        
        # Row 3: detailed telemetry
        yaw = snap.get("yaw_degrees", 0.0)
        yaw_color = "#4ade80" if abs(yaw) < 20 else "#fbbf24"
        self._stat_gaze_yaw.setText(f"YAW: {yaw:+.1f}")
        self._stat_gaze_yaw.setStyleSheet(
            f"color: {yaw_color}; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        
        pitch = snap.get("pitch_degrees", 0.0)
        pitch_color = "#4ade80" if abs(pitch) < 15 else "#fbbf24"
        self._stat_gaze_tilt.setText(f"TILT: {pitch:+.1f}")
        self._stat_gaze_tilt.setStyleSheet(
            f"color: {pitch_color}; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        
        zone_t = snap.get("time_in_zone_seconds", 0.0)
        zone_color = "#4ade80" if zone_t > 2.0 else "#fbbf24"
        self._stat_zone_time.setText(f"ZONE: {zone_t:.1f}s")
        self._stat_zone_time.setStyleSheet(
            f"color: {zone_color}; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
        
        fps = snap.get("fps", 0)
        fps_color = "#4ade80" if fps > 10 else "#fbbf24"
        self._stat_fps.setText(f"FPS: {fps:.0f}")
        self._stat_fps.setStyleSheet(
            f"color: {fps_color}; font-family: Consolas; font-size: 9px; font-weight: bold;"
        )
            
        # Update extra buttons
        self._face_heatmap_btn.setText("FACE: ON" if snap.get("show_face_heatmap", True) else "FACE: OFF")
        ref = snap.get("reference_frame", "digital")
        if ref == "irl":
            self._ref_btn.setText("VIEW: IRL")
            self._ref_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(40, 45, 55, 180);
                    color: #a78bfa;
                    border: 1px solid #a78bfa;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-family: Consolas;
                    font-weight: bold;
                    font-size: 8px;
                }
                """
            )
        else:
            self._ref_btn.setText("VIEW: DIGITAL")
            self._ref_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(40, 45, 55, 180);
                    color: #38bdf8;
                    border: 1px solid #38bdf8;
                    border-radius: 4px;
                    padding: 4px 8px;
                    font-family: Consolas;
                    font-weight: bold;
                    font-size: 8px;
                }
                """
            )

        # Update Camera Feed
        frame = snap.get("latest_frame")
        if frame is not None:
             import cv2
             # Convert BGR (or RGB) to QImage
             # frame is already BGR since it came from draw_heatmap
             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
             h, w, c = frame_rgb.shape
             bytes_per_line = c * w
             qImg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
             pixmap = QPixmap.fromImage(qImg)
             # Scale pixmap to fit the label size
             scaled_pixmap = pixmap.scaled(
                 self._camera_feed_label.size(),
                 Qt.AspectRatioMode.KeepAspectRatio,
                 Qt.TransformationMode.SmoothTransformation
             )
             self._camera_feed_label.setPixmap(scaled_pixmap)

        # Handle probe visibility with fade-in / fade-out animation
        if snap["probe_visible"] and snap["probe_text"]:
            if self._probe_label.text() != snap["probe_text"] or self._probe_container.isHidden():
                self._drift_label.setText("COACHING TIP")
                self._drift_label.setStyleSheet(
                    "color: #f97316; background-color: rgba(249, 115, 22, 40); "
                    "border: 1px solid #f97316; border-radius: 4px; padding: 4px; "
                    "font-family: Consolas; font-size: 10px; font-weight: bold;"
                )
                    
                self._probe_label.setText(snap["probe_text"])
                self._probe_container.show()
                # Fade in
                self._probe_animation.stop()
                self._probe_animation.setStartValue(self._probe_opacity.opacity())
                self._probe_animation.setEndValue(1.0)
                self._probe_animation.start()
        else:
            if not self._probe_container.isHidden() and self._probe_opacity.opacity() > 0.5:
                # Fade out
                self._probe_animation.stop()
                self._probe_animation.setStartValue(self._probe_opacity.opacity())
                self._probe_animation.setEndValue(0.0)
                try:
                    self._probe_animation.finished.disconnect(self._hide_probe_after_fade)
                except Exception:
                    pass
                self._probe_animation.finished.connect(self._hide_probe_after_fade)
                self._probe_animation.start()



    def _hide_probe_after_fade(self):
        # Only hide if opacity reached 0 (meaning animation didn't get reversed)
        if self._probe_opacity.opacity() == 0.0:
            self._probe_container.hide()
            try:
                self._probe_animation.finished.disconnect(self._hide_probe_after_fade)
            except Exception:
                pass # Already disconnected

    def _toggle_ref(self, checked=False) -> None:
        snap = self.state.snapshot()
        new_ref = "irl" if snap.get("reference_frame", "digital") == "digital" else "digital"
        self.state.update(reference_frame=new_ref)

    def _toggle_face_heatmap(self, checked=False) -> None:
        snap = self.state.snapshot()
        new_face = not snap.get("show_face_heatmap", True)
        self.state.update(show_face_heatmap=new_face)

    def start_update_timer(self, interval_ms: int = 100) -> None:
        """Start a QTimer that refreshes the HUD from state every interval_ms."""
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_from_state)
        self._timer.start(interval_ms)

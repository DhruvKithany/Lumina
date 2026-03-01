"""
Zoom-style Microphone Test Dialog for Lumina.

Shows a live audio level meter and lets the user record + transcribe
a short clip to verify their mic and speech-to-text pipeline work.
"""

from __future__ import annotations

import struct
import threading
import time
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QLinearGradient
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
    QComboBox,
)

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


# ── Audio Level Meter Widget ────────────────────────────────────

class AudioLevelMeter(QWidget):
    """Horizontal bar that visualizes microphone input level (0.0–1.0)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(28)
        self.setMinimumWidth(200)
        self._level = 0.0

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        p.setBrush(QColor(15, 23, 42))
        p.setPen(QColor(51, 65, 85))
        p.drawRoundedRect(0, 0, w, h, 6, 6)

        # Level bar with gradient (green → yellow → red)
        bar_w = int(self._level * (w - 4))
        if bar_w > 0:
            grad = QLinearGradient(2, 0, w - 2, 0)
            grad.setColorAt(0.0, QColor(74, 222, 128))    # green
            grad.setColorAt(0.6, QColor(250, 204, 21))     # yellow
            grad.setColorAt(1.0, QColor(248, 113, 113))    # red
            p.setBrush(grad)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(2, 2, bar_w, h - 4, 4, 4)

        # Tick marks
        p.setPen(QColor(51, 65, 85, 120))
        for frac in (0.25, 0.5, 0.75):
            x = int(2 + frac * (w - 4))
            p.drawLine(x, 4, x, h - 4)

        p.end()


# ── Signal bridge (thread → Qt) ─────────────────────────────────

class _Signals(QObject):
    level_changed = pyqtSignal(float)
    status_changed = pyqtSignal(str)
    transcription_ready = pyqtSignal(str)
    test_finished = pyqtSignal(bool)


# ── Mic Test Dialog ─────────────────────────────────────────────

class MicTestDialog(QDialog):
    """
    Zoom-style microphone test dialog.

    Shows:
    - Device selector dropdown
    - Live audio level meter
    - "Test Mic" / "Stop" toggle button
    - Status messages
    - Transcription result
    """

    def __init__(self, parent=None, device_index=None):
        super().__init__(parent)
        self.setWindowTitle("Microphone Test")
        self.setFixedSize(440, 340)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
            }
            QLabel {
                color: #94a3b8;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QLabel#title {
                color: #f8fafc;
                font-size: 16px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            QLabel#status {
                color: #38bdf8;
                font-size: 12px;
                font-weight: bold;
            }
            QLabel#result {
                color: #4ade80;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
                background-color: rgba(74, 222, 128, 0.08);
                border: 1px solid rgba(74, 222, 128, 0.3);
                border-radius: 6px;
            }
            QLabel#fail {
                color: #f87171;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #38bdf8;
                border-radius: 6px;
                padding: 10px 24px;
                font-family: Consolas;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 0.15);
            }
            QPushButton:disabled {
                color: #475569;
                border-color: #475569;
            }
            QPushButton#recording {
                color: #f87171;
                border-color: #f87171;
            }
            QComboBox {
                background-color: #1e293b;
                color: #38bdf8;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 6px;
                font-family: Consolas;
                font-size: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e293b;
                color: #38bdf8;
                selection-background-color: rgba(56, 189, 248, 0.2);
            }
        """)

        self._device_index = device_index
        self._signals = _Signals()
        self._listening = False
        self._thread = None
        self._stop_event = threading.Event()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        # Title
        title = QLabel("🎤 MICROPHONE TEST")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Device selector
        dev_layout = QHBoxLayout()
        dev_label = QLabel("Input Device:")
        dev_layout.addWidget(dev_label)
        self._dev_combo = QComboBox()
        self._populate_devices()
        dev_layout.addWidget(self._dev_combo, stretch=1)
        layout.addLayout(dev_layout)

        # Level meter
        meter_label = QLabel("Audio Level:")
        layout.addWidget(meter_label)
        self._meter = AudioLevelMeter()
        layout.addWidget(self._meter)

        # Status
        self._status_label = QLabel("Press 'Test Mic' and speak into your microphone.")
        self._status_label.setObjectName("status")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Result area
        self._result_label = QLabel("")
        self._result_label.setObjectName("result")
        self._result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._result_label.setWordWrap(True)
        self._result_label.hide()
        layout.addWidget(self._result_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self._test_btn = QPushButton("🎤 Test Mic")
        self._test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._test_btn.clicked.connect(self._toggle_test)
        btn_layout.addWidget(self._test_btn)

        close_btn = QPushButton("Close")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()

        # Connect signals
        self._signals.level_changed.connect(self._meter.set_level)
        self._signals.status_changed.connect(self._on_status)
        self._signals.transcription_ready.connect(self._on_transcription)
        self._signals.test_finished.connect(self._on_finished)

        # Decay timer to smoothly drop the meter when quiet
        self._decay_timer = QTimer(self)
        self._decay_timer.timeout.connect(self._decay_level)
        self._decay_timer.start(50)
        self._peak_level = 0.0

    def _populate_devices(self):
        self._dev_combo.clear()
        self._dev_combo.addItem("Default (system)", None)
        if pyaudio is None:
            self._dev_combo.addItem("(PyAudio not installed)", None)
            return
        try:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", f"Device {i}")
                    self._dev_combo.addItem(f"[{i}] {name}", i)
            pa.terminate()
        except Exception:
            pass

        # Pre-select the device if one was provided
        if self._device_index is not None:
            for i in range(self._dev_combo.count()):
                if self._dev_combo.itemData(i) == self._device_index:
                    self._dev_combo.setCurrentIndex(i)
                    break

    def _decay_level(self):
        self._peak_level *= 0.85
        self._meter.set_level(self._peak_level)

    def _on_status(self, text: str):
        self._status_label.setText(text)

    def _on_transcription(self, text: str):
        self._result_label.setText(f'"{text}"')
        self._result_label.show()

    def _on_finished(self, success: bool):
        self._listening = False
        self._test_btn.setText("🎤 Test Again")
        self._test_btn.setObjectName("")
        self._test_btn.setEnabled(True)
        self._dev_combo.setEnabled(True)
        self._test_btn.style().unpolish(self._test_btn)
        self._test_btn.style().polish(self._test_btn)

    def _toggle_test(self):
        if self._listening:
            self._stop_event.set()
            return

        self._result_label.hide()
        self._listening = True
        self._stop_event.clear()
        self._test_btn.setText("⏹ Stop")
        self._test_btn.setObjectName("recording")
        self._test_btn.style().unpolish(self._test_btn)
        self._test_btn.style().polish(self._test_btn)
        self._dev_combo.setEnabled(False)

        device_idx = self._dev_combo.currentData()
        self._thread = threading.Thread(
            target=self._test_worker, args=(device_idx,), daemon=True
        )
        self._thread.start()

    def _test_worker(self, device_index):
        """Background thread: record via speech_recognition → transcribe."""
        print()
        print("=" * 56)
        print("  [MicTest] MICROPHONE TEST STARTED")
        print("=" * 56)
        print(f"  [MicTest] Device index: {device_index or 'Default (system)'}")

        if sr is None:
            print("  [MicTest] FAIL — SpeechRecognition not installed")
            self._signals.status_changed.emit("SpeechRecognition not installed.")
            self._signals.test_finished.emit(False)
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 150  # Low threshold for better sensitivity
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 2.0

        print("  [MicTest] Creating Microphone object...")
        try:
            mic = sr.Microphone(device_index=device_index)
            print("  [MicTest] ✓ Microphone object created")
        except Exception as e:
            print(f"  [MicTest] FAIL — Mic init: {e}")
            self._signals.status_changed.emit(f"Mic init failed: {e}")
            self._signals.test_finished.emit(False)
            return

        # Phase 1: Calibrate ambient noise
        print("  [MicTest] Phase 1: Calibrating ambient noise...")
        self._signals.status_changed.emit("Calibrating... stay quiet for a moment.")
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=4.0)
            print(f"  [MicTest] ✓ Calibration done (energy_threshold={recognizer.energy_threshold:.0f})")
        except Exception as e:
            print(f"  [MicTest] FAIL — Stream error: {e}")
            self._signals.status_changed.emit(f"Stream error: {e}")
            self._signals.test_finished.emit(False)
            return

        # Phase 2: Record using speech_recognition (proven to work)
        print("  [MicTest] Phase 2: Listening for speech (up to 15 seconds)...")
        print("  [MicTest] 🔴 Speak now!")
        self._signals.status_changed.emit("🔴 Speak now! Say a full sentence...")

        # Pulse the level meter to show we're listening
        self._signals.level_changed.emit(0.15)

        # Ensure energy threshold isn't too low after calibration
        recognizer.energy_threshold = max(recognizer.energy_threshold, 50)
        print(f"  [MicTest] Adjusted energy_threshold={recognizer.energy_threshold:.0f}")

        # Pulse animation while listening (runs in a separate thread)
        pulse_stop = threading.Event()
        def _pulse():
            import math
            t = 0
            while not pulse_stop.is_set():
                # Gentle breathing animation
                level = 0.08 + 0.07 * math.sin(t * 3.0)
                self._signals.level_changed.emit(level)
                t += 0.05
                pulse_stop.wait(0.05)
        pulse_thread = threading.Thread(target=_pulse, daemon=True)
        pulse_thread.start()

        try:
            with mic as source:
                audio = recognizer.listen(
                    source, timeout=15, phrase_time_limit=10
                )

            pulse_stop.set()

            # Flash the meter to indicate capture success
            self._signals.level_changed.emit(0.8)

            audio_bytes = audio.get_raw_data()
            sample_rate = audio.sample_rate
            duration = len(audio_bytes) / (sample_rate * audio.sample_width)
            print(f"  [MicTest] ✓ Audio captured: {len(audio_bytes):,} bytes ({duration:.1f}s at {sample_rate}Hz)")

        except sr.WaitTimeoutError:
            pulse_stop.set()
            print("  [MicTest] WARN — No speech detected within timeout")
            self._signals.status_changed.emit(
                "⚠ No speech detected. Is your mic muted?"
            )
            self._signals.test_finished.emit(False)
            return
        except Exception as e:
            pulse_stop.set()
            print(f"  [MicTest] FAIL — Recording error: {e}")
            self._signals.status_changed.emit(f"Recording error: {e}")
            self._signals.test_finished.emit(False)
            return

        # Phase 3: Transcribe
        print("  [MicTest] Phase 3: Sending to Google Speech-to-Text API...")
        self._signals.status_changed.emit("Transcribing with Google Speech-to-Text...")

        try:
            text = recognizer.recognize_google(audio)
            print()
            print("  ┌──────────────────────────────────────────────┐")
            print(f"  │  TRANSCRIBED: \"{text}\"")
            print("  └──────────────────────────────────────────────┘")
            print()
            print("  [MicTest] ✓ ALL TESTS PASSED — Mic + STT working!")
            print("=" * 56)
            self._signals.status_changed.emit("✓ Mic works! Speech recognized:")
            self._signals.transcription_ready.emit(text)
            self._signals.test_finished.emit(True)
        except sr.UnknownValueError:
            print("  [MicTest] WARN — Audio captured but STT couldn't understand")
            self._signals.status_changed.emit(
                "Audio captured but couldn't understand speech. Try speaking louder."
            )
            self._signals.test_finished.emit(False)
        except sr.RequestError as e:
            print(f"  [MicTest] FAIL — Google STT API error: {e}")
            self._signals.status_changed.emit(f"Google STT error: {e}")
            self._signals.test_finished.emit(False)

    def closeEvent(self, event):
        self._stop_event.set()
        event.accept()

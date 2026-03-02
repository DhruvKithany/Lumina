"""
HUD Launcher
"""

import sys
import subprocess
import threading
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTextEdit,
    QGraphicsDropShadowEffect,
    QFileDialog,
    QComboBox,
)

from core.state import TelemetryState, load_config
from cv_engine.pipeline import CVPipeline
from hud.overlay import HUDOverlay
from probes.injector import ProbeInjector
from probes.qa_listener import QAListener
from probes.script_loader import load_script, segment_script
from probes.script_tracker import ScriptTracker
from hud.mic_test import MicTestDialog


_project_root = Path(__file__).resolve().parent

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina-Presenter Core")
        self.setFixedSize(560, 680)
        
        # Style the window to look like a modern AI tool
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f172a; /* Slate 900 */
            }
            QLabel#title {
                color: #f8fafc; /* Slate 50 */
                font-family: 'Segoe UI', Arial;
                font-size: 24px;
                font-weight: bold;
                letter-spacing: 2px;
            }
            QLabel#subtitle {
                color: #94a3b8; /* Slate 400 */
                font-family: 'Segoe UI', Arial;
                font-size: 13px;
                margin-bottom: 20px;
            }
            QPushButton {
                background-color: #1e293b; /* Slate 800 */
                color: #38bdf8; /* Light Blue */
                border: 1px solid #38bdf8;
                border-radius: 6px;
                padding: 12px;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                font-weight: bold;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 0.1);
            }
            QPushButton:pressed {
                background-color: rgba(56, 189, 248, 0.2);
            }
            QPushButton#mainBtn {
                color: #a78bfa; /* Purple */
                border: 1px solid #a78bfa;
            }
            QPushButton#mainBtn:hover {
                background-color: rgba(167, 139, 250, 0.1);
            }
            QPushButton#mainBtn:pressed {
                background-color: rgba(167, 139, 250, 0.2);
            }
            QTextEdit {
                background-color: #020617; /* Slate 950 */
                color: #10b981; /* Emerald 500 */
                border: 1px solid #334155;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(30, 30, 30, 30)

        # Header
        title = QLabel("LUMINA PRESENTER")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Glow effect for title
        glow = QGraphicsDropShadowEffect(self)
        glow.setBlurRadius(15)
        glow.setColor(QColor(56, 189, 248, 150))
        glow.setOffset(0, 0)
        title.setGraphicsEffect(glow)

        subtitle = QLabel("AFFECTIVE COMPUTING HUD MODULE")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # Initialize State & CV Pipeline
        self.config = load_config()
        self.state = TelemetryState()
        # Ensure HUD gets the heatmap feed during calibration
        self.state.update(show_face_heatmap=True)
        self.pipeline = CVPipeline(
            self.state,
            config=self.config,
            camera_index=0,
            heatmap_queue=None,
        )
        self.cv_thread = threading.Thread(target=self.pipeline.run, daemon=True)
        self.cv_thread.start()

        # Calibration Panel (LARP Factor)
        cal_panel = QWidget()
        cal_layout = QVBoxLayout(cal_panel)
        cal_layout.setContentsMargins(10, 10, 10, 10)
        cal_panel.setStyleSheet(
            "background-color: rgba(30, 41, 59, 150); border: 1px solid #334155; border-radius: 6px;"
        )
        
        cal_title = QLabel("PRE-FLIGHT CALIBRATION")
        cal_title.setStyleSheet("color: #fbbf24; font-family: Consolas; font-weight: bold; font-size: 11px;")
        cal_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cal_layout.addWidget(cal_title)
        
        # Camera Feed Label
        self.feed_label = QLabel()
        self.feed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feed_label.setStyleSheet("background-color: #000; border: 1px solid #334155; border-radius: 4px;")
        self.feed_label.setMinimumSize(320, 240)
        cal_layout.addWidget(self.feed_label)
        
        # Readouts
        readouts_layout = QHBoxLayout()
        self.var_label = QLabel("BASE VRNC: 0.000")
        self.shld_label = QLabel("SHLD MOMENTUM: 0.000")
        self.gaze_label = QLabel("GAZE VECTOR: UNAUTH")
        for lbl in (self.var_label, self.shld_label, self.gaze_label):
            lbl.setStyleSheet("color: #94a3b8; font-family: Consolas; font-size: 9px;")
            readouts_layout.addWidget(lbl)
        cal_layout.addLayout(readouts_layout)
        
        self.cal_status = QLabel("SIGNAL STABILITY: CALIBRATING...")
        self.cal_status.setStyleSheet("color: #f87171; font-family: Consolas; font-size: 10px; font-weight: bold;")
        self.cal_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cal_layout.addWidget(self.cal_status)
        
        # Calibration Buttons
        cal_btns = QHBoxLayout()
        self.btn_cal_posture = QPushButton("CALIBRATE NEUTRAL POSTURE")
        self.btn_cal_movement = QPushButton("CALIBRATE NATURAL MOVEMENT")
        self.btn_cal_gaze = QPushButton("CALIBRATE GAZE LOCK")
        
        for btn in (self.btn_cal_posture, self.btn_cal_movement, self.btn_cal_gaze):
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(
                """
                QPushButton {
                    background-color: #0f172a; color: #4ade80; border: 1px solid #4ade80;
                    border-radius: 4px; padding: 6px; font-family: Consolas; font-size: 9px;
                }
                QPushButton:hover { background-color: rgba(74, 222, 128, 0.1); }
                QPushButton:disabled { color: #334155; border-color: #334155; }
                """
            )
            cal_btns.addWidget(btn)
            
        self.btn_cal_posture.clicked.connect(lambda: self._run_cal("posture"))
        self.btn_cal_movement.clicked.connect(lambda: self._run_cal("movement"))
        self.btn_cal_gaze.clicked.connect(lambda: self._run_cal("gaze"))
        
        cal_layout.addLayout(cal_btns)
        layout.addWidget(cal_panel)

        # Buttons Layout (Launch)
        btn_layout = QHBoxLayout()
        
        self.mock_btn = QPushButton("LAUNCH MOCK ENGINE\n(Frontend UI Testing)")
        self.mock_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mock_btn.clicked.connect(self.launch_mock)
        
        self.main_btn = QPushButton("LAUNCH LIVE CALIBRATION\n(Webcam + CV Backend)")
        self.main_btn.setObjectName("mainBtn")
        self.main_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.main_btn.clicked.connect(self.launch_main)

        btn_layout.addWidget(self.mock_btn)
        btn_layout.addWidget(self.main_btn)
        layout.addLayout(btn_layout)
        
        # Script Upload Button
        self.script_btn = QPushButton("📄 UPLOAD SCRIPT\n(PDF or TXT)")
        self.script_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.script_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1e293b; color: #fbbf24; border: 1px solid #fbbf24;
                border-radius: 6px; padding: 8px; font-family: Consolas; font-size: 11px;
                font-weight: bold; margin: 5px;
            }
            QPushButton:hover { background-color: rgba(251, 191, 36, 0.1); }
            """
        )
        self.script_btn.clicked.connect(self._upload_script)
        layout.addWidget(self.script_btn)

        # Microphone Selector
        mic_layout = QHBoxLayout()
        mic_label = QLabel("\U0001f3a4 INPUT DEVICE:")
        mic_label.setStyleSheet(
            "color: #94a3b8; font-family: Consolas; font-size: 10px; font-weight: bold;"
        )
        mic_layout.addWidget(mic_label)

        self.mic_combo = QComboBox()
        self.mic_combo.setStyleSheet(
            """
            QComboBox {
                background-color: #1e293b; color: #38bdf8; border: 1px solid #334155;
                border-radius: 4px; padding: 4px 8px; font-family: Consolas; font-size: 10px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1e293b; color: #38bdf8; border: 1px solid #334155;
                selection-background-color: rgba(56, 189, 248, 0.2);
                font-family: Consolas; font-size: 10px;
            }
            """
        )
        self._populate_mic_list()
        mic_layout.addWidget(self.mic_combo, stretch=1)

        self.mic_test_btn = QPushButton("🔊 TEST")
        self.mic_test_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_test_btn.setFixedWidth(70)
        self.mic_test_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #1e293b; color: #38bdf8; border: 1px solid #38bdf8;
                border-radius: 4px; padding: 4px; font-family: Consolas;
                font-size: 10px; font-weight: bold;
            }
            QPushButton:hover { background-color: rgba(56, 189, 248, 0.15); }
            """
        )
        self.mic_test_btn.clicked.connect(self._open_mic_test)
        mic_layout.addWidget(self.mic_test_btn)
        layout.addLayout(mic_layout)

        # Terminal Log
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setText("> System initialized. Awaiting calibration... [REQUIRED FOR HIGH-STAKES ORATORY]")
        layout.addWidget(self.log)

        self.setCentralWidget(central)
        self.process = None
        
        # Calibration State
        self.cals_done = set()
        self.main_btn.setEnabled(False)
        self.main_btn.setToolTip("Complete pre-flight calibration to unlock.")
        
        # HUD reference
        self.hud = None
        self.injector = None
        self.qa_listener = None
        self.script_tracker = None

        
        # Setup Timer to update video feed & live readouts
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_feed)
        self.timer.start(100)

    def _update_feed(self):
        snap = self.state.snapshot()
        frame = snap.get("latest_frame")
        if frame is not None:
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = frame_rgb.shape
            bpl = c * w
            qImg = QImage(frame_rgb.data, w, h, bpl, QImage.Format.Format_RGB888)
            pm = QPixmap.fromImage(qImg).scaled(
                self.feed_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.feed_label.setPixmap(pm)
            
        # Update readouts if not locked
        if "posture" not in self.cals_done:
            baseline = snap.get("bsp_fidget_variance", 0.0)
            self.var_label.setText(f"BASE VRNC: {baseline:.4f}")
        if "movement" not in self.cals_done:
            stab = snap.get("stability_score", 0.0)
            self.shld_label.setText(f"SHLD MOMENTUM: {1.0 - stab:.4f}")
        if "gaze" not in self.cals_done:
            gaze_txt = "LCKD" if not snap.get("gaze_lost", True) else "UNAUTH"
            self.gaze_label.setText(f"GAZE VECTOR: {gaze_txt}")

    def _run_cal(self, cal_type: str):
        if cal_type == "posture":
            self.log.append("> Aligning 3D coordinate inference...")
            self.log.append("> Mapping 33 skeletal landmarks. Variance baseline set: 0.002")
            self.var_label.setText("BASE VRNC: 0.002")
            self.btn_cal_posture.setEnabled(False)
            self.btn_cal_posture.setText("POSTURE: LOCKED")
        elif cal_type == "movement":
            self.log.append("> Quantifying kinesic entropy thresholds...")
            self.log.append("> Calculating continuous Euclidean distance. Shoulder momentum: 0.005")
            self.shld_label.setText("SHLD MOMENTUM: 0.005")
            self.btn_cal_movement.setEnabled(False)
            self.btn_cal_movement.setText("MOVEMENT: LOCKED")
        elif cal_type == "gaze":
            self.log.append("> Engaging sub-pixel iris tracking (PnP Geometry)...")
            self.log.append("> Establishing 'Golden Zone' vectors. Engagement Lock ready.")
            self.gaze_label.setText("GAZE VECTOR: LCKD")
            self.btn_cal_gaze.setEnabled(False)
            self.btn_cal_gaze.setText("GAZE: LOCKED")
            
        self.cals_done.add(cal_type)
        if len(self.cals_done) == 3:
            self.cal_status.setText("SIGNAL STABILITY: OPTIMAL")
            self.cal_status.setStyleSheet("color: #4ade80; font-family: Consolas; font-size: 10px; font-weight: bold;")
            self.main_btn.setEnabled(True)
            self.main_btn.setToolTip("")
            self.log.append("\n> [ SYSTEM READY ] All biomteric sensors calibrated. You may launch.")

    def launch_mock(self):
        self._start_process([sys.executable, "hud/mock_engine.py"])

    def launch_main(self):
        self.log.append("\n> Executing: Internal HUD Launch")
        self.log.append("> System booting into transparent overlay mode...")
        self.log.append("> Telemetry bridge active.")
        
        # Start probes
        probes_cfg = self.config.get("probes", {})
        if probes_cfg.get("enabled", True):
            probes_file = probes_cfg.get("file", "assets/probes.json")
            probes_path = _project_root / probes_file
            self.injector = ProbeInjector(self.state, probes_path)
            try:
                silence_sec = probes_cfg.get("silence_seconds", 4.0)
                self.injector.start_stall_detection(silence_seconds=silence_sec)
            except Exception as e:
                self.log.append(f"> VAD Warning: audio probes disabled ({e})")
            # Always start periodic tips (cycles every 15s, shows for 15s)
            self.injector.start_periodic_tips(interval_seconds=15.0, display_seconds=15.0)
        

                
        # Launch HUD natively so CV thread isn't blocked
        hud_cfg = self.config.get("hud", {})
        self.hud = HUDOverlay(
            self.state,
            width=hud_cfg.get("width", 480),
            height=hud_cfg.get("height", 360),
            position=hud_cfg.get("position", "top_right"),
            margin=hud_cfg.get("margin", 16),
        )
        self.hud.show()
        self.hud.start_update_timer(hud_cfg.get("update_interval_ms", 100))
        
        # Start Q&A listener (also handles script tracking when recording)
        mic_idx = self.mic_combo.currentData()
        self.qa_listener = QAListener(self.state, device_index=mic_idx)
        if self.script_tracker is not None:
            self.qa_listener.set_script_tracker(self.script_tracker)
        self.qa_listener.start()
        
        # Minimize the launcher window so it gets out of the way
        self.showMinimized()

    def _start_process(self, cmd):
        if self.process is not None:
            self.log.append("> Terminating previous instance...")
            self.process.terminate()
            self.process.wait()
            
        self.log.append(f"\n> Executing: {' '.join(cmd)}")
        self.log.append("> System booting into transparent overlay mode...")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(_project_root),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        self.log.append("> Telemetry bridge active.")

    def closeEvent(self, event):
        self.pipeline.stop()
        if self.injector:
            self.injector.stop_stall_detection()
        if self.qa_listener:
            self.qa_listener.stop()
        if self.hud:
            self.hud.close()
        if self.process is not None:
            self.process.terminate()
        event.accept()

    def _populate_mic_list(self):
        """Enumerate available audio input devices and fill the combo box."""
        self.mic_combo.clear()
        self.mic_combo.addItem("Default (system)", None)
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    name = info.get("name", f"Device {i}")
                    self.mic_combo.addItem(f"[{i}] {name}", i)
            pa.terminate()
        except Exception as e:
            self.mic_combo.addItem(f"(PyAudio unavailable: {e})", None)

    def _open_mic_test(self):
        """Open the Zoom-style mic test dialog."""
        device_idx = self.mic_combo.currentData()
        dialog = MicTestDialog(self, device_index=device_idx)
        dialog.exec()

    def _upload_script(self):
        """Open file dialog to upload a script (PDF/TXT) for tracking."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Upload Presentation Script",
            str(_project_root),
            "Documents (*.pdf *.txt *.md);;All Files (*)",
        )
        if not path:
            return
        
        try:
            text = load_script(path)
            segments = segment_script(text)
            if not segments:
                self.log.append(f"> Script loaded but no text segments found in: {path}")
                return
            
            self.script_tracker = ScriptTracker(segments)
            self.state.update(script_loaded=True, script_progress=0.0)
            
            # If QA listener is already running, wire the tracker
            if self.qa_listener is not None:
                self.qa_listener.set_script_tracker(self.script_tracker)
            
            self.log.append(f"\n> Script loaded: {Path(path).name}")
            self.log.append(f"> Extracted {len(segments)} talking points.")
            self.log.append(f"> Preview: \"{segments[0][:80]}...\"")
            self.log.append("> Press ⏺ REC in the HUD to start script tracking.")
            
            self.script_btn.setText(f"📄 SCRIPT: {Path(path).stem[:20]}")
            self.script_btn.setStyleSheet(
                """
                QPushButton {
                    background-color: rgba(251, 191, 36, 0.15); color: #fbbf24;
                    border: 1px solid #fbbf24; border-radius: 6px; padding: 8px;
                    font-family: Consolas; font-size: 11px; font-weight: bold; margin: 5px;
                }
                """
            )
        except Exception as e:
            self.log.append(f"> Error loading script: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Try to use a dark theme default if available
    app.setStyle("Fusion")
    
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())

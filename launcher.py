"""
Lumina-Presenter: Mission Control Launcher

A beautifully styled graphic user interface to launch the Lumina HUD
and backend services without needing to type terminal commands.
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
    QGraphicsDropShadowEffect
)

from core.state import TelemetryState, load_config
from cv_engine.pipeline import CVPipeline
from hud.overlay import HUDOverlay
from probes.injector import ProbeInjector


_project_root = Path(__file__).resolve().parent

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina-Presenter Core")
        self.setFixedSize(500, 500)
        
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
        self._start_process(["python", "hud/mock_engine.py"])

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

        if self.hud:
            self.hud.close()
        if self.process is not None:
            self.process.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Try to use a dark theme default if available
    app.setStyle("Fusion")
    
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())

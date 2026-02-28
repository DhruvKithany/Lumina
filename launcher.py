"""
Lumina-Presenter: Mission Control Launcher

A beautifully styled graphic user interface to launch the Lumina HUD
and backend services without needing to type terminal commands.
"""

import sys
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QIcon, QColor, QPalette
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

_project_root = Path(__file__).resolve().parent

class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lumina-Presenter Core")
        self.setFixedSize(500, 360)
        
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

        # Buttons Layout
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
        self.log.setText("> System initialized. Awaiting launch command...")
        layout.addWidget(self.log)

        self.setCentralWidget(central)
        self.process = None

    def launch_mock(self):
        self._start_process(["python", "hud/mock_engine.py"])

    def launch_main(self):
        self._start_process(["python", "main.py"])

    def _start_process(self, cmd):
        if self.process is not None:
            self.log.append("> Terminating previous instance...")
            self.process.terminate()
            self.process.wait()
            
        self.log.append(f"\n> Executing: {' '.join(cmd)}")
        self.log.append("> System booting into transparent overlay mode...")
        # Start subprocess without blocking UI
        self.process = subprocess.Popen(
            cmd,
            cwd=str(_project_root),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        self.log.append("> Telemetry bridge active.")

    def closeEvent(self, event):
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

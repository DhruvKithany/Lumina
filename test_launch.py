import sys
import traceback
from PyQt6.QtCore import QTimer

try:
    from launcher import LauncherWindow
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = LauncherWindow()
    window.show()
    
    def simulate_clicks():
        print("Simulating clicks...")
        window._run_cal("posture")
        window._run_cal("movement")
        window._run_cal("gaze")
        print("Clicking launch...")
        window.main_btn.click()
        
    QTimer.singleShot(1000, simulate_clicks)
    sys.exit(app.exec())
    
except Exception as e:
    print("CRASH ERROR:", e)
    traceback.print_exc()

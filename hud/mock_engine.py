"""
Mock engine to test the Lumina-Presenter HUD without a webcam.

Generates fake telemetry data (stability, gaze, probes) and feeds it into
the TelemetryState, driving the HUD for UI/UX testing.
"""

import sys
import time
import math
import random
import threading
from pathlib import Path

# Ensure project root is on path so imports work
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt6.QtWidgets import QApplication
from core.state import TelemetryState
from hud.overlay import HUDOverlay


def mock_producer(state: TelemetryState):
    """Background thread that continuously updates the TelemetryState with fake data."""
    t = 0.0
    dt = 1.0 / 30.0  # 30 FPS update rate

    while True:
        # Simulate stability score: oscillates and has some noise
        # Base stability around 0.8, dipping below 0.4 occasionally
        stability = 0.8 + 0.3 * math.sin(t * 0.5) + random.uniform(-0.05, 0.05)
        stability = max(0.0, min(1.0, stability))
        high_entropy = stability < 0.4
        
        # Simulate gaze
        # Most of the time in zone, sometimes wanders out to trigger "gaze lost"
        yaw = 15.0 * math.sin(t * 1.2) + random.uniform(-2, 2)
        
        # Adjust ideal pitch based on presentation mode
        is_irl = state.presentation_mode == "irl"
        ideal_pitch = -15.0 if is_irl else 0.0 # -15 means looking UP in our coordinate system
        
        pitch = ideal_pitch + 10.0 * math.cos(t * 0.8) + random.uniform(-2, 2)
        
        # Gaze lost if too far from ideal
        gaze_lost = abs(yaw) > 12.0 or abs(pitch - ideal_pitch) > 10.0
        
        time_in_zone = state.time_in_zone_seconds
        if gaze_lost:
            time_in_zone = 0.0
        else:
            time_in_zone += dt
            
        # Simulate knowledge probes appearing randomly
        probe_visible = state.probe_visible
        probe_text = state.probe_text
        
        if random.random() < 0.01 and not probe_visible:
            # 1% chance per frame to trigger a probe if none is visible
            probe_visible = True
            probe_text = random.choice([
                "Hint: Mention the 15% revenue growth in Q3.",
                "Stall detected: Transition to the architecture slide.",
                "Probe: Speak louder, engagement is dropping.",
                "Recall: The unit testing framework is Pytest.",
                "Nuance: Emphasize the 'no external hardware' angle."
            ])
        elif random.random() < 0.02 and probe_visible:
            # 2% chance per frame to dismiss the probe
            probe_visible = False
        
        # Dispatch updates to state
        state.update(
            stability_score=stability,
            high_entropy=high_entropy,
            gaze_lost=gaze_lost,
            time_in_zone_seconds=time_in_zone,
            yaw_degrees=yaw,
            pitch_degrees=pitch,
            probe_visible=probe_visible,
            probe_text=probe_text,
            last_frame_ok=True,
            fps=30.0 + random.uniform(-2, 2)
        )
        
        t += dt
        time.sleep(dt)


def main():
    app = QApplication(sys.argv)
    state = TelemetryState()
    
    # Start mock producer thread
    t = threading.Thread(target=mock_producer, args=(state,), daemon=True)
    t.start()
    
    # Initialize and show HUD
    hud = HUDOverlay(
        state,
        width=320,
        height=280,
        position="top_right",
        margin=24,
    )
    hud.show()
    hud.start_update_timer(33)  # ~30fps UI refresh
    
    print("Starting Lumina-Presenter Mock Engine...")
    print("Press Ctrl+C in the terminal or close the window to exit.")
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

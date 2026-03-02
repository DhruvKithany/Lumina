"""Diagnostic: run pipeline for 5s and print fidget + gaze BSP values."""
import time, threading
from core.state import TelemetryState, load_config
from cv_engine.pipeline import CVPipeline

state = TelemetryState()
config = load_config()
pipe = CVPipeline(state, config=config, camera_index=0)
t = threading.Thread(target=pipe.run, daemon=True)
t.start()

print("Move your wrists around! Monitoring for 8 seconds...\n")
for i in range(16):
    time.sleep(0.5)
    snap = state.snapshot()
    var = snap.get("bsp_fidget_variance", -1)
    fidget = snap.get("bsp_fidgeting", None)
    iris = snap.get("bsp_gaze_status", "?")
    iris_d = snap.get("bsp_gaze_distance", -1)
    stab = snap.get("stability_score", -1)
    entropy = snap.get("high_entropy", None)
    fps = snap.get("fps", 0)
    ok = snap.get("last_frame_ok", False)
    print(f"[{i:2d}] frame_ok={ok} fps={fps:.0f} | FIDGET var={var:.6f} is_fidget={fidget} | IRIS={iris} dist={iris_d:.4f} | stab={stab:.3f} entropy={entropy}")

pipe.stop()
print("\nDone.")

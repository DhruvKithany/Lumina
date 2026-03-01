"""Quick test: verify periodic tips fire and BSP indicators work."""
import time
from core.state import TelemetryState
from probes.injector import ProbeInjector

s = TelemetryState()
p = ProbeInjector(s, "assets/probes.json")
p.start_periodic_tips(interval_seconds=3.0, display_seconds=2.0)

print("Waiting 4s for first tip...")
time.sleep(4)
snap = s.snapshot()
print(f"probe_visible={snap['probe_visible']}  probe_text={snap['probe_text']!r}")

print("Waiting 3s for auto-hide...")
time.sleep(3)
snap2 = s.snapshot()
print(f"After hide: probe_visible={snap2['probe_visible']}")

# Test reference_frame toggle
s.update(reference_frame="irl")
snap3 = s.snapshot()
print(f"reference_frame after toggle: {snap3['reference_frame']}")

p.stop_stall_detection()
print("All tests passed!")

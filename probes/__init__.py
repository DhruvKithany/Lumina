"""
Adaptive knowledge probe injections.

When a "cognitive stall" is detected (e.g. prolonged silence via VAD), the
injector surfaces the next pre-loaded probe text to the HUD so the presenter
can recover narrative flow.
"""

from probes.injector import ProbeInjector
from probes.probe_loader import load_probes
from probes.stall_detector import StallDetector

__all__ = ["ProbeInjector", "load_probes", "StallDetector"]

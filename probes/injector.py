"""
Injector: pushes contextual coaching tips into TelemetryState based on
live telemetry data.

Tips are DATA-DRIVEN — they react to what the CV pipeline is actually
detecting (gaze lost, fidgeting, low stability, etc.)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from core.state import TelemetryState
from probes.probe_loader import load_probes

try:
    from probes.stall_detector import StallDetector
except ImportError:
    StallDetector = None  # type: ignore

if TYPE_CHECKING:
    pass

# ── Data-driven tip rules ──────────────────────────────────────────
# Each rule is (condition_fn, tip_text). First matching rule wins.

def _gaze_lost(s: dict) -> bool:
    return s.get("gaze_lost", False)

def _high_entropy(s: dict) -> bool:
    return s.get("high_entropy", False)

def _fidgeting(s: dict) -> bool:
    return s.get("bsp_fidgeting", False)

def _iris_warning(s: dict) -> bool:
    return s.get("bsp_gaze_status", "LOCKED") in ("WARNING", "GAZE LOST")

def _low_stability(s: dict) -> bool:
    return s.get("stability_score", 1.0) < 0.6

def _medium_stability(s: dict) -> bool:
    return s.get("stability_score", 1.0) < 0.8

def _always(s: dict) -> bool:
    return True


CONTEXTUAL_TIPS = [
    (_gaze_lost,        "[!] Re-establish eye contact with the camera lens now."),
    (_gaze_lost,        "[!] Your gaze has drifted -- look directly at the sensor."),
    (_iris_warning,     "[!] Iris deviation detected -- center your gaze."),
    (_high_entropy,     "[!] High kinesic entropy -- anchor your hands and breathe."),
    (_high_entropy,     "[!] Excessive micro-movements detected. Ground your posture."),
    (_fidgeting,        "[!] Wrist variance elevated -- keep hands still or gesture deliberately."),
    (_fidgeting,        "[!] Fidget detected -- place hands on desk or clasp naturally."),
    (_low_stability,    "[~] Stability dropping -- slow your breathing, square shoulders."),
    (_low_stability,    "[~] Authority presence weakening -- project calm, deliberate energy."),
    (_medium_stability, "[>] Consider a 2-second power pause before your next point."),
    (_medium_stability, "[>] Lower your vocal register -- deeper tone conveys confidence."),
    (_always,           "[>] Pivot narrative to unit economics and market defensibility."),
    (_always,           "[>] Reiterate your competitive moat -- what can't be replicated."),
    (_always,           "[>] Address sub-pixel iris tracking architecture advantages."),
    (_always,           "[>] Emphasize the data flywheel effect in your pipeline."),
    (_always,           "[>] Mention reinforcement learning optimization loops."),
    (_always,           "[>] Highlight non-intrusive sensory loop benefits."),
    (_always,           "[>] Summarize your last point before transitioning."),
    (_always,           "[>] Reference your TAM/SAM/SOM breakdown."),
    (_always,           "[>] Transition to the live demo -- show, don't tell."),
]


class ProbeInjector:
    """
    Pushes data-driven coaching tips to the HUD. Tips are contextual:
    if gaze is lost it tells you to re-establish eye contact, if fidgeting
    it tells you to anchor your hands, etc.
    """

    def __init__(
        self,
        state: TelemetryState,
        probes_path: str | Path,
    ) -> None:
        self.state = state
        self._generic_probes = load_probes(probes_path)
        self._tip_index = 0
        self._stall_detector = None

        # Periodic timer
        self._timer_thread: threading.Thread | None = None
        self._timer_stop = threading.Event()
        self._probe_display_seconds = 6.0
        self._probe_interval_seconds = 10.0

    # ── Pick the best tip based on current telemetry ───────────────

    def _pick_contextual_tip(self) -> str:
        """Select a tip that matches the current telemetry state."""
        snap = self.state.snapshot()

        # Find all matching tips
        matching = [tip for cond, tip in CONTEXTUAL_TIPS if cond(snap)]

        if matching:
            tip = matching[self._tip_index % len(matching)]
            self._tip_index += 1
            return tip

        # Ultimate fallback
        if self._generic_probes:
            tip = self._generic_probes[self._tip_index % len(self._generic_probes)]
            self._tip_index += 1
            return tip

        return "→ Maintain steady eye contact with the camera lens."

    # ── Inject tip ─────────────────────────────────────────────────

    def _inject_contextual_probe(self) -> None:
        """Push a data-driven tip to the HUD."""
        tip = self._pick_contextual_tip()
        self.state.update(probe_text=tip, probe_visible=True)

    def _on_stall(self) -> None:
        """Called on silence or timer tick."""
        self._inject_contextual_probe()

    # ── Periodic timer ─────────────────────────────────────────────

    def _timer_loop(self) -> None:
        """Background loop: fires tips periodically then auto-hides."""
        while not self._timer_stop.is_set():
            self._timer_stop.wait(timeout=self._probe_interval_seconds)
            if self._timer_stop.is_set():
                break

            self._on_stall()

            self._timer_stop.wait(timeout=self._probe_display_seconds)
            if self._timer_stop.is_set():
                break
            self.clear_probe()

    # ── Public API ─────────────────────────────────────────────────

    def start_stall_detection(self, silence_seconds: float = 4.0) -> None:
        if self._stall_detector is not None:
            return
        if StallDetector is None:
            raise ImportError("webrtcvad/pyaudio not available")
        self._stall_detector = StallDetector(
            silence_seconds=silence_seconds,
            on_stall=self._on_stall,
        )
        self._stall_detector.start()

    def start_periodic_tips(
        self,
        interval_seconds: float = 10.0,
        display_seconds: float = 6.0,
    ) -> None:
        if self._timer_thread is not None and self._timer_thread.is_alive():
            return
        self._probe_interval_seconds = interval_seconds
        self._probe_display_seconds = display_seconds
        self._timer_stop.clear()
        self._timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self._timer_thread.start()

    def stop_stall_detection(self) -> None:
        if self._stall_detector is not None:
            self._stall_detector.stop()
            self._stall_detector = None
        self._timer_stop.set()
        if self._timer_thread is not None:
            self._timer_thread.join(timeout=2.0)
            self._timer_thread = None

    def clear_probe(self) -> None:
        self.state.update(probe_visible=False)

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

# ── Data-driven tip rules (priority-tiered) ────────────────────────
#
# Tips are organized into PRIORITY TIERS. The selector picks from the
# highest-priority tier that has matching conditions, so data-driven
# tips (gaze lost, fidgeting) always beat generic filler.
#
# Tiers:  CRITICAL → WARNING → GENTLE → FILLER

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


# CRITICAL — immediate corrective action needed
CRITICAL_TIPS = [
    (_gaze_lost,    "[!] Re-establish eye contact with the camera lens now."),
    (_gaze_lost,    "[!] Your gaze has drifted — look directly at the sensor."),
    (_iris_warning, "[!] Iris deviation detected — center your gaze on the lens."),
    (_iris_warning, "[!] Eye tracking shows off-center focus — realign."),
]

# WARNING — body language flags
WARNING_TIPS = [
    (_high_entropy, "[!] High kinesic entropy — anchor your hands and breathe."),
    (_high_entropy, "[!] Excessive micro-movements detected. Ground your posture."),
    (_fidgeting,    "[!] Wrist variance elevated — keep hands still or gesture deliberately."),
    (_fidgeting,    "[!] Fidget detected — place hands on desk or clasp naturally."),
    (_low_stability,"[~] Stability dropping — slow your breathing, square shoulders."),
    (_low_stability,"[~] Authority presence weakening — project calm, deliberate energy."),
]

# GENTLE — mild suggestions when things are mostly fine
GENTLE_TIPS = [
    (_medium_stability, "[>] Consider a 2-second power pause before your next point."),
    (_medium_stability, "[>] Lower your vocal register — deeper tone conveys confidence."),
    (_medium_stability, "[>] Reduce gesture velocity — smooth motions read as authority."),
]

# FILLER — shown only when ALL biometrics are green (nothing to correct)
FILLER_TIPS = [
    "[>] Summarize your last point before transitioning.",
    "[>] Use a power pause — silence commands attention.",
    "[>] Project to the back of the room with your voice.",
    "[>] Open your next point with a concrete data point.",
    "[>] Vary your pacing — slow down for key arguments.",
    "[>] Make eye contact with different quadrants of the audience.",
    "[>] Transition with a bridging phrase: 'Building on that...'",
    "[>] Check your posture — shoulders back, chin level.",
    "[>] Smile briefly — it resets audience perception of confidence.",
]

# Ordered from highest to lowest priority
_TIERED_TIPS = [CRITICAL_TIPS, WARNING_TIPS, GENTLE_TIPS]


class ProbeInjector:
    """
    Pushes priority-tiered coaching tips to the HUD.

    When the CV pipeline detects issues (gaze lost, fidgeting, low stability),
    the injector shows THOSE specific tips. Only when all biometrics are green
    does it fall back to generic presentation advice.
    """

    def __init__(
        self,
        state: TelemetryState,
        probes_path: str | Path,
    ) -> None:
        self.state = state
        self._generic_probes = load_probes(probes_path)
        self._tip_index = 0
        self._filler_index = 0
        self._stall_detector = None

        # Periodic timer
        self._timer_thread: threading.Thread | None = None
        self._timer_stop = threading.Event()
        self._probe_display_seconds = 6.0
        self._probe_interval_seconds = 10.0

    # ── Pick the best tip based on current telemetry ───────────────

    def _pick_contextual_tip(self) -> str:
        """
        Select a tip from the highest-priority tier that has a matching
        condition.  If no data-driven conditions fire, fall through to
        generic filler tips.
        """
        snap = self.state.snapshot()

        # Walk tiers top-down; first tier with matches wins
        for tier in _TIERED_TIPS:
            matching = [tip for cond, tip in tier if cond(snap)]
            if matching:
                tip = matching[self._tip_index % len(matching)]
                self._tip_index += 1
                return tip

        # All biometrics green — use filler tips
        pool = FILLER_TIPS if FILLER_TIPS else self._generic_probes
        if pool:
            tip = pool[self._filler_index % len(pool)]
            self._filler_index += 1
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

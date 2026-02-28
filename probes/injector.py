"""
Injector: on cognitive stall, push the next probe text into TelemetryState
so the HUD can display it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from core.state import TelemetryState
from probes.probe_loader import load_probes
from probes.stall_detector import StallDetector

if TYPE_CHECKING:
    pass


class ProbeInjector:
    """
    Holds a list of probes and, when the stall detector fires, updates
    state with the next probe text and sets probe_visible True. Call
    clear_probe() from the HUD or a timer to hide after a few seconds.
    """

    def __init__(
        self,
        state: TelemetryState,
        probes_path: str | Path,
    ) -> None:
        self.state = state
        self._probes = load_probes(probes_path)
        self._index = 0
        self._stall_detector: StallDetector | None = None

    def _on_stall(self) -> None:
        """Called by StallDetector when silence exceeds threshold."""
        if not self._probes:
            return
        text = self._probes[self._index % len(self._probes)]
        self._index += 1
        self.state.update(probe_text=text, probe_visible=True)

    def start_stall_detection(self, silence_seconds: float = 4.0) -> None:
        """Start the VAD-based stall detector; on stall, inject next probe."""
        if self._stall_detector is not None:
            return
        self._stall_detector = StallDetector(
            silence_seconds=silence_seconds,
            on_stall=self._on_stall,
        )
        self._stall_detector.start()

    def stop_stall_detection(self) -> None:
        """Stop the stall detector."""
        if self._stall_detector is not None:
            self._stall_detector.stop()
            self._stall_detector = None

    def clear_probe(self) -> None:
        """Hide the current probe on the HUD."""
        self.state.update(probe_visible=False)

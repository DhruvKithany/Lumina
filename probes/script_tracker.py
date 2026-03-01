"""
Script tracker: compares spoken text against a loaded script and
detects when the presenter deviates from the planned talking points.

Uses difflib.SequenceMatcher for fuzzy matching (no external AI needed
for the comparison itself). Tracks a "cursor" through the script
segments and advances as the presenter covers each point.
"""

from __future__ import annotations

import difflib
from typing import List


class ScriptTracker:
    """
    Tracks progress through a script by matching spoken text against
    script segments.

    Parameters
    ----------
    segments : list[str]
        Ordered list of script segments (talking points).
    match_threshold : float
        Minimum fuzzy match ratio (0-1) to consider a segment "covered".
    lookahead : int
        Number of upcoming segments to search when matching spoken text.
    """

    def __init__(
        self,
        segments: List[str],
        match_threshold: float = 0.35,
        lookahead: int = 3,
    ) -> None:
        self.segments = segments
        self.match_threshold = match_threshold
        self.lookahead = lookahead
        self._cursor: int = 0  # Index of the next segment to cover

    @property
    def progress(self) -> float:
        """Fraction of the script completed (0.0 to 1.0)."""
        if not self.segments:
            return 1.0
        return min(1.0, self._cursor / len(self.segments))

    @property
    def is_complete(self) -> bool:
        """True when all segments have been covered."""
        return self._cursor >= len(self.segments)

    @property
    def current_segment(self) -> str | None:
        """The segment the presenter should be covering now."""
        if self._cursor < len(self.segments):
            return self.segments[self._cursor]
        return None

    def get_upcoming_hint(self, max_chars: int = 120) -> str:
        """
        Return a short preview of the next segment for the HUD.
        Truncated to max_chars with ellipsis.
        """
        seg = self.current_segment
        if seg is None:
            return "✓ Script complete — you covered all points."
        hint = seg[:max_chars]
        if len(seg) > max_chars:
            hint = hint.rsplit(" ", 1)[0] + "..."
        return f"Next: {hint}"

    def advance(self, spoken_text: str) -> tuple[bool, str]:
        """
        Compare spoken text against upcoming segments and advance if matched.

        Parameters
        ----------
        spoken_text : str
            The transcribed speech from the microphone.

        Returns
        -------
        on_track : bool
            True if spoken text matches an upcoming segment.
        message : str
            A coaching message: either confirmation or deviation reminder.
        """
        if self.is_complete:
            return True, "✓ Script complete — you covered all points."

        spoken_lower = spoken_text.lower().strip()
        if not spoken_lower:
            return True, ""

        # Search current + lookahead segments for a match
        end = min(self._cursor + self.lookahead, len(self.segments))
        best_ratio = 0.0
        best_idx = self._cursor

        for i in range(self._cursor, end):
            segment_lower = self.segments[i].lower()

            # Try matching against the full segment
            ratio = difflib.SequenceMatcher(
                None, spoken_lower, segment_lower
            ).ratio()

            # Also try partial: does the spoken text appear as a substring?
            # This handles when the presenter says a fragment of a longer point
            if len(spoken_lower) > 10:
                partial = difflib.SequenceMatcher(
                    None,
                    spoken_lower,
                    segment_lower[:len(spoken_lower) + 50],
                ).ratio()
                ratio = max(ratio, partial)

            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i

        if best_ratio >= self.match_threshold:
            # Advance cursor past the matched segment
            self._cursor = best_idx + 1
            pct = int(self.progress * 100)
            if self.is_complete:
                return True, f"✓ Script complete ({pct}%) — all points covered!"
            return True, f"✓ On track ({pct}%) — covered point {best_idx + 1}/{len(self.segments)}"

        # Deviation: spoken text doesn't match upcoming segments
        hint = self.get_upcoming_hint(100)
        return False, f"📋 {hint}"

    def reset(self) -> None:
        """Reset the cursor to the beginning."""
        self._cursor = 0

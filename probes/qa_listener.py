"""
Q&A Listener: continuously listens to microphone using Google's free
Speech-to-Text API.

Modes:
- **Script tracking**: when `recording_active=True` and a script is loaded,
  feeds transcribed speech to the ScriptTracker for deviation detection.
- **Q&A / Interview**: when in Q&A or Interview presentation mode,
  sends the heard question to a free AI model (g4f) to generate a
  suggested response that appears on the HUD.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

try:
    import speech_recognition as sr
except ImportError:
    sr = None

try:
    import g4f
except ImportError:
    g4f = None

if TYPE_CHECKING:
    from core.state import TelemetryState
    from probes.script_tracker import ScriptTracker


class QAListener:
    """
    Background listener that:
    1. Listens to microphone via speech_recognition
    2. In script mode: feeds text to ScriptTracker for deviation detection
    3. In Q&A/Interview mode: generates AI-suggested responses
    4. Pushes results as probes to the HUD
    """

    def __init__(self, state: "TelemetryState", device_index: int | None = None) -> None:
        self.state = state
        self._device_index = device_index
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_question: str = ""
        self._script_tracker: ScriptTracker | None = None

    def set_script_tracker(self, tracker: "ScriptTracker | None") -> None:
        """Attach or detach a ScriptTracker for script deviation mode."""
        self._script_tracker = tracker

    def _listen_loop(self) -> None:
        """Main listener loop running in background thread."""
        if sr is None:
            print("[QAListener] speech_recognition not installed")
            return

        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 1.5

        try:
            mic = sr.Microphone(device_index=self._device_index)
        except Exception as e:
            print(f"[QAListener] Microphone init failed: {e}")
            return

        # ── Self-test: verify the mic can actually open a stream ──
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.3)
            print(f"[QAListener] Mic self-test passed (device={self._device_index})")
        except Exception as e:
            print(
                f"[QAListener] Mic self-test FAILED (device={self._device_index}): {e}\n"
                f"[QAListener] Speech features disabled. Check PyAudio installation or try a different device."
            )
            return

        print("[QAListener] Speech listener started")

        while not self._stop.is_set():
            snap = self.state.snapshot()
            mode = snap.get("presentation_mode", "pitch")
            recording = snap.get("recording_active", False)
            script_loaded = snap.get("script_loaded", False)

            # Decide what mode to operate in
            script_mode = recording and script_loaded and self._script_tracker is not None
            qa_mode = mode in ("q&a", "interview")

            # Only listen when there's a reason to
            if not script_mode and not qa_mode:
                self._stop.wait(timeout=1.0)
                continue

            try:
                with mic as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = recognizer.listen(
                        source, timeout=5, phrase_time_limit=10
                    )
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[QAListener] Audio error: {e}")
                self._stop.wait(timeout=3.0)
                continue

            # Transcribe using Google's free API
            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[QAListener] Google STT error: {e}")
                continue

            if not text:
                continue

            print(f"[QAListener] Heard: {text!r}")

            # Route to the appropriate handler
            if script_mode:
                self._handle_script_tracking(text)
            elif qa_mode:
                self._handle_qa(text, mode)

    def _handle_script_tracking(self, spoken_text: str) -> None:
        """Feed spoken text to the ScriptTracker and update HUD."""
        if self._script_tracker is None:
            return

        on_track, message = self._script_tracker.advance(spoken_text)
        progress = self._script_tracker.progress

        # Update state with progress and show the message
        self.state.update(
            script_progress=progress,
            probe_text=message,
            probe_visible=True,
        )

        # Auto-hide after 4 seconds if on track (don't clutter HUD)
        if on_track and not self._script_tracker.is_complete:
            time.sleep(4.0)
            # Only hide if no newer probe was pushed
            current = self.state.snapshot()
            if current.get("probe_text") == message:
                self.state.update(probe_visible=False)

    def _handle_qa(self, text: str, mode: str) -> None:
        """Generate AI answer for Q&A / interview mode."""
        if text == self._last_question:
            return

        self._last_question = text

        # Show "processing" immediately
        self.state.update(
            probe_text=f"Q: \"{text}\" — Generating response...",
            probe_visible=True,
        )

        answer = self._get_ai_answer(text, mode)
        self.state.update(
            probe_text=answer,
            probe_visible=True,
        )

    def _get_ai_answer(self, question: str, mode: str) -> str:
        """Use g4f to generate a response to the heard question."""
        if g4f is None:
            return f'Q: "{question}" — Try mentioning your core differentiators.'

        context = "pitch presentation Q&A" if mode == "q&a" else "job interview"
        prompt = (
            f"You are an expert {context} coach. Your client was just asked: "
            f'"{question}". '
            f"Give them a concise suggested response direction in 1-2 sentences. "
            f"Be specific and strategic. Return ONLY the suggestion."
        )

        try:
            response = g4f.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            if response:
                return f'Q: "{question}" — {str(response).strip()}'
        except Exception as e:
            print(f"[QAListener] AI error: {e}")

        return f'Q: "{question}" — Pivot to your value proposition and data.'

    def start(self) -> None:
        """Start the background listener thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background listener thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None


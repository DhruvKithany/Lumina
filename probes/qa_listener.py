"""
Q&A Listener: continuously listens to microphone using Google's free
Speech-to-Text API, and when the user is in Q&A or Interview mode,
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


class QAListener:
    """
    Background listener that:
    1. Listens to microphone via speech_recognition
    2. When in Q&A/Interview mode, transcribes speech
    3. Sends transcribed text to g4f for an AI answer
    4. Pushes the answer as a probe to the HUD
    """

    def __init__(self, state: "TelemetryState") -> None:
        self.state = state
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_question: str = ""

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
            mic = sr.Microphone()
        except Exception as e:
            print(f"[QAListener] Microphone init failed: {e}")
            return

        print("[QAListener] Q&A speech listener started")

        while not self._stop.is_set():
            snap = self.state.snapshot()
            mode = snap.get("presentation_mode", "pitch")

            # Only listen actively in Q&A or Interview mode
            if mode not in ("q&a", "interview"):
                self._stop.wait(timeout=1.0)
                continue

            try:
                with mic as source:
                    # Listen for a phrase (max 10s, timeout after 5s silence)
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(
                        source, timeout=5, phrase_time_limit=10
                    )
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[QAListener] Audio error: {e}")
                self._stop.wait(timeout=1.0)
                continue

            # Transcribe using Google's free API
            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[QAListener] Google STT error: {e}")
                continue

            if not text or text == self._last_question:
                continue

            self._last_question = text
            print(f"[QAListener] Heard: {text!r}")

            # Show "processing" immediately
            self.state.update(
                probe_text=f"Q: \"{text}\" — Generating response...",
                probe_visible=True,
            )

            # Generate AI answer in this same thread (already background)
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

        # Fallback
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

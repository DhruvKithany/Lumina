"""
Cognitive stall detection via voice activity (VAD).

When the user is silent for longer than silence_seconds, we consider it
a "cognitive stall" and the injector can surface a knowledge probe.
Uses webrtcvad for lightweight, real-time VAD on microphone input.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Callable

try:
    import webrtcvad
    import pyaudio
except ImportError:
    webrtcvad = None  # type: ignore
    pyaudio = None  # type: ignore

# webrtcvad expects 16-bit mono 8/16/32 kHz
VAD_SAMPLE_RATE = 16000
VAD_FRAME_MS = 30
VAD_FRAME_BYTES = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # 16-bit = 2 bytes


class StallDetector:
    """
    Listens to microphone and calls on_stall() when speech has been absent
    for at least silence_seconds. Runs in a background thread.
    """

    def __init__(
        self,
        silence_seconds: float = 4.0,
        on_stall: Callable[[], None] | None = None,
        vad_aggressiveness: int = 2,
    ) -> None:
        if webrtcvad is None or pyaudio is None:
            raise ImportError("webrtcvad and pyaudio are required for StallDetector. pip install webrtcvad pyaudio")
        self.silence_seconds = silence_seconds
        self.on_stall = on_stall
        self._vad = webrtcvad.Vad(vad_aggressiveness)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_speech_time: float = time.monotonic()
        self._stall_triggered_this_silence = False

    def _audio_loop(self) -> None:
        """Read microphone and track silence duration."""
        pa = pyaudio.PyAudio()
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=VAD_SAMPLE_RATE,
                input=True,
                frames_per_buffer=VAD_FRAME_BYTES,
            )
        except Exception:
            pa.terminate()
            return
        try:
            while not self._stop.is_set():
                try:
                    data = stream.read(VAD_FRAME_BYTES, exception_on_overflow=False)
                except Exception:
                    break
                if len(data) != VAD_FRAME_BYTES:
                    continue
                is_speech = self._vad.is_speech(data, VAD_SAMPLE_RATE)
                now = time.monotonic()
                if is_speech:
                    self._last_speech_time = now
                    self._stall_triggered_this_silence = False
                else:
                    silence_duration = now - self._last_speech_time
                    if (
                        silence_duration >= self.silence_seconds
                        and not self._stall_triggered_this_silence
                        and self.on_stall
                    ):
                        self._stall_triggered_this_silence = True
                        try:
                            self.on_stall()
                        except Exception:
                            pass
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()

    def start(self) -> None:
        """Start the VAD thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._last_speech_time = time.monotonic()
        self._stall_triggered_this_silence = False
        self._thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the VAD thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

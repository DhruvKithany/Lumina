"""
Lumina Microphone Test
======================
Tests that your microphone works and speech-to-text is functional.

Usage:
    python scripts/test_mic.py
    python scripts/test_mic.py --device 1      (use specific device index)

Steps:
    1. Lists all available input devices
    2. Opens the selected mic and records a short clip
    3. Transcribes the audio via Google STT
    4. Prints the transcribed text
"""

import sys
import argparse
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def list_devices():
    """Print all available audio input devices."""
    import pyaudio
    pa = pyaudio.PyAudio()
    print("\n── Available Input Devices ──────────────────────────")
    count = 0
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            name = info.get("name", f"Device {i}")
            rate = int(info.get("defaultSampleRate", 0))
            chans = info.get("maxInputChannels", 0)
            print(f"  [{i}] {name}  (channels={chans}, rate={rate}Hz)")
            count += 1
    if count == 0:
        print("  (!) No input devices found. Check your microphone connection.")
    pa.terminate()
    print()
    return count


def test_mic(device_index=None):
    """Run the microphone self-test."""
    try:
        import pyaudio
    except ImportError:
        print("[FAIL] PyAudio is not installed. Run: pip install pyaudio")
        return False

    try:
        import speech_recognition as sr
    except ImportError:
        print("[FAIL] SpeechRecognition is not installed. Run: pip install SpeechRecognition")
        return False

    # Step 1: List devices
    print("=" * 56)
    print("  LUMINA MICROPHONE TEST")
    print("=" * 56)
    num_devices = list_devices()
    if num_devices == 0:
        return False

    if device_index is not None:
        print(f"[INFO] Using device index: {device_index}")
    else:
        print("[INFO] Using system default microphone")
    print()

    # Step 2: Create recognizer + mic
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300
    recognizer.dynamic_energy_threshold = True

    try:
        mic = sr.Microphone(device_index=device_index)
    except Exception as e:
        print(f"[FAIL] Could not create Microphone object: {e}")
        return False

    # Step 3: Test opening the stream
    print("[TEST] Opening audio stream...")
    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("[PASS] Audio stream opened and closed successfully.")
    except Exception as e:
        print(f"[FAIL] Could not open audio stream: {e}")
        print("       Try a different --device index or reinstall PyAudio.")
        return False

    # Step 4: Record + transcribe
    print()
    print("── Speak now! (max 8 seconds, or pause to stop) ────")
    print()

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=8)
        print("[PASS] Audio captured successfully.")
    except sr.WaitTimeoutError:
        print("[WARN] No speech detected within timeout. Is your mic muted?")
        return False
    except Exception as e:
        print(f"[FAIL] Recording failed: {e}")
        return False

    # Step 5: Transcribe via Google STT
    print("[INFO] Sending audio to Google Speech-to-Text API...")
    try:
        text = recognizer.recognize_google(audio)
        print()
        print("── Transcription Result ────────────────────────────")
        print(f"  \"{text}\"")
        print("────────────────────────────────────────────────────")
        print()
        print("[PASS] Speech-to-text is working! ✓")
        print(f"[INFO] This text is what the script tracker will match against.")
        return True
    except sr.UnknownValueError:
        print("[WARN] Audio was captured but Google STT could not understand it.")
        print("       Speak more clearly or move closer to the mic.")
        return False
    except sr.RequestError as e:
        print(f"[FAIL] Google STT API error: {e}")
        print("       Check your internet connection.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lumina Microphone Test")
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (from the device list)"
    )
    args = parser.parse_args()

    ok = test_mic(device_index=args.device)
    print()
    if ok:
        print("Result: ALL TESTS PASSED ✓")
        print("Your microphone and speech recognition are ready for Lumina.")
    else:
        print("Result: TEST FAILED ✗")
        print("Fix the issues above before using speech features.")
    sys.exit(0 if ok else 1)

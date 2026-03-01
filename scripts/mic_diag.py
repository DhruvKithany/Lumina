"""
Mic diagnostic — records raw audio from each device to find which one works.
Bypasses speech_recognition's threshold detection entirely.

Usage: python scripts/mic_diag.py
"""
import sys, wave, struct, time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent

try:
    import pyaudio
except ImportError:
    print("pip install pyaudio"); sys.exit(1)


def record_device(device_index, name, duration=3.0):
    """Record raw audio from a device, return (rms, peak, audio_data, rate)."""
    pa = pyaudio.PyAudio()
    info = pa.get_device_info_by_index(device_index)
    rate = int(info["defaultSampleRate"])
    channels = 1  # Force mono

    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=4096,
        )
    except Exception as e:
        pa.terminate()
        return None, None, None, None, str(e)

    frames = []
    chunks = int(rate / 4096 * duration)
    for _ in range(chunks):
        try:
            data = stream.read(4096, exception_on_overflow=False)
            frames.append(data)
        except:
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()

    if not frames:
        return 0, 0, b"", rate, None

    audio = b"".join(frames)
    count = len(audio) // 2
    if count == 0:
        return 0, 0, audio, rate, None

    shorts = struct.unpack(f"<{count}h", audio)
    rms = (sum(s * s for s in shorts) / count) ** 0.5
    peak = max(abs(s) for s in shorts)
    return rms, peak, audio, rate, None


def main():
    pa = pyaudio.PyAudio()

    # Collect input devices
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info.get("maxInputChannels", 0) > 0:
            devices.append((i, info["name"], int(info["defaultSampleRate"])))
    pa.terminate()

    print()
    print("=" * 60)
    print("  LUMINA MIC DIAGNOSTIC — Testing all input devices")
    print("=" * 60)
    print(f"  Found {len(devices)} input device(s)")
    print()
    print("  Speak continuously while this runs (~3s per device)...")
    print()

    results = []
    for idx, name, rate in devices:
        print(f"  [{idx:2d}] {name[:50]:<50s} ", end="", flush=True)
        rms, peak, audio, actual_rate, err = record_device(idx, name, duration=3.0)
        if err:
            print(f"ERROR: {err}")
            continue
        if peak is None:
            print("SKIP (no data)")
            continue

        bar_len = min(30, int(peak / 1000))
        bar = "█" * bar_len + "░" * (30 - bar_len)
        status = "✓ GOOD" if peak > 2000 else ("~ WEAK" if peak > 500 else "✗ SILENT")
        print(f"|{bar}| peak={peak:5d} rms={rms:6.0f}  {status}")
        results.append((idx, name, rms, peak, audio, actual_rate))

    print()

    # Find the best device
    if not results:
        print("  No working devices found!")
        return

    best = max(results, key=lambda x: x[3])  # Highest peak
    idx, name, rms, peak, audio, rate = best

    if peak < 100:
        print("  ⚠️  ALL devices are essentially SILENT.")
        print("     Check Windows Sound Settings > Input > make sure a mic is enabled")
        print("     and the input volume is turned up.")
        return

    print(f"  🏆 Best device: [{idx}] {name}")
    print(f"     Peak={peak}  RMS={rms:.0f}  Rate={rate}Hz")
    print()

    # Save WAV from best device
    out = _root / "mic_test_output.wav"
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(audio)
    print(f"  ✓ Saved {out.name} — play this to hear what device [{idx}] captured")
    print()

    # Try transcription on best
    try:
        import speech_recognition as sr
        audio_obj = sr.AudioData(audio, sample_rate=rate, sample_width=2)
        recognizer = sr.Recognizer()
        text = recognizer.recognize_google(audio_obj)
        print(f'  ✓ Google STT heard: "{text}"')
    except Exception as e:
        print(f"  ✗ Transcription: {e}")

    print()
    print(f"  ➡️  Use --device {idx} in the launcher mic dropdown to select this device.")
    print()


if __name__ == "__main__":
    main()

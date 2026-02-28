# Lumina

An **affective computing layer** that turns a laptop webcam into a real-time presenter coaching HUD. It monitors kinesic stability (fidgeting), gaze engagement (eye contact with camera), and can inject knowledge probes when it detects a cognitive stall (e.g. prolonged silence).

## Features (MVP)

- **Kinesic entropy**: Tracks wrists, elbows, and shoulders via MediaPipe Pose. Flags “high entropy” (fidgeting) when micro-movement exceeds a calibrated baseline and shows a stability meter on the HUD.
- **Gaze tracking**: Uses MediaPipe Face Mesh + OpenCV PnP for head pose. If the head leaves the “golden zone” (looking at camera) for more than 1.5 s, a “Gaze lost” warning appears.
- **Temporal HUD**: A transparent, always-on-top PyQt6 overlay shows stability and gaze duration so presenters can self-correct during pitches.
- **Knowledge probes**: When the system detects prolonged silence (VAD), it can surface the next pre-loaded prompt (e.g. “Summarize your main point”) on the HUD.

## Requirements

- Python 3.10+
- Webcam
- Microphone (optional; required only for probe injection on silence)

Install core dependencies:

```bash
pip install -r requirements.txt
```

For **silence-based probe injection** (VAD), also install:

```bash
pip install webrtcvad pyaudio
```

On Windows, `pyaudio` and `webrtcvad` may require [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) or pre-built wheels. If they are not installed, run with `--no-audio` to disable VAD; probes can still be triggered manually or in a future release.

## Quick start

```bash
cd Lumina
pip install -r requirements.txt
python main.py
```

Options:

- `--no-probes` — Disable probe injection.
- `--no-audio` — Disable microphone (no stall-from-silence detection).
- `--camera 0` — Use camera index 0 (default from config).
- `--config path/to/config.yaml` — Custom config file.

## Project layout

```
Lumina/
├── main.py              # Entry point: CV thread + HUD
├── config.yaml          # Runtime config (camera, thresholds, HUD, probes)
├── requirements.txt
├── assets/
│   └── probes.json      # List of probe strings (edit for your pitch)
├── core/                # Shared state and config
│   ├── __init__.py
│   └── state.py         # TelemetryState, load_config()
├── cv_engine/           # Computer vision pipeline
│   ├── __init__.py
│   ├── capture.py       # OpenCV camera capture
│   ├── pose_entropy.py # Kinesic entropy from MediaPipe Pose
│   ├── gaze_tracker.py # Head pose (PnP) and golden-zone logic
│   └── pipeline.py     # Runs Pose + Face Mesh, updates state
├── hud/
│   ├── __init__.py
│   └── overlay.py      # PyQt6 HUD window
└── probes/              # Knowledge probe injection
    ├── __init__.py
    ├── probe_loader.py # Load probes from JSON
    ├── stall_detector.py # VAD-based silence detection
    └── injector.py    # On stall -> push next probe to state
```

## Configuration

Edit `config.yaml` to adjust:

- **camera**: `index`, `width`, `height`, `fps`
- **kinesic**: `calibration_seconds`, `threshold_sigma`, `window_seconds`, `landmark_indices`
- **gaze**: `golden_zone_*` (pitch/yaw in degrees), `gaze_lost_seconds`
- **probes**: `enabled`, `silence_seconds`, `file`
- **hud**: `width`, `height`, `position`, `margin`, `update_interval_ms`

## Contributing

- **State**: All real-time telemetry lives in `core.state.TelemetryState`. The CV pipeline and probe injector update it; the HUD reads via `state.snapshot()`.
- **Adding metrics**: Add fields to `TelemetryState`, update `cv_engine/pipeline.py` or `probes/injector.py`, and extend `hud/overlay.py` to display them.
- **Calibration**: The first few seconds of video are used to calibrate kinesic baseline; stay relatively still during that period or tune `kinesic.calibration_seconds` and `threshold_sigma`.

## License

See repository license file.

<p align="center">
  <h1 align="center">✦ Lumina</h1>
  <p align="center">
    <strong>Real-time presenter coaching, powered by nothing but your laptop webcam.</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+">
    <img src="https://img.shields.io/badge/cv-MediaPipe-4285F4?logo=google&logoColor=white" alt="MediaPipe">
    <img src="https://img.shields.io/badge/ui-PyQt6-41CD52?logo=qt&logoColor=white" alt="PyQt6">
  </p>
</p>

---

## What is Lumina?

Lumina is an **affective computing layer** that turns a standard laptop webcam into a real-time presenter coaching system. It monitors body language, eye contact, and speech patterns — then surfaces live feedback through a transparent Heads-Up Display (HUD) overlaid on your screen.

No wearables. No external sensors. Just your camera and a desire to present better.

### Key capabilities

| Feature | What it does |
|---|---|
| **Fidget Detection** | Tracks 33 skeletal landmarks via MediaPipe Pose and flags excessive micro-movements in wrists, elbows, and shoulders |
| **Gaze & Eye-Contact Tracking** | Uses a 468-point 3D face mesh with PnP geometry to estimate head pose and detect when eye contact drifts from the camera |
| **Transparent HUD Overlay** | Always-on-top PyQt6 panel showing stability meters, gaze-lock indicators, and coaching tips — works over Zoom, PowerPoint, etc. |
| **Knowledge Probe Injection** | Detects cognitive stalls (prolonged silence via VAD) and surfaces coaching prompts to keep you on track |
| **Script Upload & Tracking** | Upload your script (PDF/TXT), and Lumina tracks your progress in real-time using speech recognition |
| **Biometric Heatmap** | Optional OpenCV window visualizing pose and face landmarks in real-time |

---

## Architecture

```
Camera ──► CV Pipeline ──► TelemetryState ◄── HUD Overlay
                               ▲
                               │
                         Probe Injector
                        (silence / stall)
```

| Module | Role |
|---|---|
| `core/` | Shared `TelemetryState` and YAML config loader — the single source of truth |
| `cv_engine/` | Computer vision: webcam capture, MediaPipe pose & face mesh, gaze tracking, heatmap rendering |
| `backend_signalprocessing_unit/` | Signal processing: fidget detection, gaze smoothing, temporal variance filtering |
| `hud/` | PyQt6 always-on-top overlay UI — reads from state, never writes |
| `probes/` | Stall detection (WebRTC VAD), knowledge probe injection, script loading & tracking |
| `scripts/` | Utilities: model downloader, microphone diagnostics |
| `launcher.py` | GUI launcher — styled mission control panel to start the HUD without the terminal |
| `main.py` | CLI entry point with full argument support |

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- A working webcam
- *(Optional)* A microphone — required for silence-based probe injection in knowledge-probe and script-tracking features
- *(Optional)* [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) — may be needed on Windows for `webrtcvad` / `pyaudio`

### Installation

```bash
# Clone the repo
git clone https://github.com/DhruvKithany/Lumina.git
cd Lumina

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```


## Usage

### Option 1 — GUI Launcher (recommended)

```bash
python launcher.py
```

Opens a styled mission control panel where you can:
- Preview your camera feed
- Select a microphone and run a mic test
- Upload a presentation script (PDF / TXT)
- Run calibration routines
- Launch the HUD with one click

### Option 2 — Command Line

```bash
python main.py [OPTIONS]
```

| Flag | Description |
|---|---|
| `--no-probes` | Disable knowledge-probe injection |
| `--no-audio` | Disable microphone / VAD (probes won't trigger on silence) |
| `--camera INDEX` | Camera device index (default: from `config.yaml`) |
| `--config PATH` | Path to a custom `config.yaml` |
| `--show-heatmap` | Open the biometric heatmap window alongside the HUD |

**Examples:**

```bash
# Basic launch
python main.py

# Use camera 1, show the heatmap
python main.py --camera 1 --show-heatmap

# Run without audio or probes
python main.py --no-probes --no-audio
```

---

## Configuration

All runtime parameters live in [`config.yaml`](config.yaml). Key sections:

| Section | Controls |
|---|---|
| `camera` | Resolution, FPS, device index |
| `kinesic` | Fidget detection thresholds, calibration duration, tracked landmarks |
| `gaze` | Golden-zone pitch/yaw limits, gaze-lost timeout |
| `probes` | Enable/disable, silence duration threshold, probe data file |
| `bsp` | Backend signal-processing thresholds for gaze deviation and fidget sensitivity |
| `hud` | Overlay size, position, update interval |
| `heatmap` | Enable/disable biometric heatmap window |

---

## Project Structure

```
Lumina/
├── assets/
│   ├── models/            # MediaPipe model files
│   └── probes.json        # Default coaching prompts
├── backend_signalprocessing_unit/
│   ├── fidget_detector.py # Temporal variance fidget detection
│   ├── gaze_tracker.py    # Iris-based gaze deviation tracking
│   └── smoothing_buffer.py
├── core/
│   └── state.py           # TelemetryState + config loader
├── cv_engine/
│   ├── capture.py         # Webcam capture wrapper
│   ├── gaze_tracker.py    # PnP gaze-vector estimation
│   ├── heatmap.py         # Landmark visualization
│   ├── mediapipe_backend.py
│   ├── pipeline.py        # Main CV processing loop
│   └── pose_entropy.py    # Skeletal kinesic entropy
├── hud/
│   ├── mic_test.py        # Microphone test dialog
│   ├── mock_engine.py     # Mock CV data for UI development
│   └── overlay.py         # PyQt6 transparent HUD
├── probes/
│   ├── injector.py        # Probe injection controller
│   ├── probe_loader.py    # JSON probe file reader
│   ├── qa_listener.py     # AI Q&A via Groq + Llama
│   ├── script_loader.py   # PDF/TXT script parser
│   ├── script_tracker.py  # Real-time script progress
│   └── stall_detector.py  # WebRTC VAD silence detection
├── scripts/
│   ├── download_models.py # Download MediaPipe models
│   ├── mic_diag.py        # Microphone diagnostics
│   └── test_mic.py        # Mic testing utility
├── config.yaml            # Runtime configuration
├── launcher.py            # GUI mission control
├── main.py                # CLI entry point
└── requirements.txt
```

---

## Tech Stack

| Category | Technologies |
|---|---|
| **Computer Vision** | OpenCV, MediaPipe (Pose + Face Mesh) |
| **UI / HUD** | PyQt6 |
| **Signal Processing** | NumPy |
| **Speech & Audio** | SpeechRecognition, PyAudio, WebRTC VAD |
| **Document Parsing** | pdfplumber |
| **Config** | PyYAML, python-dotenv |

---

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for code conventions, architecture guidelines, and how to add new metrics.

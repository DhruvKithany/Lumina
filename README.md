# Lumina-Presenter

**Single-Stream Behavioral Augmentation for High-Stakes Oratory**

---

### The Vision: A Lifestyle Improvement Tool for Communicators
Public speaking and pitching are high-stress activities that define careers, secure funding, and drive leadership. Yet, mastering these skills usually requires expensive coaching suites, subjective feedback, or cumbersome wearable tech. 

**Lumina-Presenter** changes that. It is an affective computing layer that transforms a standard, low-resolution laptop webcam into a sophisticated biometric sensor array. By utilizing advanced computer vision pipelines, the system provides a real-time, non-intrusive Heads-Up Display (HUD) that monitors and corrects a presenter's physiological and narrative performance during live pitches.

It’s not just tech; it’s a **lifestyle improvement**. It normalizes the stress of high-stakes oratory by giving the user a "Second Brain" to self-correct in real-time, helping them build unshakeable confidence, master their physical presence, and deliver better pitches.

---

## Technical Feature Breakdown (Single-Sensor Architecture)

### 1. Monocular Kinesic Entropy Analysis (The "Fidget" Engine)
- **The Tech:** Utilizing MediaPipe Pose to perform 3D Coordinate Inference from a 2D video stream.
- **The Logic:** The system tracks 33 skeletal landmarks. It calculates the Temporal Variance of the wrists, elbows, and shoulders. If the "Micro-Movement Velocity" exceeds a calibrated baseline, the system flags High Kinesic Entropy (fidgeting/anxiety).
- **The Goal:** Provides a real-time HUD alert to stabilize the presenter’s physical "Authority Presence" without needing external accelerometers or wearables.

### 2. Sub-Pixel Iris Tracking & Gaze-Vector Locking
- **The Tech:** Employs a 468-point 3D Face Mesh to execute sub-pixel tracking of the medial and lateral canthus (eye corners) and the iris.
- **The Logic:** Using Perspective-n-Point (PnP) Geometry, the system estimates the presenter's Head Pose and Gaze Vector relative to the camera lens. If the "Engagement Vector" deviates from the "Golden Zone" (the audience) for > 1.5 seconds, a "Gaze Lost" warning is triggered.
- **The Goal:** Ensures 100% perceived eye contact with judges, simulating a "Confidence Lock" that is statistically proven to increase pitch success.

### 3. Temporal HUD Overlay (The "Smart-Pilot" Interface)
- **The Tech:** A transparent PyQt6 Graphics Layer that sits on top of all active windows (PowerPoint, Zoom, etc.).
- **The Logic:** The HUD processes raw metadata from the CV engine and converts it into readable Performance Telemetry. It includes a "Stability Meter" and a "Gaze Duration Tracker."
- **The Goal:** To provide the presenter with a "Second Brain," allowing them to self-correct in real-time while maintaining focus on the audience.

### 4. Adaptive "Knowledge Probe" Injections
- **The Tech:** Contextual data retrieval mapped to the presenter’s line-of-sight.
- **The Logic:** When the system detects a "Cognitive Stall" (prolonged silence or repetitive facial tics), it can surface pre-loaded technical nuances or "Precision Keywords" onto the HUD.
- **The Goal:** To eliminate "narrative gaps," ensuring the presenter always has the next technical data point ready, even under the stress of judge Q&A.

---

## Why This Wins (The "Efficiency" Edge)
Most hackathon projects require external hardware, IoT sensors, or complex setups. Lumina-Presenter achieves Human-in-the-Loop Optimization using nothing but the camera already built into the user's laptop. It proves that with the right mathematical smoothing and landmark inference, a $10 webcam can perform like a $10,000 professional coaching suite.

---

## The Lumina-Presenter Technical Squad
This is the official squad breakdown, demonstrating our professional-grade AI development pipeline.

### HUD Architects (Frontend) — Dhruv & Chaaki
- **The Mission:** Build the Affective User Interface. Responsible for the "Always-on-Top" transparent HUD that the presenter actually sees.
- **The Code:** Using PyQt6 and CSS to design glowing "Stability Meters," "Gaze Lock" bars, and the "Knowledge Probe" pop-ups that fade in and out.
- **The "Larp" Factor:** Making the project look like a $50M Silicon Valley product. Because if the UI looks like a fighter jet cockpit, the judges will believe the math is genius.

### Computer Vision Engineer — Nathan
- **The Mission:** Manage the Sensory Input Layer. He is the "Eyes" of the system, extracting data from the single monocular laptop camera.
- **The Code:** Implementing MediaPipe Holistic and Face Mesh. He’s responsible for the raw (x,y,z) coordinate mapping of the irises, shoulders, and wrists.
- **The "Larp" Factor:** He provides the "Biometric Heatmap"—the dots on the face and the skeletal lines that prove the AI is "watching" the human in real-time.

### Signal Processing Engineer — Ram
- **The Mission:** Build the Inference Logic & Filtering. He is the "Brain" that turns Nathan's raw dots into actual "Events."
- **The Code:** Using NumPy and SciPy to calculate Temporal Variance and Standard Deviation. He filters out the "noise" (like blinking or natural breathing) so the HUD doesn't flicker.
- **The "Larp" Factor:** He’s the one who can explain the "Math" to the technical judges—talking about "Kalman Filters" and "Euclidean Distance thresholds" to make the project sound scientifically grounded.

---

## Quick start

To run the mission control launcher and start the HUD:
```bash
pip install -r requirements.txt
python launcher.py
```

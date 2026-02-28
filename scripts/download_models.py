"""
Download MediaPipe task models for Lumina-Presenter (pose + face landmarkers).

Run from project root: python scripts/download_models.py

Saves to assets/models/ so the pipeline can use them when using MediaPipe 0.10+ (tasks API).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from urllib.request import urlretrieve
except ImportError:
    urlretrieve = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "assets" / "models"

# Official MediaPipe model URLs (Google Cloud Storage)
POSE_LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
FACE_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def main() -> int:
    if urlretrieve is None:
        print("urllib.request.urlretrieve not available", file=sys.stderr)
        return 1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pose_path = MODELS_DIR / "pose_landmarker_lite.task"
    face_path = MODELS_DIR / "face_landmarker.task"
    if pose_path.exists() and face_path.exists():
        print("Models already present at", MODELS_DIR)
        return 0
    print("Downloading MediaPipe models to", MODELS_DIR)
    try:
        if not pose_path.exists():
            print("  pose_landmarker_lite.task ...")
            urlretrieve(POSE_LITE_URL, pose_path)
            print("  done.")
        if not face_path.exists():
            print("  face_landmarker.task ...")
            urlretrieve(FACE_URL, face_path)
            print("  done.")
    except Exception as e:
        print("Download failed:", e, file=sys.stderr)
        return 1
    print("Done. Run: python main.py [--show-heatmap]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

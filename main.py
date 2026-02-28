"""
Lumina-Presenter: Affective computing layer for presenter coaching.

Entry point: runs the CV pipeline in a background thread and the PyQt6 HUD
on the main thread. Optionally enables knowledge-probe injection on cognitive stall.

Usage:
    python main.py [--no-probes] [--no-audio] [--camera 0] [--config path/to/config.yaml]
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

# Ensure project root is on path
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from PyQt6.QtWidgets import QApplication

from core.state import TelemetryState, load_config
from cv_engine.pipeline import CVPipeline
from hud.overlay import HUDOverlay
from probes.injector import ProbeInjector


def _heatmap_display_loop(
    heatmap_queue: queue.Queue,
    window_title: str,
    stop_event: threading.Event,
) -> None:
    import cv2
    while not stop_event.is_set():
        try:
            frame = heatmap_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if frame is None:
            break
        cv2.imshow(window_title, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    try:
        cv2.destroyWindow(window_title)
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lumina-Presenter: real-time presenter coaching HUD from webcam."
    )
    parser.add_argument(
        "--no-probes",
        action="store_true",
        help="Disable knowledge probe injection on cognitive stall.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Disable microphone/VAD (probes will not trigger on silence).",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="INDEX",
        help="Camera device index (default: from config).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--show-heatmap",
        action="store_true",
        help="Show Biometric Heatmap window (pose + face landmarks).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    state = TelemetryState()
    heatmap_cfg = config.get("heatmap", {})
    heatmap_enabled = args.show_heatmap or heatmap_cfg.get("enabled", False)
    heatmap_queue = None
    heatmap_stop = threading.Event()
    if heatmap_enabled:
        heatmap_queue = queue.Queue(maxsize=1)
    pipeline = CVPipeline(
        state,
        config=config,
        camera_index=args.camera,
        heatmap_queue=heatmap_queue,
    )
    cv_thread = threading.Thread(target=pipeline.run, daemon=True)
    cv_thread.start()
    if heatmap_enabled:
        window_title = heatmap_cfg.get("window_title", "Lumina Biometric Heatmap")
        threading.Thread(
            target=_heatmap_display_loop,
            args=(heatmap_queue, window_title, heatmap_stop),
            daemon=True,
        ).start()

    # Optional probe injector (requires webrtcvad + pyaudio for --no-audio path)
    injector: ProbeInjector | None = None
    if not args.no_probes:
        probes_cfg = config.get("probes", {})
        probes_file = probes_cfg.get("file", "assets/probes.json")
        probes_path = _project_root / probes_file
        injector = ProbeInjector(state, probes_path)
        if not args.no_audio:
            try:
                silence_sec = probes_cfg.get("silence_seconds", 4.0)
                injector.start_stall_detection(silence_seconds=silence_sec)
            except ImportError:
                print(
                    "Warning: audio/VAD not available; probes disabled. Install webrtcvad and pyaudio for silence-based probes.",
                    file=sys.stderr,
                )

    # HUD
    app = QApplication(sys.argv)
    hud_cfg = config.get("hud", {})
    hud = HUDOverlay(
        state,
        width=hud_cfg.get("width", 280),
        height=hud_cfg.get("height", 200),
        position=hud_cfg.get("position", "top_right"),
        margin=hud_cfg.get("margin", 16),
    )
    hud.show()
    hud.start_update_timer(hud_cfg.get("update_interval_ms", 100))

    try:
        return app.exec()
    finally:
        pipeline.stop()
        if heatmap_enabled:
            heatmap_stop.set()
        if injector is not None:
            injector.stop_stall_detection()


if __name__ == "__main__":
    sys.exit(main())

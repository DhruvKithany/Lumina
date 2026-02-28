# Contributing to Lumina-Presenter

Thanks for contributing. This document outlines project structure, conventions, and how to extend the codebase.

## Code layout

- **`core/`** — Shared state and config. Add new telemetry fields in `TelemetryState` and document them; keep `load_config()` as the single place that reads YAML.
- **`cv_engine/`** — Computer vision only. Capture, pose, gaze, and the pipeline that ties them together. No UI or business logic beyond computing metrics.
- **`hud/`** — All overlay UI. Read from `state.snapshot()`; do not write to state from the HUD.
- **`probes/`** — Stall detection and probe injection. Writes to state; no direct dependency on the HUD.

Data flow: **Camera → CV pipeline → TelemetryState ← HUD**. Probes also write to `TelemetryState` when a stall is detected.

## Conventions

- **Type hints**: Use them on public functions and class methods (e.g. `def update(self, landmarks: Sequence[object], timestamp: float) -> tuple[float, bool]:`).
- **Docstrings**: Module docstrings describe the package role; classes and public functions have a one-line summary and, if needed, a short "Returns" / "Raises" section.
- **Config**: Prefer `config.yaml` and nested keys (e.g. `kinesic.threshold_sigma`) over hardcoded constants. Each module that needs config should read from the dict passed in or from `load_config()`.
- **Threading**: The CV pipeline and (optionally) the stall detector run in background threads. All updates to `TelemetryState` go through `state.update(...)`, which is thread-safe. The HUD only reads via `state.snapshot()` on its timer.

## Adding a new metric

1. Add the field(s) to `TelemetryState` in `core/state.py` and to the `snapshot()` dict.
2. In `cv_engine/pipeline.py` (or the appropriate tracker), compute the value and call `state.update(new_field=value)`.
3. In `hud/overlay.py`, read the value from the snapshot and add a widget or label for it.
4. Optionally add config keys in `config.yaml` and read them in the module that computes the metric.

## Running and testing

- From the project root: `python main.py` (use `--no-probes` or `--no-audio` if VAD is not installed).
- For headless or CI, the app currently requires a display for PyQt6; we may add a "no-HUD" or mock state mode later.

## Pull requests

- Keep PRs focused (one feature or fix).
- Ensure new code follows the conventions above and that existing tests (when added) still pass.

# PointRemote — Design

Date: 2026-11-09
Status: Approved design, pending spec review

## Goal

Point an Android phone at a smart-home appliance, have it recognise which one it is, and toggle it through Home Assistant with one tap. Recognition uses an image classifier trained on the user's own photos on their desktop GPU.

Inspired by a Wii-remote + ESP32 project that classified whole scenes at 96×96 on-device (~88% validation accuracy, 1.4 s/inference). This design keeps the "trained classifier" idea but moves inference to a phone and training to a desktop GPU.

## Scope

In v1:
- Native Android app (Kotlin): Point, Capture, Train and Settings screens; home-screen widget; Quick Settings tile.
- PC training server (Python, FastAPI) on the user's desktop (RTX 5070 Ti): stores the dataset, trains, serves models.
- Tap-to-toggle via Home Assistant REST API.

Out of v1 (YAGNI): room/presence filtering, compass tiebreak, per-type controls (sliders, shutter stop), auto-act without confirmation, MQTT, iOS, cloud anything.

Assumption: rooms look visually distinct, so whole-scene classification is sufficient without extra signals.

## Architecture

```
┌────────── Android app ──────────┐          ┌──── PC training server ────┐
│ Widget / QS tile → Point screen │  photos  │ FastAPI                    │
│ Capture screen ─ upload queue ──┼────────► │ dataset/<label>/*.jpg      │
│ Train screen ── POST /train ────┼────────► │ trainer (PyTorch, GPU)     │
│ Model store ◄── model + meta ───┼◄──────── │ models/v<N>/model.onnx     │
│ HA client ── REST toggle ───┐   │          │ web UI: dataset + history  │
└─────────────────────────────┼───┘          └────────────────────────────┘
                              ▼
                       Home Assistant
```

The phone works fully offline from the PC once it has a model: recognition and toggling only need Home Assistant. The PC is needed only for uploads and training.

### Repository layout

```
PointRemote/
  android/        Kotlin app (Gradle)
  server/         Python FastAPI server + trainer
  docs/
```

## Android app

Stack: Kotlin, Jetpack Compose, CameraX (ImageAnalysis), ONNX Runtime Android, Glance (widget), TileService (QS tile), WorkManager (upload queue), EncryptedSharedPreferences, OkHttp. Min SDK 29.

### Screens

**Point** (launch target of widget and tile)
- Full-screen camera preview. Frames go to the classifier via CameraX ImageAnalysis (keep-latest backpressure), centre-cropped and resized to 224×224.
- Top pill: `<Friendly name> · <confidence>%` when locked; hidden otherwise.
- Bottom: large Toggle button, disabled unless locked.
- Haptic tick on lock acquisition; haptic confirm on toggle success; error haptic on failure.
- If no model is installed: message "Capture and train first" with a button to Capture.

**Capture**
- Label dropdown (from server `labels.json`), always including `unknown`.
- "+ New label" → list of HA entities from `GET /api/states`, filtered to domains `light`, `switch`, `cover`, `fan`, `media_player`, `input_boolean`. Choosing one creates the label on the server (label id derived from entity id, e.g. `light.kitchen` → `light_kitchen`; friendly name from HA).
- Hold shutter → burst at ~3 fps. Photos saved as JPEG (longest side 512 px) to app storage and enqueued for upload.
- Per-label counts shown (server count + pending local count).

**Train**
- Server status (online/offline), photo count per label.
- Train button (disabled when offline, while training, or when any label has fewer than 30 photos — warn but allow below 100).
- During training: progress (epoch / total) by polling status.
- After: validation accuracy, per-class accuracy, top confusions. New model downloads automatically.

**Settings**
- HA base URL, HA long-lived token, PC server URL, server access key, confidence threshold (default 0.80), stable-frames count (default 5).
- "Test connection" buttons for HA and server.

### Launchers
- Glance widget: single tap opens Point screen.
- Quick Settings tile: opens Point screen (`startActivityAndCollapse`).

### Recognition decision (pure, unit-tested)

`LockState` from a stream of `(probabilities[], classes[])`:
- Candidate = argmax. Locked iff candidate ≠ `unknown`, p ≥ threshold, and the same candidate held for ≥ N consecutive frames.
- Lock is lost after N consecutive frames that fail the condition (prevents flicker).

### Home Assistant client
- `POST {ha}/api/services/homeassistant/toggle` with `{"entity_id": ...}`, bearer token.
- The returned changed states are used to show the new state ("Kitchen lights → on"). If the response has no state for the entity, `GET /api/states/{entity_id}`.
- Errors: network failure, 401 (bad token), 4xx/5xx → visible message + error haptic. No retries on toggle (toggling twice is worse than failing once).

### Model store
- On app start and on Train completion: `GET /model/latest/meta`; if the version differs from the installed one, download `model.onnx` + `meta.json` to a temp dir, verify size/sha256 from meta, then atomically swap. The previous model is kept until the swap succeeds.
- Failure to reach the server is silent in Point mode (uses installed model).

### Upload queue
- WorkManager job per photo, constraint: network connected; exponential backoff. Survives restarts. Deleted locally after server 2xx.

## PC training server

Stack: Python 3.12, FastAPI + uvicorn, PyTorch + torchvision (MobileNetV3-Large, ImageNet weights), ONNX export, onnxruntime for post-export verification, pytest. Managed with `uv`.

### Storage

```
server/data/
  labels.json                 [{id, entity_id, friendly_name}]
  dataset/<label>/<uuid>.jpg
  models/v<N>/model.onnx
  models/v<N>/meta.json
  models/latest -> v<N>
```

`meta.json`: `{version, created_at, classes: [{id, entity_id, friendly_name}] (in output order), input_size: 224, normalization: {mean, std}, val_accuracy, per_class_accuracy, confusions, sha256, size_bytes}`.

### API

All endpoints require header `X-Api-Key` matching the configured key (LAN-only, shared secret).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness + whether training is running |
| GET | `/labels` | labels with photo counts |
| POST | `/labels` | create label `{entity_id, friendly_name}` |
| DELETE | `/labels/{id}` | remove label and its photos |
| POST | `/photos/{label}` | upload one JPEG (multipart); 404 if label unknown |
| POST | `/train` | start training; 409 if already running |
| GET | `/train/status` | `{state: idle/running/done/failed, epoch, epochs, message}` |
| GET | `/model/latest/meta` | latest `meta.json` (404 if none) |
| GET | `/model/latest/file` | latest `model.onnx` |

Web UI (served at `/ui`, same key via a login form stored in a cookie): browse photos per label, delete individual photos, view training history with accuracy and confusion matrices.

### Training

- Classes = all labels + `unknown` (the `unknown` label always exists).
- Stratified 80/20 train/val split with a fixed seed recorded in meta.
- Augmentation (train only): random resized crop (scale 0.6–1.0), horizontal flip off (scenes are not mirror-symmetric), colour jitter (brightness/contrast, light), small rotation (±10°).
- Fine-tune MobileNetV3-Large: replace the classifier head; train the head for a few epochs with the backbone frozen, then unfreeze and train everything at a lower learning rate. Keep the best epoch by val accuracy. Total ~15–25 epochs; expected a few minutes on the GPU.
- Export to ONNX (fp32; quantization only if on-device latency proves inadequate), then run the val set through onnxruntime and confirm accuracy matches PyTorch within 1 percentage point; fail the run otherwise.
- Runs in a background worker; only one at a time.

## Error handling summary

| Situation | Behaviour |
|---|---|
| HA unreachable / bad token | Message + error haptic; Toggle stays available |
| PC server offline | Point works; Capture queues; Train disabled ("PC offline") |
| No model installed | Point shows "Capture and train first" |
| Model download corrupt | sha256 mismatch → discard, keep old model |
| Training fails / ONNX check fails | Status `failed` with message; `latest` unchanged |
| Upload for deleted label | Server 404 → app drops that photo from the queue |

## Testing

Server (pytest):
- Label CRUD, photo upload storage, auth rejection.
- Deterministic stratified split.
- Smoke training run on a tiny synthetic dataset (CPU allowed): produces model.onnx + meta.json, ONNX check passes.

Android (JUnit):
- Recognition decision logic: thresholds, stable-frame lock and unlock, `unknown` handling.
- HA client against MockWebServer: success, 401, network error, state parsing.
- Model store: version compare, sha256 verification, atomic swap.

Acceptance (manual):
- ≥ 95% validation accuracy with ~100–150 photos per class.
- Real-world: 20 point-and-toggle attempts per device; record correct-lock rate and wrong-lock count. Target: no wrong-device toggles, ≥ 90% correct locks.
- Time from widget tap to first lock < 1.5 s.

## Open items

None blocking. Possible later additions: per-type controls, room gating via HA presence, quantised model, wireless model push notifications.

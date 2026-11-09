# Desktop server app — design

Approved in chat on 2026-11-09.

## Goal

People who don't use a terminal can run the training server: download one file for their OS, double-click it, and a window starts the server; closing the window stops it. The window helps them pair the phone (QR code) and opens a step-by-step guide.

## Pieces

### Launcher (`server/pointremote/desktop/`)
Tk window, no PyTorch imports, frozen with PyInstaller per OS (Windows `.exe`, macOS `.app` for Apple Silicon, Linux binary). It bundles the server source, `uv.lock` and the `uv` binary.

- **First launch** copies the server source to the user data folder and runs `uv sync --frozen --no-dev`, which also installs Python. It shows the current step. It also writes `settings.json` (data folder, generated API key, port 8765) to the user config folder.
- **Every launch** starts the server as a child process: `<venv python> -m pointremote.serve`. The server exits when its stdin closes, so it never outlives the window, even after a crash.
- **The window** shows the status, the address (LAN IP), a QR code (`pointremote://setup?url=…&key=…`), and when the phone last connected. It has buttons for the guide, the photos page (logged in with a one-time ticket, never the key in a URL) and showing the key.
- **On Linux**, the frozen app adds a `.desktop` menu entry on first run.
- **Dev mode:** `uv run pointremote-app` in the repo uses the current venv and reads `server/.env` and `server/data`.
- **`--self-test`:** no window. It does setup, starts the server, waits for `/health`, stops it and exits with 0. CI uses this.

### Server changes
- `pointremote/serve.py` runs uvicorn, with a stdin watchdog when `POINTREMOTE_WATCH_STDIN=1`.
- `/health` adds `phone_last_seen` (seconds since the last request from a non-loopback address).
- `POST /ui/ticket` (needs the API key) issues a single-use ticket valid for 60 s. `GET /ui/ticket/{t}` sets the login cookie.
- Training device: CUDA if present, otherwise Apple MPS, otherwise CPU.
- ONNX export uses the dynamo exporter (adds `onnxscript`). The existing ONNX-vs-PyTorch accuracy gate still applies.
- PyTorch comes from the CUDA 12.8 index on Windows/Linux and from PyPI on macOS.

### Guide
A static, offline HTML page bundled with the launcher. It covers: install the APK, allow the firewall, scan the QR, make an HA token, capture devices and Unknown, train, add the widget, troubleshooting.

### Android
- Settings gets **Scan QR from PC**, which fills in the server URL and key. It uses zxing core with the existing CameraX preview.
- The release build keeps only the ARM ABIs (`armeabi-v7a`, `arm64-v8a`). It is signed with a release key kept outside the repo (`android/keystore.properties`, git-ignored).

### CI (`.github/workflows/desktop.yml`)
Runs on Windows, macOS and Linux:
- the server tests
- a PyInstaller build
- `--self-test` on the built app

The built archives upload as artifacts and, on a `v*` tag, attach to the release. The README says that Windows and macOS are built and start-up tested by CI only, not tried by a person.

## Out of scope
Code signing and notarisation for Windows/macOS, auto-update, Intel Macs.

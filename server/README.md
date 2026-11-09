# PointRemote training server

Stores labelled photos from the phone, trains the recogniser on the GPU, and serves the model.

## Setup

```bash
cd server
uv sync
echo "POINTREMOTE_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" > .env
./run.sh
```

- API: `http://<pc-ip>:8765` with header `X-Api-Key: <key from .env>`
- Web UI: `http://<pc-ip>:8765/ui` (log in with the same key)
- Data: `server/data/` (labels.json, dataset/, models/)

## Tests

```bash
uv run pytest            # GPU test runs only when CUDA is available
```

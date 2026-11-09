from __future__ import annotations

import json
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PORT = 8765


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    api_key: str
    port: int = DEFAULT_PORT


def _read_env_file(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip("'\"")
    return values


def load_or_create(config_dir: Path, default_data_dir: Path, dev_server_dir: Path | None = None) -> AppConfig:
    """Reads settings.json, creating it (with a fresh API key) on first run.

    In dev mode (running from the repo) a server/.env with a key wins, so the repo's
    existing photos, models and key keep working."""
    if dev_server_dir is not None and (dev_server_dir / ".env").is_file():
        env = _read_env_file(dev_server_dir / ".env")
        if env.get("POINTREMOTE_API_KEY"):
            data = env.get("POINTREMOTE_DATA_DIR") or str(dev_server_dir / "data")
            return AppConfig(
                data_dir=(dev_server_dir / data).resolve(),
                api_key=env["POINTREMOTE_API_KEY"],
                port=int(env.get("POINTREMOTE_PORT", DEFAULT_PORT)),
            )

    path = config_dir / "settings.json"
    if path.is_file():
        raw = json.loads(path.read_text())
        return AppConfig(Path(raw["data_dir"]), raw["api_key"], int(raw.get("port", DEFAULT_PORT)))

    cfg = AppConfig(default_data_dir / "server-data", secrets.token_urlsafe(24))
    config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = asdict(cfg) | {"data_dir": str(cfg.data_dir)}
    # Created owner-only from the start (on Windows the user profile folder is already private).
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(raw, indent=2))
    return cfg

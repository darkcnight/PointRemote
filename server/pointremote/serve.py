"""Runs the server for the desktop app: settings from the environment, and exits when the
app that started it goes away (its stdin closes), so the server never outlives the window."""
from __future__ import annotations

import os
import sys
import threading


def _exit_when_stdin_closes() -> None:
    try:
        while sys.stdin.buffer.read(1024):
            pass
    finally:
        os._exit(0)


def main() -> None:
    print(f"PointRemote server starting (Python {sys.version.split()[0]})", flush=True)
    import uvicorn

    if os.environ.get("POINTREMOTE_WATCH_STDIN") == "1":
        threading.Thread(target=_exit_when_stdin_closes, daemon=True).start()
    port = int(os.environ.get("POINTREMOTE_PORT", "8765"))
    uvicorn.run("pointremote.main:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()

"""Runs the server for the desktop app: settings from the environment, and exits when the
app that started it goes away (its stdin closes, or on Windows its process ends), so the
server never outlives the window."""
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


def _exit_when_process_exits(pid: int) -> None:
    """Windows: a thread blocked reading stdin stalls the server's startup there, so wait on
    the desktop app's process handle instead."""
    import ctypes

    SYNCHRONIZE, INFINITE = 0x00100000, 0xFFFFFFFF
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    try:
        if handle:
            kernel32.WaitForSingleObject(handle, INFINITE)
    finally:
        os._exit(0)


def main() -> None:
    print(f"PointRemote server starting (Python {sys.version.split()[0]})", flush=True)
    import uvicorn

    parent = os.environ.get("POINTREMOTE_PARENT_PID")
    if os.name == "nt" and parent:
        threading.Thread(target=_exit_when_process_exits, args=(int(parent),), daemon=True).start()
    elif os.environ.get("POINTREMOTE_WATCH_STDIN") == "1":
        threading.Thread(target=_exit_when_stdin_closes, daemon=True).start()
    port = int(os.environ.get("POINTREMOTE_PORT", "8765"))
    uvicorn.run("pointremote.main:app", host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()

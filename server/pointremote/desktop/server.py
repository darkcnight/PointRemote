"""Starts and stops the training server as a child process."""
from __future__ import annotations

import json
import os
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import AppConfig
from .runtime import NO_WINDOW


class ServerProcess:
    def __init__(self, python: Path, workdir: Path, cfg: AppConfig, log_path: Path):
        self.python, self.workdir, self.cfg, self.log_path = python, workdir, cfg, log_path
        self.proc: subprocess.Popen | None = None
        self.instance = secrets.token_urlsafe(16)
        self.verified = False

    def start(self) -> None:
        self.cfg.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ | {
            "POINTREMOTE_API_KEY": self.cfg.api_key,
            "POINTREMOTE_DATA_DIR": str(self.cfg.data_dir),
            "POINTREMOTE_PORT": str(self.cfg.port),
            "POINTREMOTE_WATCH_STDIN": "1",
            "POINTREMOTE_INSTANCE": self.instance,
            "PYTHONUNBUFFERED": "1",
        }
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(
            [str(self.python), "-m", "pointremote.serve"], cwd=self.workdir, env=env,
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW,
        )
        log.close()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _is_ours(self) -> bool:
        """Only send the key once the server on the port proves it's the one we started."""
        if not self.verified:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.cfg.port}/instance", timeout=2) as r:
                    self.verified = json.loads(r.read()).get("instance") == self.instance
            except (urllib.error.URLError, OSError, ValueError):
                return False
        return self.verified

    def health(self) -> dict | None:
        if not self._is_ours():
            return None
        req = urllib.request.Request(f"http://127.0.0.1:{self.cfg.port}/health",
                                     headers={"X-Api-Key": self.cfg.api_key})
        try:
            with urllib.request.urlopen(req, timeout=2) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, OSError, ValueError):
            return None

    def wait_ready(self, timeout_s: float = 120) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.running():
                return False
            if self.health():
                return True
            time.sleep(0.5)
        return False

    def ui_ticket(self) -> str | None:
        if not self._is_ours():
            return None
        req = urllib.request.Request(f"http://127.0.0.1:{self.cfg.port}/ui-ticket", method="POST",
                                     headers={"X-Api-Key": self.cfg.api_key})
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read())["ticket"]
        except (urllib.error.URLError, OSError, ValueError, KeyError):
            return None

    def log_tail(self, lines: int = 15) -> str:
        try:
            return "\n".join(self.log_path.read_text(errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()  # the server's watchdog exits on this
            self.proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

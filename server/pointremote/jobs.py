from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

from .progress import ProgressFn


class AlreadyRunning(Exception):
    pass


@dataclass
class _Status:
    state: str = "idle"
    epoch: int = 0
    epochs: int = 0
    message: str = ""
    version: int | None = None


class TrainingRunner:
    def __init__(self, fn: Callable[[ProgressFn], dict]):
        self._fn = fn
        self._lock = threading.Lock()
        self._status = _Status()
        self._thread: threading.Thread | None = None

    def status(self) -> dict:
        with self._lock:
            return asdict(self._status)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._status.state == "running"

    def start(self) -> None:
        with self._lock:
            if self._status.state == "running":
                raise AlreadyRunning()
            self._status = _Status(state="running", message="starting")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def _progress(self, epoch: int, total: int, message: str) -> None:
        with self._lock:
            self._status.epoch = epoch
            self._status.epochs = total
            self._status.message = message

    def _run(self) -> None:
        try:
            meta = self._fn(self._progress)
        except Exception as e:  # report any training failure to the client
            with self._lock:
                self._status.state = "failed"
                self._status.message = str(e)
            return
        with self._lock:
            self._status.state = "done"
            self._status.version = meta.get("version")
            acc = meta.get("val_accuracy")
            self._status.message = f"val_accuracy={acc:.3f}" if acc is not None else "done"

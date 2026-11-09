"""Sets up the server's own Python environment (PyTorch and friends) for the frozen app."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from . import VERSION
from .paths import bundle_dir, frozen, server_source_dir

Progress = Callable[[str], None]

# Windows: keep child processes from flashing a console window.
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class SetupFailed(Exception):
    pass


def venv_python(env_dir: Path) -> Path:
    return env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _stamp(src: Path) -> str:
    h = hashlib.sha256(VERSION.encode())
    h.update((src / "uv.lock").read_bytes())
    for f in sorted((src / "pointremote").rglob("*.py")):
        h.update(f.relative_to(src).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def _copy_source(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(src / name, dest / name)
    shutil.rmtree(dest / "pointremote", ignore_errors=True)
    shutil.copytree(src / "pointremote", dest / "pointremote",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def uv_binary() -> Path:
    return bundle_dir() / ("uv.exe" if os.name == "nt" else "uv")


def ensure_runtime(data_dir: Path, progress: Progress) -> tuple[Path, Path]:
    """Returns (python, working dir) to run the server with, installing it first if needed.

    From the repo (dev mode) that's just this Python. The frozen app copies the bundled
    server source to data_dir/runtime and lets uv install Python and the dependencies."""
    if not frozen():
        return Path(sys.executable), server_source_dir()

    src, runtime = server_source_dir(), data_dir / "runtime"
    python = venv_python(runtime / ".venv")
    stamp_file, stamp = runtime / ".installed", _stamp(src)
    if python.exists() and stamp_file.is_file() and stamp_file.read_text() == stamp:
        return python, runtime

    progress("Copying the server files")
    _copy_source(src, runtime)
    stamp_file.unlink(missing_ok=True)
    env = os.environ | {
        "UV_PYTHON_INSTALL_DIR": str(data_dir / "python"),
        # Cache next to the environment so uv can hardlink instead of storing PyTorch twice.
        "UV_CACHE_DIR": str(data_dir / "uv-cache"),
        "UV_PYTHON_PREFERENCE": "only-managed",
        "UV_NO_PROGRESS": "1",
    }
    progress("Downloading Python, PyTorch and the rest (about 3 GB; needs about 7 GB of disk)")
    cmd = [str(uv_binary()), "sync", "--frozen", "--no-dev", "--project", str(runtime), "--python", "3.12"]
    proc = subprocess.Popen(cmd, cwd=runtime, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", creationflags=NO_WINDOW)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        tail = (tail + [line])[-20:]
        if line.startswith(("Downloading", "Downloaded", "Installed", "Prepared", "Using", "Creating")):
            progress(line)
    if proc.wait() != 0 or not python.exists():
        raise SetupFailed("Setting up the server failed:\n" + "\n".join(tail))
    stamp_file.write_text(stamp)
    progress("Setup finished")
    return python, runtime

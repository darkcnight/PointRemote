"""Adds the frozen Linux app to the desktop's app menu on first run."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .paths import asset, data_dir

MARKER = "X-PointRemote-Created-By=app"


def launch_path() -> Path:
    """What the menu should run. An AppImage runs from a temporary mount that changes every
    launch, so point at the .AppImage file itself."""
    appimage = os.environ.get("APPIMAGE")
    return Path(appimage) if appimage else Path(sys.executable).resolve()


def entry_text(icon: Path) -> str:
    return (
        "[Desktop Entry]\nType=Application\nName=PointRemote\n"
        "Comment=Training server for the PointRemote phone app\n"
        f'Exec="{launch_path()}"\nIcon={icon}\nTerminal=false\nCategories=Utility;\n'
        f"StartupWMClass=PointRemote\n{MARKER}\n"
    )


def ensure_menu_entry(apps: Path | None = None) -> None:
    try:
        apps = apps or Path.home() / ".local/share/applications"
        entry = apps / "pointremote.desktop"
        # Leave alone an entry someone made by hand (e.g. one that runs the repo version).
        if entry.exists() and MARKER not in entry.read_text(errors="replace"):
            return
        icon = data_dir() / "icon-256.png"
        data_dir().mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset("icon-256.png"), icon)
        apps.mkdir(parents=True, exist_ok=True)
        entry.write_text(entry_text(icon))
    except OSError:
        pass  # a missing menu entry shouldn't stop the app

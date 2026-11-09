"""Adds the frozen Linux app to the desktop's app menu on first run."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .paths import asset, data_dir


def ensure_menu_entry() -> None:
    try:
        apps = Path.home() / ".local/share/applications"
        icon = data_dir() / "icon-256.png"
        data_dir().mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset("icon-256.png"), icon)
        apps.mkdir(parents=True, exist_ok=True)
        (apps / "pointremote.desktop").write_text(
            "[Desktop Entry]\nType=Application\nName=PointRemote\n"
            "Comment=Training server for the PointRemote phone app\n"
            f'Exec="{Path(sys.executable).resolve()}"\nIcon={icon}\nTerminal=false\nCategories=Utility;\n'
        )
    except OSError:
        pass  # a missing menu entry shouldn't stop the app

from __future__ import annotations

import sys
from pathlib import Path

from platformdirs import user_config_path, user_data_path

APP = "PointRemote"


def frozen() -> bool:
    """True when running as the PyInstaller-built app rather than from the repo."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> Path:
    """Where the frozen app's bundled files (server source, uv, guide) are unpacked."""
    return Path(getattr(sys, "_MEIPASS", Path(__file__).parent))


def package_dir() -> Path:
    return bundle_dir() / "pointremote" / "desktop" if frozen() else Path(__file__).parent


def server_source_dir() -> Path:
    """The folder holding pyproject.toml and the pointremote package."""
    return bundle_dir() / "server_src" if frozen() else Path(__file__).parents[2]


def config_dir() -> Path:
    return user_config_path(APP, appauthor=False)


def data_dir() -> Path:
    return user_data_path(APP, appauthor=False)


def guide_path() -> Path:
    return package_dir() / "guide.html"


def asset(name: str) -> Path:
    return package_dir() / "assets" / name

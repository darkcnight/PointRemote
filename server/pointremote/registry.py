from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


class ModelRegistry:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def versions(self) -> list[int]:
        return sorted(
            int(p.name[1:])
            for p in self.root.glob("v*")
            if p.is_dir() and not p.is_symlink() and p.name[1:].isdigit()
        )

    def next_version(self) -> int:
        versions = self.versions()
        return versions[-1] + 1 if versions else 1

    def staging_dir(self, version: int) -> Path:
        d = self.root / f".staging-v{version}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        return d

    def publish(self, version: int, staging: Path) -> Path:
        final = self.root / f"v{version}"
        staging.rename(final)
        # A pointer file rather than a symlink: Windows can't atomically replace a directory link.
        tmp = self.root / "LATEST.tmp"
        tmp.write_text(final.name)
        os.replace(tmp, self.root / "LATEST")
        return final

    def latest_dir(self) -> Path | None:
        pointer = self.root / "LATEST"
        if pointer.is_file():
            d = self.root / pointer.read_text().strip()
            return d if d.is_dir() else None
        link = self.root / "latest"  # older installs used a symlink
        return link.resolve() if link.exists() else None

    def latest_meta(self) -> dict | None:
        d = self.latest_dir()
        return json.loads((d / "meta.json").read_text()) if d else None

    def history(self) -> list[dict]:
        metas = []
        for v in reversed(self.versions()):
            meta_path = self.root / f"v{v}" / "meta.json"
            if meta_path.exists():
                metas.append(json.loads(meta_path.read_text()))
        return metas

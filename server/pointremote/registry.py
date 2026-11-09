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
        tmp = self.root / "latest.tmp"
        if tmp.is_symlink() or tmp.exists():
            tmp.unlink()
        tmp.symlink_to(final.name)
        os.replace(tmp, self.root / "latest")
        return final

    def latest_dir(self) -> Path | None:
        link = self.root / "latest"
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

from __future__ import annotations

import io
import re
import shutil
import uuid
from pathlib import Path

from PIL import Image, UnidentifiedImageError

_NAME_RE = re.compile(r"^[0-9a-f]{32}\.jpg$")


class InvalidImage(ValueError):
    pass


class PhotoNotFound(Exception):
    pass


class PhotoStore:
    def __init__(self, data_dir: Path):
        self.root = data_dir / "dataset"
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, label_id: str) -> Path:
        return self.root / label_id

    def save(self, label_id: str, data: bytes) -> str:
        try:
            with Image.open(io.BytesIO(data)) as img:
                if img.format != "JPEG":
                    raise InvalidImage(f"expected JPEG, got {img.format}")
                img.verify()
        except (UnidentifiedImageError, OSError) as e:
            raise InvalidImage(str(e)) from e
        directory = self._dir(label_id)
        directory.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex}.jpg"
        (directory / name).write_bytes(data)
        return name

    def names(self, label_id: str) -> list[str]:
        directory = self._dir(label_id)
        if not directory.exists():
            return []
        return sorted(p.name for p in directory.glob("*.jpg"))

    def count(self, label_id: str) -> int:
        return len(self.names(label_id))

    def path(self, label_id: str, name: str) -> Path:
        if not _NAME_RE.match(name):
            raise PhotoNotFound(name)
        p = self._dir(label_id) / name
        if not p.is_file():
            raise PhotoNotFound(name)
        return p

    def delete(self, label_id: str, name: str) -> None:
        self.path(label_id, name).unlink()

    def delete_label(self, label_id: str) -> None:
        shutil.rmtree(self._dir(label_id), ignore_errors=True)

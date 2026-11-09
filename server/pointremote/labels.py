from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

UNKNOWN_ID = "unknown"


@dataclass(frozen=True)
class Label:
    id: str
    entity_id: str | None
    friendly_name: str


UNKNOWN = Label(UNKNOWN_ID, None, "Unknown")

_ENTITY_RE = re.compile(r"^[a-z_]+\.[a-z0-9_]+$")


class LabelExists(Exception):
    pass


class LabelNotFound(Exception):
    pass


class InvalidEntityId(ValueError):
    pass


def label_id_for(entity_id: str) -> str:
    if not _ENTITY_RE.match(entity_id):
        raise InvalidEntityId(entity_id)
    return entity_id.replace(".", "_")


class LabelStore:
    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "labels.json"
        self._lock = threading.Lock()

    def _read(self) -> list[Label]:
        if not self._path.exists():
            return []
        return [Label(**d) for d in json.loads(self._path.read_text())]

    def _write(self, labels: list[Label]) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(l) for l in labels], indent=2))
        tmp.replace(self._path)

    def all(self) -> list[Label]:
        with self._lock:
            return [UNKNOWN, *self._read()]

    def get(self, label_id: str) -> Label:
        for label in self.all():
            if label.id == label_id:
                return label
        raise LabelNotFound(label_id)

    def create(self, entity_id: str, friendly_name: str) -> Label:
        label = Label(label_id_for(entity_id), entity_id, friendly_name)
        with self._lock:
            labels = self._read()
            if any(l.id == label.id for l in labels):
                raise LabelExists(label.id)
            labels.append(label)
            self._write(labels)
        return label

    def delete(self, label_id: str) -> None:
        if label_id == UNKNOWN_ID:
            raise ValueError("the unknown label cannot be deleted")
        with self._lock:
            labels = self._read()
            remaining = [l for l in labels if l.id != label_id]
            if len(remaining) == len(labels):
                raise LabelNotFound(label_id)
            self._write(remaining)

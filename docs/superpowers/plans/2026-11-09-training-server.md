# PointRemote Training Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan 1 of 2.** This plan builds the PC training server. Plan 2 (Android app) is written separately and consumes the HTTP API defined here.

**Goal:** A FastAPI server on the desktop that stores labelled photos, fine-tunes MobileNetV3 on the GPU, exports a verified ONNX model, and serves it to the phone, plus a small web UI for the dataset and training history.

**Architecture:** Small single-purpose modules (label store, photo store, split, model registry, trainer, background training runner) composed in `create_app()`. All state lives on disk under `data/`. Training runs in one background thread; the HTTP layer only starts it and reports status. The trainer writes into a staging directory that the registry atomically publishes as `models/v<N>` with a `latest` symlink.

**Tech Stack:** Python 3.12, uv, FastAPI, uvicorn, Pillow, PyTorch + torchvision (CUDA 12.8 wheels, needed for the RTX 5070 Ti), onnx, onnxruntime, pytest, httpx.

**Spec:** `docs/superpowers/specs/2026-11-09-point-remote-design.md`

## Global Constraints

- Python `>=3.12,<3.13`, managed with `uv`; all commands run from `server/`.
- Every JSON endpoint requires header `X-Api-Key` equal to the configured key; the web UI uses cookie `pr_key` with the same value.
- Config via env: `POINTREMOTE_API_KEY` (required), `POINTREMOTE_DATA_DIR` (default `data`), `POINTREMOTE_MIN_PHOTOS` (default `30`).
- Storage layout: `data/labels.json`, `data/dataset/<label>/<32-hex>.jpg`, `data/models/v<N>/{model.onnx,meta.json}`, `data/models/latest -> v<N>`.
- The `unknown` label always exists, is always class 0, and cannot be deleted.
- Label id = entity id with `.` replaced by `_` (`light.kitchen` → `light_kitchen`).
- Model input 224×224 RGB, ImageNet mean/std normalisation, centre square crop then resize; output named `logits` (app applies softmax).
- Augmentation: random resized crop scale 0.6–1.0, rotation ±10°, brightness/contrast jitter 0.3. **No horizontal flip.**
- Stratified 80/20 split with a recorded seed.
- ONNX export is verified with onnxruntime; val accuracy must match PyTorch within 0.01 or the run fails and `latest` is unchanged.
- Only one training run at a time (`POST /train` → 409 while running).
- Server port `8765`.

## File Structure

```
server/
  pyproject.toml            deps, torch CUDA index, pytest config
  .python-version           3.12
  .gitignore
  run.sh                    starts uvicorn, loads .env
  README.md
  pointremote/
    __init__.py
    config.py               Settings / TrainConfig, env loading
    labels.py               Label, LabelStore (labels.json)
    photos.py               PhotoStore (dataset/<label>/*.jpg)
    split.py                stratified_split
    registry.py             ModelRegistry (versions, staging, latest)
    progress.py             ProgressFn type
    jobs.py                 TrainingRunner (background thread + status)
    trainer.py              model, transforms, train loop, ONNX export + verify, meta
    pipeline.py             training_blockers, run_training (glue)
    auth.py                 key checks
    api.py                  create_app + JSON endpoints
    ui.py                   /ui HTML pages
    main.py                 module-level app for uvicorn
  tests/
    conftest.py
    test_config.py test_labels.py test_photos.py test_split.py
    test_registry.py test_jobs.py test_trainer.py test_pipeline.py
    test_api.py test_ui.py
```

---

### Task 1: Project scaffold and config

**Files:**
- Create: `server/pyproject.toml`, `server/.python-version`, `server/.gitignore`, `server/pointremote/__init__.py`, `server/pointremote/config.py`
- Test: `server/tests/conftest.py`, `server/tests/test_config.py`

**Interfaces:**
- Produces: `TrainConfig(epochs_head=5, epochs_full=15, batch_size=32, lr_head=1e-3, lr_full=1e-4, pretrained=True, device="auto", seed=1234, num_workers=4)` (frozen dataclass); `Settings(data_dir: Path, api_key: str, min_photos: int = 30, train: TrainConfig = TrainConfig())` (frozen dataclass) with `Settings.from_env() -> Settings`.
- Produces (fixtures): `settings` (tmp data dir, key `"test-key"`, `min_photos=1`), `make_jpeg(color=(200,30,30), size=(64,64), fmt="JPEG") -> bytes`.

- [ ] **Step 1: Create project files**

`server/pyproject.toml`:
```toml
[project]
name = "pointremote-server"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = [
  "fastapi>=0.115",
  "uvicorn[standard]>=0.32",
  "python-multipart>=0.0.12",
  "pillow>=11",
  "numpy>=2",
  "torch>=2.8",
  "torchvision>=0.23",
  "onnx>=1.17",
  "onnxruntime>=1.20",
]

[dependency-groups]
dev = ["pytest>=8", "httpx>=0.27"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["pointremote"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["gpu: needs a CUDA GPU"]
```

`server/.python-version`:
```
3.12
```

`server/.gitignore`:
```
.venv/
__pycache__/
data/
.env
```

`server/pointremote/__init__.py`: empty file.

- [ ] **Step 2: Install dependencies**

Run: `cd server && uv sync`
Expected: completes; `uv run python -c "import torch, torchvision, onnxruntime, fastapi; print(torch.__version__)"` prints a version ending in `+cu128`.

- [ ] **Step 3: Write the failing tests**

`server/tests/conftest.py`:
```python
import io

import pytest
from PIL import Image

from pointremote.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", api_key="test-key", min_photos=1)


@pytest.fixture
def make_jpeg():
    def make(color=(200, 30, 30), size=(64, 64), fmt="JPEG") -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format=fmt)
        return buf.getvalue()

    return make
```

`server/tests/test_config.py`:
```python
import pytest

from pointremote.config import Settings, TrainConfig


def test_from_env_reads_values(monkeypatch, tmp_path):
    monkeypatch.setenv("POINTREMOTE_API_KEY", "k")
    monkeypatch.setenv("POINTREMOTE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("POINTREMOTE_MIN_PHOTOS", "12")
    s = Settings.from_env()
    assert s.api_key == "k"
    assert s.data_dir == tmp_path.resolve()
    assert s.min_photos == 12
    assert s.train == TrainConfig()


def test_from_env_defaults(monkeypatch):
    monkeypatch.setenv("POINTREMOTE_API_KEY", "k")
    monkeypatch.delenv("POINTREMOTE_DATA_DIR", raising=False)
    monkeypatch.delenv("POINTREMOTE_MIN_PHOTOS", raising=False)
    s = Settings.from_env()
    assert s.data_dir.name == "data"
    assert s.min_photos == 30


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("POINTREMOTE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="POINTREMOTE_API_KEY"):
        Settings.from_env()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.config'`

- [ ] **Step 5: Implement config**

`server/pointremote/config.py`:
```python
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    epochs_head: int = 5
    epochs_full: int = 15
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_full: float = 1e-4
    pretrained: bool = True
    device: str = "auto"
    seed: int = 1234
    num_workers: int = 4


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    api_key: str
    min_photos: int = 30
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_env(cls) -> Settings:
        key = os.environ.get("POINTREMOTE_API_KEY")
        if not key:
            raise RuntimeError("POINTREMOTE_API_KEY must be set")
        return cls(
            data_dir=Path(os.environ.get("POINTREMOTE_DATA_DIR", "data")).resolve(),
            api_key=key,
            min_photos=int(os.environ.get("POINTREMOTE_MIN_PHOTOS", "30")),
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 7: Commit**

```bash
git add server/
git commit -m "feat(server): scaffold project and settings"
```

---

### Task 2: Label store

**Files:**
- Create: `server/pointremote/labels.py`
- Test: `server/tests/test_labels.py`

**Interfaces:**
- Produces: `Label(id: str, entity_id: str | None, friendly_name: str)` frozen dataclass; `UNKNOWN_ID = "unknown"`; `UNKNOWN = Label("unknown", None, "Unknown")`; `label_id_for(entity_id: str) -> str`; exceptions `LabelExists`, `LabelNotFound`, `InvalidEntityId(ValueError)`; `LabelStore(data_dir: Path)` with `all() -> list[Label]` (UNKNOWN first, then creation order), `get(label_id) -> Label`, `create(entity_id, friendly_name) -> Label`, `delete(label_id) -> None` (raises `ValueError` for unknown).

- [ ] **Step 1: Write the failing tests**

`server/tests/test_labels.py`:
```python
import pytest

from pointremote.labels import (
    UNKNOWN,
    InvalidEntityId,
    Label,
    LabelExists,
    LabelNotFound,
    LabelStore,
    label_id_for,
)


def test_unknown_always_first(tmp_path):
    assert LabelStore(tmp_path).all() == [UNKNOWN]


def test_create_persists(tmp_path):
    LabelStore(tmp_path).create("light.kitchen", "Kitchen lights")
    LabelStore(tmp_path).create("cover.big_shutter", "Big shutter")
    labels = LabelStore(tmp_path).all()
    assert labels == [
        UNKNOWN,
        Label("light_kitchen", "light.kitchen", "Kitchen lights"),
        Label("cover_big_shutter", "cover.big_shutter", "Big shutter"),
    ]


def test_duplicate_rejected(tmp_path):
    store = LabelStore(tmp_path)
    store.create("light.kitchen", "Kitchen lights")
    with pytest.raises(LabelExists):
        store.create("light.kitchen", "Again")


@pytest.mark.parametrize("bad", ["kitchen", "Light.Kitchen", "light.", "../etc", "light.a/b"])
def test_invalid_entity_id(tmp_path, bad):
    with pytest.raises(InvalidEntityId):
        LabelStore(tmp_path).create(bad, "x")


def test_label_id_for():
    assert label_id_for("media_player.tv") == "media_player_tv"


def test_get(tmp_path):
    store = LabelStore(tmp_path)
    store.create("light.kitchen", "Kitchen lights")
    assert store.get("light_kitchen").entity_id == "light.kitchen"
    assert store.get("unknown") == UNKNOWN
    with pytest.raises(LabelNotFound):
        store.get("nope")


def test_delete(tmp_path):
    store = LabelStore(tmp_path)
    store.create("light.kitchen", "Kitchen lights")
    store.delete("light_kitchen")
    assert store.all() == [UNKNOWN]
    with pytest.raises(LabelNotFound):
        store.delete("light_kitchen")
    with pytest.raises(ValueError):
        store.delete("unknown")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_labels.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.labels'`

- [ ] **Step 3: Implement**

`server/pointremote/labels.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_labels.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/labels.py server/tests/test_labels.py
git commit -m "feat(server): label store with unknown class"
```

---

### Task 3: Photo store

**Files:**
- Create: `server/pointremote/photos.py`
- Test: `server/tests/test_photos.py`

**Interfaces:**
- Consumes: label ids (already validated by `LabelStore`).
- Produces: exceptions `InvalidImage(ValueError)`, `PhotoNotFound`; `PhotoStore(data_dir: Path)` with attribute `root: Path` (= `data_dir/"dataset"`), `save(label_id, data: bytes) -> str` (returns file name `<32hex>.jpg`), `names(label_id) -> list[str]` (sorted), `count(label_id) -> int`, `path(label_id, name) -> Path`, `delete(label_id, name) -> None`, `delete_label(label_id) -> None`.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_photos.py`:
```python
import pytest

from pointremote.photos import InvalidImage, PhotoNotFound, PhotoStore


def test_save_and_count(tmp_path, make_jpeg):
    store = PhotoStore(tmp_path)
    data = make_jpeg()
    name = store.save("light_kitchen", data)
    assert len(name) == 36 and name.endswith(".jpg")
    assert store.count("light_kitchen") == 1
    assert store.names("light_kitchen") == [name]
    assert store.path("light_kitchen", name).read_bytes() == data
    assert store.path("light_kitchen", name).parent == tmp_path / "dataset" / "light_kitchen"


def test_count_missing_label_is_zero(tmp_path):
    assert PhotoStore(tmp_path).count("nothing") == 0


def test_rejects_garbage(tmp_path):
    with pytest.raises(InvalidImage):
        PhotoStore(tmp_path).save("x", b"not an image")


def test_rejects_png(tmp_path, make_jpeg):
    with pytest.raises(InvalidImage):
        PhotoStore(tmp_path).save("x", make_jpeg(fmt="PNG"))


def test_path_rejects_traversal(tmp_path):
    with pytest.raises(PhotoNotFound):
        PhotoStore(tmp_path).path("x", "../labels.json")


def test_delete_and_delete_label(tmp_path, make_jpeg):
    store = PhotoStore(tmp_path)
    a = store.save("x", make_jpeg())
    store.save("x", make_jpeg())
    store.delete("x", a)
    assert store.count("x") == 1
    with pytest.raises(PhotoNotFound):
        store.delete("x", a)
    store.delete_label("x")
    assert store.count("x") == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_photos.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.photos'`

- [ ] **Step 3: Implement**

`server/pointremote/photos.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_photos.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/photos.py server/tests/test_photos.py
git commit -m "feat(server): photo store with JPEG validation"
```

---

### Task 4: Stratified split

**Files:**
- Create: `server/pointremote/split.py`
- Test: `server/tests/test_split.py`

**Interfaces:**
- Produces: `stratified_split(items: list[tuple[Path, int]], val_fraction: float = 0.2, seed: int = 0) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]`. Per class: classes with ≥ 2 items put `max(1, round(n * val_fraction))` in val; classes with 1 item go entirely to train. Result is independent of input order.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_split.py`:
```python
import random
from collections import Counter
from pathlib import Path

from pointremote.split import stratified_split


def _items(counts):
    items = []
    for cls, n in enumerate(counts):
        items += [(Path(f"c{cls}/{i:03}.jpg"), cls) for i in range(n)]
    return items


def test_split_is_stratified():
    train, val = stratified_split(_items([10, 20, 5]), 0.2, seed=1)
    assert Counter(c for _, c in val) == {0: 2, 1: 4, 2: 1}
    assert len(train) + len(val) == 35
    assert not set(train) & set(val)


def test_split_is_deterministic_and_order_independent():
    items = _items([10, 20, 5])
    shuffled = items[:]
    random.Random(99).shuffle(shuffled)
    assert stratified_split(items, 0.2, seed=3) == stratified_split(shuffled, 0.2, seed=3)


def test_different_seed_changes_split():
    items = _items([30])
    assert stratified_split(items, 0.2, seed=1)[1] != stratified_split(items, 0.2, seed=2)[1]


def test_single_item_class_goes_to_train():
    train, val = stratified_split(_items([1, 5]), 0.2, seed=0)
    assert [c for _, c in train].count(0) == 1
    assert all(c == 1 for _, c in val)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.split'`

- [ ] **Step 3: Implement**

`server/pointremote/split.py`:
```python
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

Item = tuple[Path, int]


def stratified_split(
    items: list[Item], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[Item], list[Item]]:
    rng = random.Random(seed)
    by_class: dict[int, list[Item]] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)
    train: list[Item] = []
    val: list[Item] = []
    for cls in sorted(by_class):
        group = sorted(by_class[cls], key=lambda it: str(it[0]))
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) >= 2 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return train, val
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_split.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/split.py server/tests/test_split.py
git commit -m "feat(server): deterministic stratified split"
```

---

### Task 5: Model registry

**Files:**
- Create: `server/pointremote/registry.py`
- Test: `server/tests/test_registry.py`

**Interfaces:**
- Produces: `ModelRegistry(data_dir: Path)` with attribute `root: Path` (= `data_dir/"models"`), `versions() -> list[int]` (ascending), `next_version() -> int`, `staging_dir(version: int) -> Path` (fresh empty `.staging-v<N>`), `publish(version: int, staging: Path) -> Path` (renames to `v<N>`, atomically repoints `latest`), `latest_dir() -> Path | None`, `latest_meta() -> dict | None`, `history() -> list[dict]` (metas, newest first).

- [ ] **Step 1: Write the failing tests**

`server/tests/test_registry.py`:
```python
import json

from pointremote.registry import ModelRegistry


def publish_fake(registry: ModelRegistry, **meta) -> int:
    v = registry.next_version()
    d = registry.staging_dir(v)
    (d / "model.onnx").write_bytes(b"onnx")
    (d / "meta.json").write_text(json.dumps({"version": v, **meta}))
    registry.publish(v, d)
    return v


def test_empty(tmp_path):
    r = ModelRegistry(tmp_path)
    assert r.versions() == []
    assert r.next_version() == 1
    assert r.latest_dir() is None
    assert r.latest_meta() is None
    assert r.history() == []


def test_publish_moves_latest(tmp_path):
    r = ModelRegistry(tmp_path)
    publish_fake(r, val_accuracy=0.5)
    publish_fake(r, val_accuracy=0.9)
    assert r.versions() == [1, 2]
    assert r.latest_meta() == {"version": 2, "val_accuracy": 0.9}
    assert r.latest_dir().name == "v2"
    assert (r.latest_dir() / "model.onnx").read_bytes() == b"onnx"
    assert [m["version"] for m in r.history()] == [2, 1]
    assert not list(r.root.glob(".staging-*"))


def test_staging_dir_is_fresh(tmp_path):
    r = ModelRegistry(tmp_path)
    (r.staging_dir(1) / "leftover").write_text("x")
    assert list(r.staging_dir(1).iterdir()) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.registry'`

- [ ] **Step 3: Implement**

`server/pointremote/registry.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_registry.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/registry.py server/tests/test_registry.py
git commit -m "feat(server): versioned model registry with atomic latest"
```

---

### Task 6: Background training runner

**Files:**
- Create: `server/pointremote/progress.py`, `server/pointremote/jobs.py`
- Test: `server/tests/test_jobs.py`

**Interfaces:**
- Produces: `ProgressFn = Callable[[int, int, str], None]` (epoch, total_epochs, message) and `no_progress(epoch, total, message) -> None` in `progress.py`.
- Produces: `AlreadyRunning(Exception)`; `TrainingRunner(fn: Callable[[ProgressFn], dict])` with `start() -> None`, `status() -> dict` (keys `state` ∈ `idle|running|done|failed`, `epoch`, `epochs`, `message`, `version`), property `running -> bool`, `join(timeout: float | None = None) -> None`. On success `version` = `fn(...)["version"]`, `message` = `"val_accuracy=<x.xxx>"` when present.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_jobs.py`:
```python
import threading

import pytest

from pointremote.jobs import AlreadyRunning, TrainingRunner


def test_idle_initially():
    r = TrainingRunner(lambda progress: {"version": 1})
    assert r.status() == {"state": "idle", "epoch": 0, "epochs": 0, "message": "", "version": None}
    assert not r.running


def test_runs_to_done():
    def fn(progress):
        progress(1, 2, "a")
        progress(2, 2, "b")
        return {"version": 3, "val_accuracy": 0.9}

    r = TrainingRunner(fn)
    r.start()
    r.join(5)
    s = r.status()
    assert s["state"] == "done"
    assert s["version"] == 3
    assert (s["epoch"], s["epochs"]) == (2, 2)
    assert s["message"] == "val_accuracy=0.900"


def test_failure_reported():
    def fn(progress):
        raise ValueError("boom")

    r = TrainingRunner(fn)
    r.start()
    r.join(5)
    assert r.status()["state"] == "failed"
    assert r.status()["message"] == "boom"


def test_cannot_start_twice_but_can_restart_after():
    gate = threading.Event()

    def fn(progress):
        gate.wait(5)
        return {"version": 1}

    r = TrainingRunner(fn)
    r.start()
    assert r.running
    with pytest.raises(AlreadyRunning):
        r.start()
    gate.set()
    r.join(5)
    r.start()
    r.join(5)
    assert r.status()["state"] == "done"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.jobs'`

- [ ] **Step 3: Implement**

`server/pointremote/progress.py`:
```python
from collections.abc import Callable

ProgressFn = Callable[[int, int, str], None]


def no_progress(epoch: int, total: int, message: str) -> None:
    pass
```

`server/pointremote/jobs.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_jobs.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/progress.py server/pointremote/jobs.py server/tests/test_jobs.py
git commit -m "feat(server): single-run background training runner"
```

---

### Task 7: Trainer (fine-tune, export, verify, meta)

**Files:**
- Create: `server/pointremote/trainer.py`
- Test: `server/tests/test_trainer.py`

**Interfaces:**
- Consumes: `TrainConfig` (Task 1), `Label`/`UNKNOWN` (Task 2), `stratified_split` (Task 4), `ProgressFn`/`no_progress` (Task 6).
- Produces: constants `INPUT_SIZE = 224`, `MEAN`, `STD`; `ExportMismatch(RuntimeError)`; `EvalResult(accuracy: float, per_class: list[float], confusion: list[list[int]])`; `evaluate_predictions(y_true: list[int], y_pred: list[int], num_classes: int) -> EvalResult`; `top_confusions(confusion, classes: list[Label], limit=5) -> list[dict]` (`{"true", "predicted", "count"}`); `TrainResult(meta: dict, model_path: Path)`; `train(dataset_dir: Path, classes: list[Label], out_dir: Path, version: int, config: TrainConfig, progress: ProgressFn = no_progress) -> TrainResult`. Class index = position in `classes`. Writes `out_dir/model.onnx` and `out_dir/meta.json`.
- meta.json keys: `version, created_at, classes[{id, entity_id, friendly_name}], input_size, normalization{mean, std}, preprocessing, output, val_accuracy, per_class_accuracy{id: acc}, confusion_matrix, confusions, train_count, val_count, seed, epochs, sha256, size_bytes`.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_trainer.py`:
```python
import hashlib
import json

import numpy as np
import onnxruntime as ort
import pytest
import torch

from pointremote.config import TrainConfig
from pointremote.labels import UNKNOWN, Label
from pointremote.trainer import evaluate_predictions, top_confusions, train

TINY = TrainConfig(epochs_head=1, epochs_full=1, batch_size=4, pretrained=False, device="cpu", num_workers=0)
CLASSES = [UNKNOWN, Label("light_a", "light.a", "A"), Label("light_b", "light.b", "B")]


@pytest.fixture
def tiny_dataset(tmp_path, make_jpeg):
    colors = [(20, 20, 200), (200, 20, 20), (20, 200, 20)]
    root = tmp_path / "dataset"
    for label, color in zip(CLASSES, colors):
        d = root / label.id
        d.mkdir(parents=True)
        for i in range(6):
            (d / f"{i:032x}.jpg").write_bytes(make_jpeg(color=color, size=(64, 48)))
    return root


def test_evaluate_predictions():
    r = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], 2)
    assert r.accuracy == 0.75
    assert r.per_class == [0.5, 1.0]
    assert r.confusion == [[1, 1], [0, 2]]


def test_top_confusions():
    conf = [[5, 0, 2], [1, 4, 0], [0, 3, 6]]
    assert top_confusions(conf, CLASSES, limit=2) == [
        {"true": "light_b", "predicted": "light_a", "count": 3},
        {"true": "unknown", "predicted": "light_b", "count": 2},
    ]


def test_train_produces_verified_onnx(tiny_dataset, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    seen = []
    result = train(tiny_dataset, CLASSES, out, version=7, config=TINY,
                   progress=lambda e, t, m: seen.append((e, t)))

    meta = json.loads((out / "meta.json").read_text())
    assert meta == result.meta
    assert result.model_path == out / "model.onnx"
    assert meta["version"] == 7
    assert [c["id"] for c in meta["classes"]] == ["unknown", "light_a", "light_b"]
    assert meta["classes"][1]["entity_id"] == "light.a"
    assert meta["input_size"] == 224
    assert meta["output"] == "logits"
    assert meta["normalization"]["mean"] == [0.485, 0.456, 0.406]
    assert (meta["train_count"], meta["val_count"]) == (15, 3)
    assert 0.0 <= meta["val_accuracy"] <= 1.0
    assert set(meta["per_class_accuracy"]) == {"unknown", "light_a", "light_b"}
    assert len(meta["confusion_matrix"]) == 3
    model_bytes = (out / "model.onnx").read_bytes()
    assert meta["sha256"] == hashlib.sha256(model_bytes).hexdigest()
    assert meta["size_bytes"] == len(model_bytes)
    assert seen == [(1, 2), (2, 2)]

    sess = ort.InferenceSession(str(out / "model.onnx"), providers=["CPUExecutionProvider"])
    logits = sess.run(["logits"], {"input": np.zeros((2, 3, 224, 224), np.float32)})[0]
    assert logits.shape == (2, 3)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_train_on_gpu_with_pretrained_weights(tiny_dataset, tmp_path):
    config = TrainConfig(epochs_head=1, epochs_full=1, batch_size=4, pretrained=True, device="cuda", num_workers=0)
    out = tmp_path / "out"
    out.mkdir()
    result = train(tiny_dataset, CLASSES, out, version=1, config=config)
    assert result.model_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.trainer'`

- [ ] **Step 3: Implement**

`server/pointremote/trainer.py`:
```python
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

from .config import TrainConfig
from .labels import Label
from .progress import ProgressFn, no_progress
from .split import stratified_split

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
MAX_ONNX_ACCURACY_GAP = 0.01


class ExportMismatch(RuntimeError):
    pass


def eval_transform() -> transforms.Compose:
    # Same as the app: centre square crop, resize to 224.
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform() -> transforms.Compose:
    # No horizontal flip: room scenes are not mirror-symmetric.
    return transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.6, 1.0)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class ImageList(Dataset):
    def __init__(self, items: list[tuple[Path, int]], transform: transforms.Compose):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, label


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


@dataclass
class EvalResult:
    accuracy: float
    per_class: list[float]
    confusion: list[list[int]]  # confusion[true][predicted]


def evaluate_predictions(y_true: list[int], y_pred: list[int], num_classes: int) -> EvalResult:
    conf = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        conf[t][p] += 1
    correct = sum(conf[i][i] for i in range(num_classes))
    per_class = [conf[i][i] / sum(conf[i]) if sum(conf[i]) else 0.0 for i in range(num_classes)]
    accuracy = correct / len(y_true) if y_true else 0.0
    return EvalResult(accuracy, per_class, conf)


def top_confusions(confusion: list[list[int]], classes: list[Label], limit: int = 5) -> list[dict]:
    n = len(confusion)
    pairs = [(confusion[t][p], t, p) for t in range(n) for p in range(n) if t != p and confusion[t][p] > 0]
    pairs.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [{"true": classes[t].id, "predicted": classes[p].id, "count": c} for c, t, p in pairs[:limit]]


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def _predict_torch(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    model.eval()
    trues, preds = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        trues.extend(y.tolist())
    return trues, preds


def _predict_onnx(path: Path, loader: DataLoader) -> tuple[list[int], list[int]]:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    trues, preds = [], []
    for x, y in loader:
        logits = sess.run(["logits"], {"input": x.numpy()})[0]
        preds.extend(np.argmax(logits, axis=1).tolist())
        trues.extend(y.tolist())
    return trues, preds


def _run_epoch(model, loader, optimizer, loss_fn, device) -> None:
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()


def _set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for p in model.features.parameters():
        p.requires_grad = trainable


def export_onnx(model: nn.Module, path: Path) -> None:
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class TrainResult:
    meta: dict
    model_path: Path


def train(
    dataset_dir: Path,
    classes: list[Label],
    out_dir: Path,
    version: int,
    config: TrainConfig,
    progress: ProgressFn = no_progress,
) -> TrainResult:
    torch.manual_seed(config.seed)
    items = [
        (p, idx)
        for idx, label in enumerate(classes)
        for p in sorted((dataset_dir / label.id).glob("*.jpg"))
    ]
    train_items, val_items = stratified_split(items, 0.2, config.seed)
    if not train_items or not val_items:
        raise ValueError("not enough photos to train")

    device = _device(config.device)
    bs = config.batch_size
    train_loader = DataLoader(
        ImageList(train_items, train_transform()), batch_size=bs, shuffle=True,
        num_workers=config.num_workers, drop_last=len(train_items) > bs,
    )
    val_loader = DataLoader(ImageList(val_items, eval_transform()), batch_size=bs, num_workers=config.num_workers)

    model = build_model(len(classes), config.pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    total = config.epochs_head + config.epochs_full
    best_acc, best_state, epoch = -1.0, None, 0
    # Phase 1: head only (backbone frozen). Phase 2: everything, lower LR.
    for backbone_trainable, n_epochs, lr in [
        (False, config.epochs_head, config.lr_head),
        (True, config.epochs_full, config.lr_full),
    ]:
        _set_backbone_trainable(model, backbone_trainable)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_epochs):
            epoch += 1
            _run_epoch(model, train_loader, optimizer, loss_fn, device)
            acc = evaluate_predictions(*_predict_torch(model, val_loader, device), len(classes)).accuracy
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            progress(epoch, total, f"epoch {epoch}/{total} val_accuracy={acc:.3f}")

    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    torch_eval = evaluate_predictions(*_predict_torch(model, val_loader, torch.device("cpu")), len(classes))

    model_path = out_dir / "model.onnx"
    export_onnx(model, model_path)
    onnx_eval = evaluate_predictions(*_predict_onnx(model_path, val_loader), len(classes))
    if abs(onnx_eval.accuracy - torch_eval.accuracy) > MAX_ONNX_ACCURACY_GAP:
        raise ExportMismatch(
            f"ONNX accuracy {onnx_eval.accuracy:.3f} differs from PyTorch {torch_eval.accuracy:.3f}"
        )

    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "classes": [{"id": c.id, "entity_id": c.entity_id, "friendly_name": c.friendly_name} for c in classes],
        "input_size": INPUT_SIZE,
        "normalization": {"mean": MEAN, "std": STD},
        "preprocessing": "center_square_crop_then_resize_rgb",
        "output": "logits",
        "val_accuracy": onnx_eval.accuracy,
        "per_class_accuracy": {c.id: a for c, a in zip(classes, onnx_eval.per_class)},
        "confusion_matrix": onnx_eval.confusion,
        "confusions": top_confusions(onnx_eval.confusion, classes),
        "train_count": len(train_items),
        "val_count": len(val_items),
        "seed": config.seed,
        "epochs": total,
        "sha256": _sha256(model_path),
        "size_bytes": model_path.stat().st_size,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return TrainResult(meta, model_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trainer.py -v`
Expected: 3 passed on CPU, plus the GPU test passing on the desktop (it downloads ImageNet weights on first run). If the GPU test fails with a CUDA "no kernel image" error, the installed torch wheel lacks sm_120 support; change the index in `pyproject.toml` to the newest `cu12x`/`cu13x` index PyTorch publishes and re-run `uv sync`.

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/trainer.py server/tests/test_trainer.py
git commit -m "feat(server): MobileNetV3 fine-tuning with verified ONNX export"
```

---

### Task 8: Training pipeline glue

**Files:**
- Create: `server/pointremote/pipeline.py`
- Test: `server/tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Settings`, `LabelStore`, `PhotoStore`, `ModelRegistry`, `trainer.train`, `ProgressFn`.
- Produces: `NotEnoughPhotos(Exception)` (args[0] is the blocker list); `training_blockers(settings, labels, photos) -> list[str]` returning `"add at least one device label"` when only `unknown` exists, and `"<id>: <n>/<min> photos"` for each short label (in label order); `run_training(settings, labels, photos, registry, progress) -> dict` (published meta). Calls `trainer.train` through the module (`from . import trainer`) so tests can monkeypatch it.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_pipeline.py`:
```python
import dataclasses
import json

import pytest

from pointremote import pipeline, trainer
from pointremote.labels import LabelStore
from pointremote.photos import PhotoStore
from pointremote.progress import no_progress
from pointremote.registry import ModelRegistry


@pytest.fixture
def stores(settings):
    return LabelStore(settings.data_dir), PhotoStore(settings.data_dir), ModelRegistry(settings.data_dir)


def test_blocks_without_device_label(settings, stores):
    labels, photos, _ = stores
    assert "add at least one device label" in pipeline.training_blockers(settings, labels, photos)


def test_reports_short_labels(settings, stores, make_jpeg):
    settings = dataclasses.replace(settings, min_photos=2)
    labels, photos, _ = stores
    labels.create("light.kitchen", "Kitchen")
    photos.save("unknown", make_jpeg())
    photos.save("unknown", make_jpeg())
    photos.save("light_kitchen", make_jpeg())
    assert pipeline.training_blockers(settings, labels, photos) == ["light_kitchen: 1/2 photos"]


def _ready(labels, photos, make_jpeg):
    labels.create("light.kitchen", "Kitchen")
    photos.save("unknown", make_jpeg())
    photos.save("light_kitchen", make_jpeg())


def test_run_training_publishes(settings, stores, make_jpeg, monkeypatch):
    labels, photos, registry = stores
    _ready(labels, photos, make_jpeg)
    calls = {}

    def fake_train(dataset_dir, classes, out_dir, version, config, progress):
        calls["classes"] = [c.id for c in classes]
        calls["dataset_dir"] = dataset_dir
        meta = {"version": version, "val_accuracy": 1.0}
        (out_dir / "model.onnx").write_bytes(b"onnx")
        (out_dir / "meta.json").write_text(json.dumps(meta))
        return trainer.TrainResult(meta, out_dir / "model.onnx")

    monkeypatch.setattr(trainer, "train", fake_train)
    meta = pipeline.run_training(settings, labels, photos, registry, no_progress)
    assert meta["version"] == 1
    assert registry.latest_meta()["version"] == 1
    assert calls["classes"] == ["unknown", "light_kitchen"]
    assert calls["dataset_dir"] == photos.root


def test_run_training_cleans_up_on_failure(settings, stores, make_jpeg, monkeypatch):
    labels, photos, registry = stores
    _ready(labels, photos, make_jpeg)

    def failing_train(*args, **kwargs):
        raise trainer.ExportMismatch("nope")

    monkeypatch.setattr(trainer, "train", failing_train)
    with pytest.raises(trainer.ExportMismatch):
        pipeline.run_training(settings, labels, photos, registry, no_progress)
    assert registry.latest_meta() is None
    assert not list(registry.root.glob(".staging-*"))


def test_run_training_refuses_when_blocked(settings, stores):
    labels, photos, registry = stores
    with pytest.raises(pipeline.NotEnoughPhotos):
        pipeline.run_training(settings, labels, photos, registry, no_progress)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name 'pipeline'`

- [ ] **Step 3: Implement**

`server/pointremote/pipeline.py`:
```python
from __future__ import annotations

import shutil

from . import trainer
from .config import Settings
from .labels import LabelStore
from .photos import PhotoStore
from .progress import ProgressFn
from .registry import ModelRegistry


class NotEnoughPhotos(Exception):
    pass


def training_blockers(settings: Settings, labels: LabelStore, photos: PhotoStore) -> list[str]:
    classes = labels.all()
    blockers = []
    if len(classes) < 2:
        blockers.append("add at least one device label")
    for label in classes:
        n = photos.count(label.id)
        if n < settings.min_photos:
            blockers.append(f"{label.id}: {n}/{settings.min_photos} photos")
    return blockers


def run_training(
    settings: Settings,
    labels: LabelStore,
    photos: PhotoStore,
    registry: ModelRegistry,
    progress: ProgressFn,
) -> dict:
    blockers = training_blockers(settings, labels, photos)
    if blockers:
        raise NotEnoughPhotos(blockers)
    classes = labels.all()
    version = registry.next_version()
    staging = registry.staging_dir(version)
    try:
        result = trainer.train(photos.root, classes, staging, version, settings.train, progress)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    registry.publish(version, staging)
    return result.meta
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/pipeline.py server/tests/test_pipeline.py
git commit -m "feat(server): training pipeline with readiness checks"
```

---

### Task 9: Auth and JSON API

**Files:**
- Create: `server/pointremote/auth.py`, `server/pointremote/api.py`
- Modify: `server/tests/conftest.py` (add client fixtures)
- Test: `server/tests/test_api.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `auth.py`: `COOKIE_NAME = "pr_key"`, `key_matches(expected: str, given: str | None) -> bool`, `make_key_checker(expected: str)` (FastAPI dependency reading header `X-Api-Key`, 401 on mismatch).
- Produces: `api.create_app(settings: Settings, train_fn: Callable[[ProgressFn], dict] | None = None) -> FastAPI`, exposing `app.state.runner: TrainingRunner`. Endpoints exactly as in the spec table; label JSON is `{"id", "entity_id", "friendly_name", "photo_count"}`. `POST /train` returns 409 if running, 400 `{"detail": {"blockers": [...]}}` if blocked, else 202 with the status dict. The UI router is included in Task 10.

- [ ] **Step 1: Add client fixtures**

Append to `server/tests/conftest.py`:
```python
from fastapi.testclient import TestClient

from pointremote.api import create_app

AUTH = {"X-Api-Key": "test-key"}


@pytest.fixture
def make_client(settings):
    def make(train_fn=None):
        return TestClient(create_app(settings, train_fn=train_fn))

    return make


@pytest.fixture
def client(make_client):
    return make_client(train_fn=lambda progress: {"version": 1, "val_accuracy": 1.0})
```

- [ ] **Step 2: Write the failing tests**

`server/tests/test_api.py`:
```python
import threading

from conftest import AUTH
from test_registry import publish_fake

from pointremote.registry import ModelRegistry


def _jpeg_file(make_jpeg):
    return {"file": ("a.jpg", make_jpeg(), "image/jpeg")}


def _ready(client, make_jpeg):
    assert client.post("/labels", json={"entity_id": "light.kitchen", "friendly_name": "Kitchen"}, headers=AUTH).status_code == 201
    for label in ("unknown", "light_kitchen"):
        assert client.post(f"/photos/{label}", files=_jpeg_file(make_jpeg), headers=AUTH).status_code == 201


def test_requires_key(client):
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"X-Api-Key": "wrong"}).status_code == 401


def test_health(client):
    assert client.get("/health", headers=AUTH).json() == {"ok": True, "training": False, "latest_version": None}


def test_labels_crud(client):
    assert client.get("/labels", headers=AUTH).json() == [
        {"id": "unknown", "entity_id": None, "friendly_name": "Unknown", "photo_count": 0}
    ]
    r = client.post("/labels", json={"entity_id": "light.kitchen", "friendly_name": "Kitchen"}, headers=AUTH)
    assert r.status_code == 201
    assert r.json() == {"id": "light_kitchen", "entity_id": "light.kitchen", "friendly_name": "Kitchen", "photo_count": 0}
    assert client.post("/labels", json={"entity_id": "light.kitchen", "friendly_name": "x"}, headers=AUTH).status_code == 409
    assert client.post("/labels", json={"entity_id": "bad", "friendly_name": "x"}, headers=AUTH).status_code == 422
    assert client.delete("/labels/unknown", headers=AUTH).status_code == 400
    assert client.delete("/labels/light_kitchen", headers=AUTH).status_code == 204
    assert client.delete("/labels/light_kitchen", headers=AUTH).status_code == 404


def test_upload_photo(client, make_jpeg):
    r = client.post("/photos/unknown", files=_jpeg_file(make_jpeg), headers=AUTH)
    assert r.status_code == 201
    assert r.json()["count"] == 1
    assert client.get("/labels", headers=AUTH).json()[0]["photo_count"] == 1
    assert client.post("/photos/nope", files=_jpeg_file(make_jpeg), headers=AUTH).status_code == 404
    bad = {"file": ("a.jpg", b"garbage", "image/jpeg")}
    assert client.post("/photos/unknown", files=bad, headers=AUTH).status_code == 400


def test_delete_label_removes_photos(client, make_jpeg):
    _ready(client, make_jpeg)
    client.delete("/labels/light_kitchen", headers=AUTH)
    client.post("/labels", json={"entity_id": "light.kitchen", "friendly_name": "Kitchen"}, headers=AUTH)
    counts = {l["id"]: l["photo_count"] for l in client.get("/labels", headers=AUTH).json()}
    assert counts["light_kitchen"] == 0


def test_train_blocked(client):
    r = client.post("/train", headers=AUTH)
    assert r.status_code == 400
    assert "add at least one device label" in r.json()["detail"]["blockers"]


def test_train_runs(client, make_jpeg):
    _ready(client, make_jpeg)
    r = client.post("/train", headers=AUTH)
    assert r.status_code == 202
    assert r.json()["state"] == "running"
    client.app.state.runner.join(5)
    s = client.get("/train/status", headers=AUTH).json()
    assert s["state"] == "done" and s["version"] == 1


def test_train_conflict(make_client, make_jpeg):
    gate = threading.Event()

    def slow(progress):
        gate.wait(5)
        return {"version": 1}

    client = make_client(train_fn=slow)
    _ready(client, make_jpeg)
    assert client.post("/train", headers=AUTH).status_code == 202
    assert client.post("/train", headers=AUTH).status_code == 409
    assert client.get("/health", headers=AUTH).json()["training"] is True
    gate.set()
    client.app.state.runner.join(5)


def test_model_endpoints(client, settings):
    assert client.get("/model/latest/meta", headers=AUTH).status_code == 404
    assert client.get("/model/latest/file", headers=AUTH).status_code == 404
    publish_fake(ModelRegistry(settings.data_dir), val_accuracy=0.9)
    assert client.get("/model/latest/meta", headers=AUTH).json() == {"version": 1, "val_accuracy": 0.9}
    f = client.get("/model/latest/file", headers=AUTH)
    assert f.status_code == 200 and f.content == b"onnx"
    assert client.get("/health", headers=AUTH).json()["latest_version"] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pointremote.api'`

- [ ] **Step 4: Implement auth**

`server/pointremote/auth.py`:
```python
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

COOKIE_NAME = "pr_key"


def key_matches(expected: str, given: str | None) -> bool:
    return given is not None and secrets.compare_digest(given.encode(), expected.encode())


def make_key_checker(expected: str):
    def check(x_api_key: str | None = Header(default=None)) -> None:
        if not key_matches(expected, x_api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    return check
```

- [ ] **Step 5: Implement the API**

`server/pointremote/api.py`:
```python
from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .auth import make_key_checker
from .config import Settings
from .jobs import AlreadyRunning, TrainingRunner
from .labels import UNKNOWN_ID, InvalidEntityId, Label, LabelExists, LabelNotFound, LabelStore
from .photos import InvalidImage, PhotoStore
from .pipeline import run_training, training_blockers
from .progress import ProgressFn
from .registry import ModelRegistry


class NewLabel(BaseModel):
    entity_id: str
    friendly_name: str


def create_app(settings: Settings, train_fn: Callable[[ProgressFn], dict] | None = None) -> FastAPI:
    labels = LabelStore(settings.data_dir)
    photos = PhotoStore(settings.data_dir)
    registry = ModelRegistry(settings.data_dir)
    if train_fn is None:
        def train_fn(progress: ProgressFn) -> dict:
            return run_training(settings, labels, photos, registry, progress)
    runner = TrainingRunner(train_fn)

    app = FastAPI(title="PointRemote training server")
    app.state.runner = runner
    api = APIRouter(dependencies=[Depends(make_key_checker(settings.api_key))])

    def label_json(label: Label) -> dict:
        return {
            "id": label.id,
            "entity_id": label.entity_id,
            "friendly_name": label.friendly_name,
            "photo_count": photos.count(label.id),
        }

    def require_label(label_id: str) -> Label:
        try:
            return labels.get(label_id)
        except LabelNotFound:
            raise HTTPException(status_code=404, detail="label not found") from None

    @api.get("/health")
    def health() -> dict:
        meta = registry.latest_meta()
        return {"ok": True, "training": runner.running, "latest_version": meta["version"] if meta else None}

    @api.get("/labels")
    def list_labels() -> list[dict]:
        return [label_json(l) for l in labels.all()]

    @api.post("/labels", status_code=201)
    def create_label(body: NewLabel) -> dict:
        try:
            label = labels.create(body.entity_id, body.friendly_name)
        except InvalidEntityId:
            raise HTTPException(status_code=422, detail="invalid entity_id") from None
        except LabelExists:
            raise HTTPException(status_code=409, detail="label already exists") from None
        return label_json(label)

    @api.delete("/labels/{label_id}", status_code=204)
    def delete_label(label_id: str) -> Response:
        if label_id == UNKNOWN_ID:
            raise HTTPException(status_code=400, detail="the unknown label cannot be deleted")
        try:
            labels.delete(label_id)
        except LabelNotFound:
            raise HTTPException(status_code=404, detail="label not found") from None
        photos.delete_label(label_id)
        return Response(status_code=204)

    @api.post("/photos/{label_id}", status_code=201)
    async def upload_photo(label_id: str, file: UploadFile) -> dict:
        label = require_label(label_id)
        try:
            name = photos.save(label.id, await file.read())
        except InvalidImage as e:
            raise HTTPException(status_code=400, detail=str(e)) from None
        return {"name": name, "count": photos.count(label.id)}

    @api.post("/train", status_code=202)
    def start_training() -> dict:
        if runner.running:
            raise HTTPException(status_code=409, detail="training already running")
        blockers = training_blockers(settings, labels, photos)
        if blockers:
            raise HTTPException(status_code=400, detail={"blockers": blockers})
        try:
            runner.start()
        except AlreadyRunning:
            raise HTTPException(status_code=409, detail="training already running") from None
        return runner.status()

    @api.get("/train/status")
    def training_status() -> dict:
        return runner.status()

    @api.get("/model/latest/meta")
    def latest_meta() -> dict:
        meta = registry.latest_meta()
        if meta is None:
            raise HTTPException(status_code=404, detail="no model trained yet")
        return meta

    @api.get("/model/latest/file")
    def latest_file() -> FileResponse:
        d = registry.latest_dir()
        if d is None:
            raise HTTPException(status_code=404, detail="no model trained yet")
        return FileResponse(d / "model.onnx", media_type="application/octet-stream", filename="model.onnx")

    app.include_router(api)
    return app
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: all passed. (Tests import `conftest` and `test_registry` directly; pytest's default rootdir-based import mode puts `tests/` on `sys.path`, so this works.)

- [ ] **Step 7: Commit**

```bash
git add server/pointremote/auth.py server/pointremote/api.py server/tests/conftest.py server/tests/test_api.py
git commit -m "feat(server): authenticated JSON API for labels, photos, training, models"
```

---

### Task 10: Web UI

**Files:**
- Create: `server/pointremote/ui.py`
- Modify: `server/pointremote/api.py` (include UI router)
- Test: `server/tests/test_ui.py`

**Interfaces:**
- Consumes: `COOKIE_NAME`, `key_matches` (Task 9), stores and registry.
- Produces: `build_ui_router(settings, labels, photos, registry) -> APIRouter` with prefix `/ui`: `GET /ui/login`, `POST /ui/login` (form field `key`), `GET /ui`, `GET /ui/labels/{label_id}`, `GET /ui/photos/{label_id}/{name}`, `POST /ui/photos/{label_id}/{name}/delete`. Unauthenticated page requests → 303 to `/ui/login`; unauthenticated photo request → 401.

- [ ] **Step 1: Write the failing tests**

`server/tests/test_ui.py`:
```python
import pytest
from conftest import AUTH
from test_registry import publish_fake

from pointremote.registry import ModelRegistry


@pytest.fixture
def ui(client, make_jpeg):
    client.post("/labels", json={"entity_id": "light.kitchen", "friendly_name": "Kitchen lights"}, headers=AUTH)
    name = client.post("/photos/light_kitchen", files={"file": ("a.jpg", make_jpeg(), "image/jpeg")}, headers=AUTH).json()["name"]
    client.cookies.set("pr_key", "test-key")
    return client, name


def test_redirects_to_login(client):
    r = client.get("/ui", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/ui/login"


def test_login(client):
    assert client.post("/ui/login", data={"key": "wrong"}, follow_redirects=False).status_code == 401
    r = client.post("/ui/login", data={"key": "test-key"}, follow_redirects=False)
    assert r.status_code == 303
    assert r.cookies.get("pr_key") == "test-key"


def test_index_lists_labels_and_history(ui, settings):
    client, _ = ui
    publish_fake(ModelRegistry(settings.data_dir), val_accuracy=0.875, classes=[{"id": "unknown"}, {"id": "light_kitchen"}],
                 confusion_matrix=[[3, 1], [0, 4]], created_at="2026-11-09T10:00:00+00:00", train_count=40, val_count=8)
    html = client.get("/ui").text
    assert "Kitchen lights" in html
    assert "light.kitchen" in html
    assert "v1" in html and "87.5%" in html and "40/8" in html


def test_label_page_and_photo(ui, make_jpeg):
    client, name = ui
    page = client.get("/ui/labels/light_kitchen").text
    assert f"/ui/photos/light_kitchen/{name}" in page
    assert client.get(f"/ui/photos/light_kitchen/{name}").status_code == 200
    assert client.get("/ui/labels/nope").status_code == 404
    assert client.get("/ui/photos/light_kitchen/../../labels.json").status_code == 404


def test_delete_photo(ui):
    client, name = ui
    r = client.post(f"/ui/photos/light_kitchen/{name}/delete", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/ui/labels/light_kitchen"
    counts = {l["id"]: l["photo_count"] for l in client.get("/labels", headers=AUTH).json()}
    assert counts["light_kitchen"] == 0


def test_photo_requires_cookie(ui):
    client, name = ui
    client.cookies.clear()
    assert client.get(f"/ui/photos/light_kitchen/{name}").status_code == 401
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ui.py -v`
Expected: FAIL (`/ui` returns 404 because no UI router exists)

- [ ] **Step 3: Implement the UI**

`server/pointremote/ui.py`:
```python
from __future__ import annotations

from html import escape

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response

from .auth import COOKIE_NAME, key_matches
from .config import Settings
from .labels import LabelNotFound, LabelStore
from .photos import PhotoNotFound, PhotoStore
from .registry import ModelRegistry

_STYLE = (
    "body{font-family:system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem}"
    "table{border-collapse:collapse;margin-bottom:1rem}td,th{border:1px solid #ccc;padding:.3rem .6rem;text-align:left}"
    ".grid{display:flex;flex-wrap:wrap;gap:.5rem}.grid figure{margin:0}"
    ".grid img{width:160px;height:120px;object-fit:cover;display:block}"
)
_LOGIN_FORM = (
    "<h1>PointRemote</h1><form method='post' action='/ui/login'>"
    "<input type='password' name='key' placeholder='API key' autofocus> <button>Log in</button></form>"
)


def _page(title: str, body: str, status: int = 200) -> HTMLResponse:
    return HTMLResponse(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head><body>{body}</body></html>",
        status_code=status,
    )


def _history_row(meta: dict) -> str:
    ids = [c.get("id", "?") for c in meta.get("classes", [])]
    matrix = meta.get("confusion_matrix") or []
    if matrix and len(ids) == len(matrix):
        head = "".join(f"<th>{escape(i)}</th>" for i in ids)
        rows = "".join(
            f"<tr><th>{escape(ids[r])}</th>" + "".join(f"<td>{n}</td>" for n in row) + "</tr>"
            for r, row in enumerate(matrix)
        )
        matrix_html = f"<details><summary>show</summary><table><tr><th>true → predicted</th>{head}</tr>{rows}</table></details>"
    else:
        matrix_html = "—"
    acc = meta.get("val_accuracy")
    acc_s = f"{acc:.1%}" if isinstance(acc, (int, float)) else "—"
    return (
        f"<tr><td>v{meta.get('version')}</td><td>{escape(str(meta.get('created_at', '')))}</td>"
        f"<td>{acc_s}</td><td>{meta.get('train_count', '?')}/{meta.get('val_count', '?')}</td>"
        f"<td>{matrix_html}</td></tr>"
    )


def build_ui_router(settings: Settings, labels: LabelStore, photos: PhotoStore, registry: ModelRegistry) -> APIRouter:
    router = APIRouter(prefix="/ui")

    def authed(request: Request) -> bool:
        return key_matches(settings.api_key, request.cookies.get(COOKIE_NAME))

    def to_login() -> RedirectResponse:
        return RedirectResponse("/ui/login", status_code=303)

    @router.get("/login")
    def login_form() -> HTMLResponse:
        return _page("Log in", _LOGIN_FORM)

    @router.post("/login")
    def login(key: str = Form(...)) -> Response:
        if not key_matches(settings.api_key, key):
            return _page("Log in", "<p>Wrong key.</p>" + _LOGIN_FORM, status=401)
        resp = RedirectResponse("/ui", status_code=303)
        resp.set_cookie(COOKIE_NAME, key, httponly=True, samesite="strict")
        return resp

    @router.get("")
    def index(request: Request) -> Response:
        if not authed(request):
            return to_login()
        label_rows = "".join(
            f"<tr><td><a href='/ui/labels/{escape(l.id)}'>{escape(l.friendly_name)}</a></td>"
            f"<td>{escape(l.entity_id or '—')}</td><td>{photos.count(l.id)}</td></tr>"
            for l in labels.all()
        )
        history = "".join(_history_row(m) for m in registry.history()) or "<tr><td colspan='5'>No models trained yet.</td></tr>"
        body = (
            "<h1>PointRemote</h1><h2>Labels</h2>"
            f"<table><tr><th>Label</th><th>Entity</th><th>Photos</th></tr>{label_rows}</table>"
            "<h2>Training history</h2><table><tr><th>Version</th><th>Created</th><th>Val accuracy</th>"
            f"<th>Train/val</th><th>Confusion matrix</th></tr>{history}</table>"
        )
        return _page("PointRemote", body)

    @router.get("/labels/{label_id}")
    def label_page(label_id: str, request: Request) -> Response:
        if not authed(request):
            return to_login()
        try:
            label = labels.get(label_id)
        except LabelNotFound:
            return _page("Not found", "<p>No such label.</p>", status=404)
        names = photos.names(label.id)
        figures = "".join(
            f"<figure><img loading='lazy' src='/ui/photos/{label.id}/{n}' alt=''>"
            f"<form method='post' action='/ui/photos/{label.id}/{n}/delete'><button>Delete</button></form></figure>"
            for n in names
        )
        body = (
            f"<p><a href='/ui'>← back</a></p><h1>{escape(label.friendly_name)} ({len(names)} photos)</h1>"
            f"<div class='grid'>{figures or '<p>No photos yet.</p>'}</div>"
        )
        return _page(label.friendly_name, body)

    @router.get("/photos/{label_id}/{name}")
    def photo(label_id: str, name: str, request: Request) -> Response:
        if not authed(request):
            return Response(status_code=401)
        try:
            labels.get(label_id)
            return FileResponse(photos.path(label_id, name), media_type="image/jpeg")
        except (LabelNotFound, PhotoNotFound):
            return Response(status_code=404)

    @router.post("/photos/{label_id}/{name}/delete")
    def delete_photo(label_id: str, name: str, request: Request) -> Response:
        if not authed(request):
            return to_login()
        try:
            labels.get(label_id)
            photos.delete(label_id, name)
        except (LabelNotFound, PhotoNotFound):
            return Response(status_code=404)
        return RedirectResponse(f"/ui/labels/{label_id}", status_code=303)

    return router
```

- [ ] **Step 4: Wire the router into the app**

In `server/pointremote/api.py`, add the import next to the others:
```python
from .ui import build_ui_router
```
and replace the last two lines of `create_app`:
```python
    app.include_router(api)
    return app
```
with:
```python
    app.include_router(api)
    app.include_router(build_ui_router(settings, labels, photos, registry))
    return app
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -v`
Expected: all passed (the traversal URL in `test_label_page_and_photo` is normalised by the client to a non-matching path and returns 404).

- [ ] **Step 6: Commit**

```bash
git add server/pointremote/ui.py server/pointremote/api.py server/tests/test_ui.py
git commit -m "feat(server): dataset and training-history web UI"
```

---

### Task 11: Entry point, run script, README, and live check

**Files:**
- Create: `server/pointremote/main.py`, `server/run.sh`, `server/README.md`

**Interfaces:**
- Consumes: `create_app`, `Settings.from_env`.
- Produces: `pointremote.main:app` for uvicorn; `./run.sh` serving on `0.0.0.0:8765`.

- [ ] **Step 1: Create the entry point and run script**

`server/pointremote/main.py`:
```python
from .api import create_app
from .config import Settings

app = create_app(Settings.from_env())
```

`server/run.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi
exec uv run uvicorn pointremote.main:app --host 0.0.0.0 --port "${POINTREMOTE_PORT:-8765}"
```

Run: `chmod +x server/run.sh`

`server/README.md`:
````markdown
# PointRemote training server

Stores labelled photos from the phone, trains the recogniser on the GPU, and serves the model.

## Setup

```bash
cd server
uv sync
echo "POINTREMOTE_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')" > .env
./run.sh
```

- API: `http://<pc-ip>:8765` with header `X-Api-Key: <key from .env>`
- Web UI: `http://<pc-ip>:8765/ui` (log in with the same key)
- Data: `server/data/` (labels.json, dataset/, models/)

## Tests

```bash
uv run pytest            # GPU test runs only when CUDA is available
```
````

- [ ] **Step 2: Start the server and smoke-test it**

Run (in one terminal, or in the background): `cd server && echo "POINTREMOTE_API_KEY=devkey" > .env && ./run.sh`

Then:
```bash
curl -s -H 'X-Api-Key: devkey' localhost:8765/health
curl -s -H 'X-Api-Key: devkey' -H 'Content-Type: application/json' \
  -d '{"entity_id":"light.test","friendly_name":"Test"}' localhost:8765/labels
curl -s -o /dev/null -w '%{http_code}\n' localhost:8765/health
```
Expected: `{"ok":true,"training":false,"latest_version":null}`, then the label JSON, then `401`.

- [ ] **Step 3: Real GPU training run**

Generate 30 synthetic photos per class and train with the real config:
```bash
tmp=$(mktemp --suffix=.jpg)
for label in unknown light_test; do
  for i in $(seq 1 30); do
    uv run python -c "import io,random,sys;from PIL import Image;b=io.BytesIO();Image.new('RGB',(320,240),tuple(random.randint(0,255) for _ in range(3)) if sys.argv[1]=='unknown' else (200,40,40)).save(b,'JPEG');sys.stdout.buffer.write(b.getvalue())" "$label" > "$tmp"
    curl -s -o /dev/null -H 'X-Api-Key: devkey' -F "file=@$tmp;type=image/jpeg" "localhost:8765/photos/$label"
  done
done
curl -s -X POST -H 'X-Api-Key: devkey' localhost:8765/train
watch -n 2 "curl -s -H 'X-Api-Key: devkey' localhost:8765/train/status"
```
Expected: state reaches `done` with `version: 1` within a few minutes; `nvidia-smi` shows the python process using the GPU during training; `/ui` shows v1 in the history. Afterwards delete the test data: stop the server and `rm -rf server/data`.

- [ ] **Step 4: LAN access**

From the phone's browser, open `http://<pc-ip>:8765/ui`. If it doesn't load, the PC firewall is blocking port 8765. **Ask the user before changing any firewall settings**; they will decide whether to open the port.

- [ ] **Step 5: Commit**

```bash
git add server/pointremote/main.py server/run.sh server/README.md
git commit -m "feat(server): uvicorn entry point, run script and README"
```

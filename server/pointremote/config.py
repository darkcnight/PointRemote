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

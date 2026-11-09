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

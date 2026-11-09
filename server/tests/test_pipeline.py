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

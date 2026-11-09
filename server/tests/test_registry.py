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

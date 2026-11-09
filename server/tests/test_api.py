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
    # The fake job can finish before the response is built.
    assert r.json()["state"] in ("running", "done")
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

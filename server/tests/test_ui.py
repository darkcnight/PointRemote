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

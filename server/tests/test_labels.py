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

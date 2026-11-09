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

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


from fastapi.testclient import TestClient  # noqa: E402

from pointremote.api import create_app  # noqa: E402

AUTH = {"X-Api-Key": "test-key"}


@pytest.fixture
def make_client(settings):
    def make(train_fn=None):
        return TestClient(create_app(settings, train_fn=train_fn))

    return make


@pytest.fixture
def client(make_client):
    return make_client(train_fn=lambda progress: {"version": 1, "val_accuracy": 1.0})

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

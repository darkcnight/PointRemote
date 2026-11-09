import json
import sys

import pytest
from urllib.parse import parse_qs, urlparse

from pointremote.desktop.app import ago
from pointremote.desktop.config import load_or_create
from pointremote.desktop.net import lan_ip, server_url, setup_uri


def test_first_run_creates_settings_with_a_key(tmp_path):
    cfg = load_or_create(tmp_path / "conf", tmp_path / "data")
    saved = json.loads((tmp_path / "conf" / "settings.json").read_text())
    assert saved["api_key"] == cfg.api_key and len(cfg.api_key) >= 30
    assert cfg.data_dir == tmp_path / "data" / "server-data" and cfg.port == 8765
    assert load_or_create(tmp_path / "conf", tmp_path / "data") == cfg


def test_dev_mode_uses_the_repo_env(tmp_path):
    server = tmp_path / "server"
    server.mkdir()
    (server / ".env").write_text("# comment\nPOINTREMOTE_API_KEY='abc'\n")
    cfg = load_or_create(tmp_path / "conf", tmp_path / "data", dev_server_dir=server)
    assert (cfg.api_key, cfg.data_dir, cfg.port) == ("abc", (server / "data").resolve(), 8765)
    assert not (tmp_path / "conf").exists()


def test_dev_mode_without_env_falls_back_to_settings(tmp_path):
    cfg = load_or_create(tmp_path / "conf", tmp_path / "data", dev_server_dir=tmp_path)
    assert (tmp_path / "conf" / "settings.json").is_file() and cfg.api_key


def test_setup_uri_round_trips():
    uri = setup_uri(server_url("192.168.1.20", 8765), "a-b_c+d/e")
    parsed = urlparse(uri)
    assert (parsed.scheme, parsed.netloc, parsed.path) == ("pointremote", "setup", "")
    q = parse_qs(parsed.query)
    assert q == {"url": ["http://192.168.1.20:8765"], "key": ["a-b_c+d/e"]}


def test_lan_ip_is_an_address():
    assert lan_ip().count(".") == 3


def test_ago():
    assert ago(None) == "Phone hasn't connected yet"
    assert ago(5) == "Phone connected just now"
    assert ago(60) == "Phone last connected 1 minute ago"
    assert ago(600) == "Phone last connected 10 minutes ago"
    assert ago(7200) == "Phone last connected 2 hours ago"


def test_desktop_package_does_not_import_torch():
    import subprocess
    import sys

    code = "import sys, pointremote.desktop.app as a; print('torch' in sys.modules or 'fastapi' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_settings_file_is_private(tmp_path):
    import os
    import stat

    load_or_create(tmp_path / "conf", tmp_path / "data")
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "conf" / "settings.json").stat().st_mode) == 0o600


def test_instance_endpoint_identifies_the_server(client, monkeypatch):
    monkeypatch.setenv("POINTREMOTE_INSTANCE", "abc123")
    assert client.get("/instance").json() == {"instance": "abc123"}


@pytest.mark.skipif(sys.platform == "win32", reason="Linux app menu only")
def test_menu_entry_points_at_the_appimage(tmp_path, monkeypatch):
    from pointremote.desktop import linux_menu

    monkeypatch.setenv("APPIMAGE", "/home/me/Apps/PointRemote-x86_64.AppImage")
    monkeypatch.setattr(linux_menu, "data_dir", lambda: tmp_path / "data")
    linux_menu.ensure_menu_entry(tmp_path / "apps")
    text = (tmp_path / "apps" / "pointremote.desktop").read_text()
    assert 'Exec="/home/me/Apps/PointRemote-x86_64.AppImage"' in text and linux_menu.MARKER in text


@pytest.mark.skipif(sys.platform == "win32", reason="Linux app menu only")
def test_menu_entry_leaves_a_hand_made_one_alone(tmp_path, monkeypatch):
    from pointremote.desktop import linux_menu

    apps = tmp_path / "apps"
    apps.mkdir()
    (apps / "pointremote.desktop").write_text("[Desktop Entry]\nExec=uv run pointremote-app\n")
    monkeypatch.setattr(linux_menu, "data_dir", lambda: tmp_path / "data")
    linux_menu.ensure_menu_entry(apps)
    assert "uv run" in (apps / "pointremote.desktop").read_text()

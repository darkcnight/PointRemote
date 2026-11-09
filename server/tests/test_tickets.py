from pointremote.tickets import LoginTickets
from conftest import AUTH


def test_ticket_is_single_use():
    t = LoginTickets()
    ticket = t.issue()
    assert t.redeem(ticket)
    assert not t.redeem(ticket)
    assert not t.redeem("made-up")


def test_ticket_expires():
    now = [0.0]
    t = LoginTickets(ttl_s=60, clock=lambda: now[0])
    ticket = t.issue()
    now[0] = 61
    assert not t.redeem(ticket)


def test_ticket_logs_into_ui(client):
    assert client.post("/ui-ticket").status_code == 401
    ticket = client.post("/ui-ticket", headers=AUTH).json()["ticket"]
    r = client.get(f"/ui/ticket/{ticket}", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/ui"
    assert r.cookies.get("pr_key") == "test-key"
    client.cookies.clear()
    assert client.get(f"/ui/ticket/{ticket}", follow_redirects=False).status_code == 401


def test_health_reports_when_phone_last_called(settings):
    from fastapi.testclient import TestClient
    from pointremote.api import create_app

    app = create_app(settings, train_fn=lambda p: {})
    local = TestClient(app, client=("127.0.0.1", 5000))
    assert local.get("/health", headers=AUTH).json()["phone_last_seen"] is None
    phone = TestClient(app, client=("192.168.1.20", 5000))
    assert phone.get("/health").status_code == 401
    assert local.get("/health", headers=AUTH).json()["phone_last_seen"] is None
    phone.get("/labels", headers=AUTH)
    assert local.get("/health", headers=AUTH).json()["phone_last_seen"] == 0

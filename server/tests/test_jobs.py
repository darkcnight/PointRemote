import threading

import pytest

from pointremote.jobs import AlreadyRunning, TrainingRunner


def test_idle_initially():
    r = TrainingRunner(lambda progress: {"version": 1})
    assert r.status() == {"state": "idle", "epoch": 0, "epochs": 0, "message": "", "version": None}
    assert not r.running


def test_runs_to_done():
    def fn(progress):
        progress(1, 2, "a")
        progress(2, 2, "b")
        return {"version": 3, "val_accuracy": 0.9}

    r = TrainingRunner(fn)
    r.start()
    r.join(5)
    s = r.status()
    assert s["state"] == "done"
    assert s["version"] == 3
    assert (s["epoch"], s["epochs"]) == (2, 2)
    assert s["message"] == "val_accuracy=0.900"


def test_failure_reported():
    def fn(progress):
        raise ValueError("boom")

    r = TrainingRunner(fn)
    r.start()
    r.join(5)
    assert r.status()["state"] == "failed"
    assert r.status()["message"] == "boom"


def test_cannot_start_twice_but_can_restart_after():
    gate = threading.Event()

    def fn(progress):
        gate.wait(5)
        return {"version": 1}

    r = TrainingRunner(fn)
    r.start()
    assert r.running
    with pytest.raises(AlreadyRunning):
        r.start()
    gate.set()
    r.join(5)
    r.start()
    r.join(5)
    assert r.status()["state"] == "done"

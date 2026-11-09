import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the stdin watchdog is POSIX-only")

# Starts the watchdog the way the desktop app runs the server, then forks a
# multiprocessing child like PyTorch's DataLoader workers do.
CHILD = textwrap.dedent("""
    import multiprocessing as mp
    from pointremote.serve import start_stdin_watchdog

    def work(q):
        q.put("ok")

    if __name__ == "__main__":
        start_stdin_watchdog()
        ctx = mp.get_context("fork")
        q = ctx.Queue()
        p = ctx.Process(target=work, args=(q,))
        p.start()
        print(q.get(timeout=10), flush=True)
        p.join(10)
""")


def test_forked_workers_start_while_watching_stdin():
    proc = subprocess.Popen([sys.executable, "-c", CHILD], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    assert line.strip() == "ok"


def test_exits_when_stdin_closes():
    code = "from pointremote.serve import start_stdin_watchdog; import time; start_stdin_watchdog(); time.sleep(30)"
    proc = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.PIPE)
    proc.stdin.close()
    assert proc.wait(timeout=10) == 0

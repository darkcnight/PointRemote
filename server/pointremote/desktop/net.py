from __future__ import annotations

import os
import socket
from urllib.parse import quote


def lan_ip() -> str:
    """The address other devices on the network reach this PC at (no packets are sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET address: picks the default route's interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def server_url(ip: str, port: int) -> str:
    return f"http://{ip}:{port}"


def setup_uri(url: str, key: str) -> str:
    """What the QR code holds. The phone app's Scan QR reads this."""
    return f"pointremote://setup?url={quote(url, safe='')}&key={quote(key, safe='')}"


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if os.name != "nt":
            # Same as the server: lets a just-closed server's leftover connections be ignored.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False

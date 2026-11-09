from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

COOKIE_NAME = "pr_key"


def key_matches(expected: str, given: str | None) -> bool:
    return given is not None and secrets.compare_digest(given.encode(), expected.encode())


def make_key_checker(expected: str):
    def check(x_api_key: str | None = Header(default=None)) -> None:
        if not key_matches(expected, x_api_key):
            raise HTTPException(status_code=401, detail="invalid api key")

    return check

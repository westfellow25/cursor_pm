"""Minimal HS256 JWT — pure Python, no native dependencies."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time


class InvalidTokenError(Exception):
    pass


class ExpiredSignatureError(InvalidTokenError):
    pass


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(s: str) -> bytes:
    pad = 4 - len(s) % 4
    if pad != 4:
        s += "=" * pad
    return base64.urlsafe_b64decode(s)


_HEADER = _b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())


def encode(payload: dict, secret: str) -> str:
    body = _b64encode(json.dumps(payload, default=str).encode())
    msg = f"{_HEADER}.{body}"
    sig = hmac.HMAC(secret.encode(), msg.encode(), hashlib.sha256).digest()
    return f"{msg}.{_b64encode(sig)}"


def decode(token: str, secret: str, algorithms: list[str] | None = None) -> dict:
    parts = token.split(".")
    if len(parts) != 3:
        raise InvalidTokenError("Malformed token")
    header_b64, body_b64, sig_b64 = parts
    msg = f"{header_b64}.{body_b64}"
    expected_sig = hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()
    actual_sig = _b64decode(sig_b64)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise InvalidTokenError("Invalid signature")
    payload = json.loads(_b64decode(body_b64))
    exp = payload.get("exp")
    if exp and float(exp) < time.time():
        raise ExpiredSignatureError("Token expired")
    return payload

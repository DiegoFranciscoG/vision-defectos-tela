"""API key authentication, rate limiting and security headers (OWASP API2, API4, API8)."""

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from pydantic import SecretStr
from starlette.types import ASGIApp, Message, Receive, Scope, Send

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Demo API key")

_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
_API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"
_DOCS_CSP = (
    "default-src 'none'; script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; img-src 'self' data: "
    "https://fastapi.tiangolo.com; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
)


def key_fingerprint(key: str) -> str:
    """Short, non-reversible id of a key for logs and rate-limit buckets."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


class ApiKeyVerifier:
    def __init__(self, keys: list[SecretStr]) -> None:
        self._keys = [key.get_secret_value().encode() for key in keys]

    def verify(self, candidate: str | None) -> str:
        """Return the key fingerprint or raise 401. Constant-time comparison for every key."""
        if candidate:
            encoded = candidate.encode()
            matched = False
            for key in self._keys:
                matched |= hmac.compare_digest(encoded, key)
            if matched:
                return key_fingerprint(candidate)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


class RateLimiter:
    """Sliding window per key, in memory (one instance on Render free; documented limit)."""

    def __init__(self, limit_per_minute: int, window_s: float = 60.0) -> None:
        self._limit = limit_per_minute
        self._window = window_s
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, bucket: str) -> None:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[bucket]
            while hits and now - hits[0] > self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                retry_after = max(1, int(self._window - (now - hits[0])) + 1)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(retry_after)},
                )
            hits.append(now)


def client_bucket(request: Request) -> str:
    return request.client.host if request.client else "unknown"


class SecurityHeadersMiddleware:
    """Pure ASGI middleware (works with streaming responses)."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        is_docs = scope["path"].startswith(_DOCS_PATHS)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                        (b"cross-origin-opener-policy", b"same-origin"),
                        (b"content-security-policy", (_DOCS_CSP if is_docs else _API_CSP).encode()),
                        (b"cache-control", b"no-store"),
                    ]
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


class BodySizeLimitMiddleware:
    """Reject requests whose declared or streamed body exceeds the limit (API4)."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await _reject_too_large(send)
            return
        received = 0

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLargeError:
            await _reject_too_large(send)


class _BodyTooLargeError(Exception):
    pass


async def _reject_too_large(send: Send) -> None:
    body = b'{"detail":"Request body too large"}'
    await send(
        {
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})

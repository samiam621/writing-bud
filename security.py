"""
security.py
===========
Everything that guards the API lives here: API-key auth and rate limiting.
api.py attaches these as FastAPI dependencies; no other file imports this.

Two guards:
    require_api_key - rejects any request whose X-API-Key header doesn't match
                      config.API_KEY. Fail-closed: no key configured on the
                      server means NOBODY gets in (503), not everybody.
    rate_limit      - per-client-IP sliding window. At most
                      config.RATE_LIMIT_REQUESTS requests per
                      config.RATE_LIMIT_WINDOW_SECONDS, else 429.

The limiter is in-memory, which is exactly right for a single uvicorn process
(what Render's free tier runs). If you ever scale to multiple processes or
machines, each one gets its own counter — swap this for a shared store
(e.g. Redis) at that point.
"""

import secrets
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from config import API_KEY, RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SECONDS


# ---------------------------------------------------------------------------
# API-key auth
# ---------------------------------------------------------------------------

def is_valid_key(candidate: str) -> bool:
    """
    True if `candidate` matches the configured key.

    secrets.compare_digest takes the same time whether the first or last
    character differs, so an attacker can't recover the key by timing
    responses (a plain `==` short-circuits on the first mismatch).
    """
    return bool(API_KEY) and secrets.compare_digest(candidate, API_KEY)


def require_api_key(x_api_key: str = Header(default="")):
    """
    FastAPI dependency: reject the request unless X-API-Key is correct.

    FastAPI turns the parameter name `x_api_key` into the header name
    "X-API-Key" automatically. Attach with Depends(require_api_key).
    """
    if not API_KEY:
        # Server-side misconfiguration, not the client's fault: 503, and stay
        # closed rather than falling open with no auth at all.
        raise HTTPException(
            status_code=503,
            detail="Server has no WRITING_BUDDY_API_KEY configured.",
        )
    if not is_valid_key(x_api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# ip -> deque of request timestamps (monotonic seconds) inside the window.
_hits: dict[str, deque] = defaultdict(deque)

# If the table ever collects this many distinct IPs (a scan/flood), prune the
# stale ones so the limiter itself can't be used to exhaust memory.
_MAX_TRACKED_IPS = 10_000


def _client_ip(request: Request) -> str:
    """
    Best-effort client IP. Behind Render's proxy the direct peer address is
    the proxy itself; the real client comes from X-Forwarded-For.

    We take the LAST entry, not the first. Proxies append the address they
    actually saw to the end of the header, so the last entry is the one
    written by the proxy directly in front of us — trustworthy. Earlier
    entries arrive from the outside world: an attacker can send a made-up
    X-Forwarded-For of their own, and trusting the first entry would let
    them hop rate-limit buckets with a fresh fake IP per request.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request):
    """
    FastAPI dependency: 429 if this IP has already made RATE_LIMIT_REQUESTS
    requests within the last RATE_LIMIT_WINDOW_SECONDS.
    """
    now = time.monotonic()
    ip = _client_ip(request)
    hits = _hits[ip]

    # Drop timestamps that have aged out of the window.
    while hits and now - hits[0] > RATE_LIMIT_WINDOW_SECONDS:
        hits.popleft()

    if len(hits) >= RATE_LIMIT_REQUESTS:
        retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - hits[0])) + 1
        raise HTTPException(
            status_code=429,
            detail="Too many requests — slow down and try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )

    hits.append(now)

    if len(_hits) > _MAX_TRACKED_IPS:
        for tracked_ip in [k for k, v in _hits.items() if not v or now - v[-1] > RATE_LIMIT_WINDOW_SECONDS]:
            del _hits[tracked_ip]

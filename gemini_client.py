"""
gemini_client.py
================
The ONE place Gemini API clients are constructed.

Before this file existed, agent.py and embeddings.py each built their own
module-level `genai.Client(api_key=...)` at import time. That worked while
there was exactly one key (the server's), but Bring-Your-Own-Key means the
key can differ per request — so client construction moves behind a factory:

    from gemini_client import get_client

    client = get_client()            # server key (config.GEMINI_API_KEY)
    client = get_client(user_key)    # a specific user's key

Why the cache: building a genai.Client is cheap but not free, and the same
handful of keys (the server's + each active user's) repeats on every request.
lru_cache keeps ONE client per key and evicts the least-recently-used when
more than `maxsize` distinct keys have been seen — so the cache can't grow
without bound if someone sprays random keys at the API.

Security note: the cache keys are the API keys themselves, held only in this
process's memory. Nothing here ever logs, prints, or persists a key.
"""

from functools import lru_cache

from google import genai
from google.genai import errors

import config


@lru_cache(maxsize=64)
def _client_for(key: str) -> genai.Client:
    """Build (once) and cache the client for a given key. Internal — callers
    go through get_client() so the server-key fallback lives in one place."""
    return genai.Client(api_key=key)


def get_client(api_key: str | None = None) -> genai.Client:
    """
    Return a Gemini client for `api_key`, falling back to the server key
    when none is given.

    This fallback is the whole backward-compatibility story: every caller
    that existed before BYOK passes nothing and gets exactly the client the
    old module-level singletons would have been.
    """
    return _client_for(api_key or config.GEMINI_API_KEY)


def is_auth_error(exc: BaseException) -> bool:
    """
    True if `exc` looks like Gemini rejecting the API KEY itself (as opposed
    to a quota error, a bad model name, or a network failure).

    Why this exists: when a user brings their own key and it's wrong, they
    should see "your key was rejected" (a 401 they can fix), not a generic
    "generation failed" (a 502 they can't). api.py calls this inside its
    except blocks to tell the two apart.

    What Gemini actually sends for a bad key: HTTP 400 INVALID_ARGUMENT with
    "API key not valid" in the message (and 401/403 for expired/forbidden
    keys). A plain 400 can also mean a malformed request, so for 400 we only
    say yes when the message mentions the API key.
    """
    if not isinstance(exc, errors.APIError):
        return False
    if exc.code in (401, 403):
        return True
    return exc.code == 400 and "api key" in str(exc).lower()

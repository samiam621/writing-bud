# writing-bud

FastAPI backend for the Writing Buddy Chrome extension. It stores samples of
a user's writing as embeddings (FAISS) and uses Gemini to generate new text
in their voice.

## Endpoints

| Endpoint | Auth | What it does |
|---|---|---|
| `GET /health` | open (rate-limited) | Heartbeat; adds chunk count if `X-API-Key` is valid |
| `POST /ingest` | `X-API-Key` | Upload a writing sample (.txt/.md/.pdf/.docx); chunks + embeds it |
| `POST /chat` | `X-API-Key` | `{"message": "..."}` → `{"reply": "..."}` in the user's voice |
| `POST /validate-key` | `X-API-Key` | Checks the `X-Gemini-Key` header against Gemini → `{"valid": true/false}` |

## Bring your own key (BYOK)

Requests to `/ingest` and `/chat` may include an **`X-Gemini-Key`** header
with the caller's own Gemini API key:

- **Header present** — that key pays for the Gemini calls, and the caller's
  chunks are stored under a private owner label (`sha256(key)[:16]`), so
  each key only ever searches its own writing. The raw key is never stored
  or logged.
- **Header absent** — the server's `GEMINI_API_KEY` pays and data lives
  under the shared `default` owner (the pre-BYOK behavior). Set
  `REQUIRE_USER_KEY=true` to disable this fallback and make BYOK mandatory.

Note: `X-API-Key` (unlocks this server) and `X-Gemini-Key` (pays Gemini)
are independent — BYOK callers still need both.

## Rate limiting (hybrid)

Two sliding windows (see `config.py` / `security.py`):

1. **Per-IP, all endpoints** — the volumetric layer (20 req/min).
2. **Per-credential, expensive endpoints** — the authentication layer
   (10 req/min per Gemini key; per-IP for fallback callers). Rotating IPs
   doesn't reset this one — the budget follows the key.

## Environment variables (`.env`)

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | yes | Server's Gemini key (BYOK fallback) |
| `WRITING_BUDDY_API_KEY` | yes | Shared secret clients send as `X-API-Key` |
| `REQUIRE_USER_KEY` | no (default `false`) | `true` → reject requests without `X-Gemini-Key` |

## Run locally

```bash
pip install -r requirements.txt
python main.py            # http://127.0.0.1:8000 (docs at /docs)
```

Production (Render) runs `uvicorn api:app --host 0.0.0.0 --port $PORT`.

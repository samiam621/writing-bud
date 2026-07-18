"""
api.py
======
The HTTP layer. This is how the outside world (your future browser/Docs
extension, or a curl command) talks to the agent.

Design rule: keep this file DUMB. Each endpoint should just
    1. unpack the incoming request,
    2. call the right function from the other files,
    3. return JSON.
No business logic lives here — that's in ingestion.py / embeddings.py / agent.py.
If you ever swap FastAPI for something else, only this file changes.

Endpoints:
    POST /ingest   - upload a writing sample; store it in memory   (API key required)
    POST /chat     - send a request; get back text in the user's voice (API key required)
    GET  /health   - is the server up? (open, but rate-limited)

Every endpoint is rate-limited per client IP, and the two that cost money
(/ingest embeds, /chat calls Gemini) require the X-API-Key header. The guards
themselves live in security.py; the limits live in config.py.

The `app` object defined here is what main.py launches.
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import NamedTuple

from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The three building blocks. Note we import the SHARED `memory` instance so
# every request reads/writes the same FAISS index.
from ingestion import ingest
from embeddings import memory, MemoryFullError
from agent import generate

# BYOK: the cached client factory and the "was it the key's fault?" test.
from gemini_client import get_client, is_auth_error

# The guards (see security.py) and the limits they enforce (see config.py).
from security import require_api_key, rate_limit, is_valid_key, owner_for_key
from config import MAX_UPLOAD_BYTES, MAX_MESSAGE_CHARS, ALLOWED_UPLOAD_EXTENSIONS, REQUIRE_USER_KEY

# Errors get logged HERE, in full, with tracebacks — and only here. What goes
# back over HTTP is a generic message. str(some_exception) from pypdf/faiss/
# genai can contain absolute paths, library versions, and stack context: free
# reconnaissance for an attacker probing the API with malformed input. The
# rule: the log gets the truth, the client gets a category.
logger = logging.getLogger("writing_buddy")

# Startup/shutdown hook. Code before `yield` runs once, when the server
# starts accepting requests; code after would run at shutdown. Loading the
# saved memory HERE (not in main.py) matters for deployment: Render starts
# the app with `uvicorn api:app ...`, which never executes main.py — if the
# load lived there, production would boot with an empty index every time.
@asynccontextmanager
async def lifespan(app: FastAPI):
    memory.load()
    print(f"Loaded memory: {memory.index.ntotal} chunks ready.")
    yield

# Create the application object. The title shows up in the auto-generated
# docs at http://127.0.0.1:8000/docs — a free UI for testing your endpoints.
app = FastAPI(title="Writing Buddy API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS — let the browser extension / other origins call this API
# ---------------------------------------------------------------------------
# The Chrome extension and any browser-based frontend run on a different origin
# than this server, so the browser blocks their requests unless we explicitly
# allow them. allow_origins=["*"] is the permissive dev setting; tighten it to
# your extension's origin (e.g. "chrome-extension://<id>") for production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://bnfciedoefenhafalekgnahaijcodmkp"],  
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],

)


# ---------------------------------------------------------------------------
# Request body shapes (Pydantic models)
# ---------------------------------------------------------------------------
# FastAPI uses these to automatically parse + validate incoming JSON. If a
# request is missing "message" or sends the wrong type, FastAPI rejects it
# with a clear error before your code even runs.
class ChatRequest(BaseModel):
    # min_length stops empty spam; max_length caps what one request can cost —
    # without it, a 2MB message becomes a 2MB Gemini prompt on your bill.
    # Pydantic enforces both and returns a 422 before our code ever runs.
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


# ---------------------------------------------------------------------------
# BYOK: resolve which Gemini key (and which data namespace) a request uses
# ---------------------------------------------------------------------------
class GeminiContext(NamedTuple):
    """Everything downstream code needs to know about the caller's key,
    resolved ONCE per request by gemini_context() below."""
    client: object   # the genai client to bill this request's Gemini calls to
    owner: str       # namespace for FAISS chunks (hash of the key, or "default")
    user_supplied: bool  # True when the caller brought their own key


def gemini_context(x_gemini_key: str = Header(default="")) -> GeminiContext:
    """
    FastAPI dependency: read the optional X-Gemini-Key header and turn it
    into a (client, owner, user_supplied) triple.

    Three cases:
      - Header present  -> the user's own key pays; their chunks live under
                           a hashed owner label so nobody else can search them.
      - Header absent   -> the server key pays and owner is "default" — the
                           exact pre-BYOK behavior, kept as a fallback.
      - Header absent + REQUIRE_USER_KEY=true (config.py) -> 401. This is the
        switch that makes BYOK mandatory once the extension update is out.

    The raw key is passed straight into the cached client factory and then
    dropped — it is never stored, logged, or echoed back.
    """
    x_gemini_key = x_gemini_key.strip()
    if REQUIRE_USER_KEY and not x_gemini_key:
        raise HTTPException(
            status_code=401,
            detail="This server requires your own Gemini API key. Add it in the extension's settings.",
        )
    return GeminiContext(
        client=get_client(x_gemini_key or None),
        owner=owner_for_key(x_gemini_key),
        user_supplied=bool(x_gemini_key),
    )


def _rejected_key_response(exc: Exception, ctx: GeminiContext) -> HTTPException | None:
    """
    If `exc` is Gemini rejecting a USER-supplied key, return the 401 the
    caller should raise; otherwise None (meaning: not the key's fault, let
    the endpoint's normal error handling run).

    Only fires for user-supplied keys on purpose: if the SERVER key is bad,
    that's our misconfiguration — the client can't fix it, so they should
    see the generic 5xx, and the truth goes to the log as always.
    """
    if ctx.user_supplied and is_auth_error(exc):
        return HTTPException(
            status_code=401,
            detail="Your Gemini API key was rejected. Check it and try again.",
        )
    return None


# ---------------------------------------------------------------------------
# GET /health  — a heartbeat
# ---------------------------------------------------------------------------
@app.get("/health", dependencies=[Depends(rate_limit)])
def health(x_api_key: str = Header(default="")):
    """
    Returns a tiny JSON blob so you can confirm the server is alive.
    Invaluable while building the extension: hit this first to rule out
    "is the backend even running?" before debugging anything else.

    This endpoint stays unauthenticated on purpose — Render's health checker
    can't send custom headers, and a bare "ok" costs nothing and reveals
    nothing. It IS rate-limited, and the chunk count (mildly interesting to a
    stranger, useful to you) only appears when a valid API key is sent — which
    the extension always does.
    """
    body = {"status": "ok"}
    if is_valid_key(x_api_key):
        body["chunks_in_memory"] = memory.index.ntotal
    return body


# ---------------------------------------------------------------------------
# POST /ingest  — upload a writing sample
# ---------------------------------------------------------------------------
@app.post("/ingest", dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def ingest_file(
    request: Request,
    file: UploadFile = File(...),
    ctx: GeminiContext = Depends(gemini_context),
):
    """
    Receives an uploaded file, turns it into chunks, and stores them.

    Flow (each step is a call into another file — this endpoint is just glue):
        save upload to a temp path
          -> ingestion.ingest(path)   -> list[str] chunks
          -> memory.add_chunks(chunks) -> stored in FAISS
          -> memory.save()             -> persisted to disk

    BYOK: `ctx` (resolved from the X-Gemini-Key header) decides which key
    pays for the embeddings and which owner label the chunks are stored
    under. No header -> server key + "default" owner, same as always.

    Returns how many chunks were stored.
    """
    # Fast reject when the client declares its size up front. Honest clients
    # always send Content-Length; liars are caught by the streaming cap below.
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large — the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )

    # Validate the file type BEFORE writing anything to disk. Order matters:
    # letting ingestion.read_file() reject the extension later would mean
    # we'd already accepted arbitrary bytes onto the filesystem. Checking here
    # means unsupported uploads never touch disk at all. (The message lists
    # the allowed types — that's public info from our own docs, not a leak.)
    suffix = os.path.splitext(file.filename or "")[1].lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_UPLOAD_EXTENSIONS))}",
        )

    # UploadFile is a stream, not a path. ingestion.read_file() wants a real
    # file on disk (and needs the right extension to pick its reader), so we
    # write the upload to a temp file first, preserving the original extension.
    try:
        # delete=False so the file stays on disk after this block; we remove it
        # ourselves in `finally` once ingestion has read it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            # Copy the upload to disk 1MB at a time, counting as we go, and
            # abort the moment the running total crosses the cap. At no point
            # is more than 1MB of the upload in RAM — a deliberately huge POST
            # can no longer OOM the process (Render free tier: 512MB total).
            received = 0
            while chunk := await file.read(1024 * 1024):
                received += len(chunk)
                if received > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large — the limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                    )
                tmp.write(chunk)

        chunks = ingest(tmp_path)          # read + clean + chunk
        added = memory.add_chunks(
            chunks,
            owner=ctx.owner,
            source=file.filename or "upload",
            client=ctx.client,
        )
        memory.save()                      # persist so it survives a restart

    except HTTPException:
        # Our own deliberate rejections (e.g. the 413 size cap above) — let
        # them pass through untouched instead of being re-wrapped below.
        raise
    except MemoryFullError as e:
        # The index is at MAX_TOTAL_CHUNKS. 507 "Insufficient Storage" — and
        # this message is ours (written in embeddings.py), safe to show.
        raise HTTPException(status_code=507, detail=str(e))
    except ValueError:
        # A file with an allowed extension but contents we couldn't parse.
        # Log the real reason; the client just learns the category. str(e)
        # here can come from deep inside pypdf/python-docx and may embed
        # paths or library internals — that stays in the log.
        logger.exception("Failed to parse upload %r", file.filename)
        raise HTTPException(status_code=400, detail="Could not read that file — is it a valid document?")
    except Exception as exc:
        # A rejected USER key is the one failure the caller can actually fix,
        # so it gets its own clear 401 instead of the generic 500 below.
        if rejected := _rejected_key_response(exc, ctx):
            raise rejected
        # Anything else (faiss, the embedding API, disk) — same rule: full
        # traceback to the log, a generic 500 to the client.
        logger.exception("Ingest failed for upload %r", file.filename)
        raise HTTPException(status_code=500, detail="Something went wrong storing that file. Try again.")
    finally:
        # Always clean up the temp file, even if something above failed.
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"filename": file.filename, "chunks_added": added}


# ---------------------------------------------------------------------------
# POST /chat  — get a reply in the user's voice
# ---------------------------------------------------------------------------
@app.post("/chat", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def chat_endpoint(request: ChatRequest, ctx: GeminiContext = Depends(gemini_context)):
    """
    Takes {"message": "..."} and returns {"reply": "..."}.

    All the real work — retrieving style chunks and calling Gemini — happens
    inside agent.generate(). This endpoint just hands the message over and
    wraps the result in JSON.

    BYOK: `ctx` decides which key pays and which owner's chunks are searched.
    Passing ctx.owner explicitly (even for the "default" fallback) is the
    isolation guarantee in both directions — BYOK users only see their own
    style, and fallback users never see BYOK users' chunks either.
    """
    try:
        reply = generate(request.message, owner=ctx.owner, client=ctx.client)
    except Exception as exc:
        # A rejected USER key gets a clear, fixable 401 (see the helper).
        if rejected := _rejected_key_response(exc, ctx):
            raise rejected
        # Same policy as /ingest: the full error (often a genai exception
        # that names models, quotas, or internals) goes to the log; the
        # client gets a category. 502 = "the upstream service failed us."
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail="Generation failed. Try again in a moment.")
    return {"reply": reply}
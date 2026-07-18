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

from fastapi import Depends, FastAPI, Header, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# The three building blocks. Note we import the SHARED `memory` instance so
# every request reads/writes the same FAISS index.
from ingestion import ingest
from embeddings import memory, MemoryFullError
from agent import generate

# The guards (see security.py) and the limits they enforce (see config.py).
from security import require_api_key, rate_limit, is_valid_key
from config import MAX_UPLOAD_BYTES, MAX_MESSAGE_CHARS, ALLOWED_UPLOAD_EXTENSIONS

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
async def ingest_file(request: Request, file: UploadFile = File(...)):
    """
    Receives an uploaded file, turns it into chunks, and stores them.

    Flow (each step is a call into another file — this endpoint is just glue):
        save upload to a temp path
          -> ingestion.ingest(path)   -> list[str] chunks
          -> memory.add_chunks(chunks) -> stored in FAISS
          -> memory.save()             -> persisted to disk

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
        added = memory.add_chunks(chunks, source=file.filename or "upload")
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
    except Exception:
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
def chat_endpoint(request: ChatRequest):
    """
    Takes {"message": "..."} and returns {"reply": "..."}.

    All the real work — retrieving style chunks and calling Gemini — happens
    inside agent.generate(). This endpoint just hands the message over and
    wraps the result in JSON.
    """
    try:
        reply = generate(request.message)
    except Exception:
        # Same policy as /ingest: the full error (often a genai exception
        # that names models, quotas, or internals) goes to the log; the
        # client gets a category. 502 = "the upstream service failed us."
        logger.exception("Generation failed")
        raise HTTPException(status_code=502, detail="Generation failed. Try again in a moment.")
    return {"reply": reply}
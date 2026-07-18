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
    POST /ingest   - upload a writing sample; store it in memory
    POST /chat     - send a request; get back text in the user's voice
    GET  /health   - is the server up? (trivial but very handy)

The `app` object defined here is what main.py launches.
"""

import os
import tempfile

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# The three building blocks. Note we import the SHARED `memory` instance so
# every request reads/writes the same FAISS index.
from ingestion import ingest
from embeddings import memory
from agent import generate

# Create the application object. The title shows up in the auto-generated
# docs at http://127.0.0.1:8000/docs — a free UI for testing your endpoints.
app = FastAPI(title="Writing Buddy API")

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
    message: str


# ---------------------------------------------------------------------------
# GET /health  — a heartbeat
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    Returns a tiny JSON blob so you can confirm the server is alive.
    Invaluable while building the extension: hit this first to rule out
    "is the backend even running?" before debugging anything else.
    """
    return {"status": "ok", "chunks_in_memory": memory.index.ntotal}


# ---------------------------------------------------------------------------
# POST /ingest  — upload a writing sample
# ---------------------------------------------------------------------------
@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Receives an uploaded file, turns it into chunks, and stores them.

    Flow (each step is a call into another file — this endpoint is just glue):
        save upload to a temp path
          -> ingestion.ingest(path)   -> list[str] chunks
          -> memory.add_chunks(chunks) -> stored in FAISS
          -> memory.save()             -> persisted to disk

    Returns how many chunks were stored.
    """
    # UploadFile is a stream, not a path. ingestion.read_file() wants a real
    # file on disk (and needs the right extension to pick its reader), so we
    # write the upload to a temp file first, preserving the original extension.
    suffix = os.path.splitext(file.filename or "")[1]
    try:
        # delete=False so the file stays on disk after this block; we remove it
        # ourselves in `finally` once ingestion has read it.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())   # await: reading the upload is async
            tmp_path = tmp.name

        chunks = ingest(tmp_path)          # read + clean + chunk
        added = memory.add_chunks(chunks)  # embed + store in FAISS
        memory.save()                      # persist so it survives a restart

    except ValueError as e:
        # e.g. an unsupported file type from ingestion.read_file(). Turn it into
        # a proper 400 (bad request) instead of a generic 500 crash.
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        # Always clean up the temp file, even if something above failed.
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"filename": file.filename, "chunks_added": added}


# ---------------------------------------------------------------------------
# POST /chat  — get a reply in the user's voice
# ---------------------------------------------------------------------------
@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    """
    Takes {"message": "..."} and returns {"reply": "..."}.

    All the real work — retrieving style chunks and calling Gemini — happens
    inside agent.generate(). This endpoint just hands the message over and
    wraps the result in JSON.
    """
    reply = generate(request.message)
    return {"reply": reply}
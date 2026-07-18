"""
config.py
=========
One home for every setting in the project: API keys, model names, file
paths, and tuning knobs. Nothing here does work — it just holds values the
other files import.

Why centralize: when you want to try a different model, bigger chunks, or
more retrieved results, you change ONE line here instead of editing four files.

Rule of thumb: if a value is a "magic number" or a secret, it belongs in config.py.
"""

import os

# Load variables from a .env file into the environment. WITHOUT this, Python
# never reads .env on its own — os.environ would only see shell-exported vars.
# Requires: pip install python-dotenv
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# SECRETS  (never hardcode these — read them from the environment)
# ---------------------------------------------------------------------------
# The string below MUST exactly match the variable name in your .env file.
# Your .env uses `GEMINI_API_KEY`, so we look up "GEMINI_API_KEY" here. (Renaming it to
# GEMINI_API_KEY in .env and here is the tidier convention, but either works
# as long as the two names match.)
#
# Example .env line:
#     geminiAPI="your-key-here"
#
# The second argument to os.environ.get is a fallback ("") if it's not set.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Fail loudly and early if the key is missing, instead of getting a confusing
# error later from deep inside the Gemini library.
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Set it before making API calls.")

# The shared secret clients must send in the X-API-Key header to use /ingest
# and /chat. Generate one with:
#     python -c "import secrets; print(secrets.token_urlsafe(32))"
# and put it in .env (and in the extension's config.js). If this is unset the
# API refuses all authenticated requests — fail CLOSED, never open.
API_KEY = os.environ.get("WRITING_BUDDY_API_KEY", "")

if not API_KEY:
    print("WARNING: WRITING_BUDDY_API_KEY is not set. /ingest and /chat will reject all requests.")

# ---------------------------------------------------------------------------
# REQUEST LIMITS  (protect the server and the Gemini budget from abuse)
# ---------------------------------------------------------------------------
# Hard cap on upload size for /ingest. The endpoint streams the upload and
# aborts as soon as this many bytes have arrived, so a huge POST can never
# fill RAM (Render's free tier only gives the process 512MB).
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB — generous for a writing sample

# Hard cap on /chat message length, enforced by Pydantic before our code runs.
# Prompt cost scales with input size; this keeps one request's cost bounded.
MAX_MESSAGE_CHARS = 4000

# Simple per-IP rate limit shared by all endpoints: at most RATE_LIMIT_REQUESTS
# requests per RATE_LIMIT_WINDOW_SECONDS from one client IP. Even with a valid
# (or leaked) key, one client can't hammer the API in a tight loop.
RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60

# The only file types /ingest will accept. Checked BEFORE the upload is written
# anywhere, so bytes of any other type never touch the disk. Must stay in sync
# with what ingestion.read_file() can actually parse.
ALLOWED_UPLOAD_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

# Hard ceiling on how many chunks the FAISS index may ever hold. Two reasons:
#   1. Memory: vectors live in RAM (768 floats * 4 bytes ≈ 3KB per chunk, plus
#      the text itself) — unbounded uploads would eventually OOM the process.
#   2. Speed: IndexFlatIP compares the query against EVERY stored vector, so
#      search latency grows linearly with size.
# 5000 chunks ≈ 2.5M characters of writing samples — plenty for one voice.
MAX_TOTAL_CHUNKS = 5000

# The owner label stamped on chunks when no specific user is known. There's a
# single shared API key today, so everything is stored under this one owner —
# but every chunk carries the column, so adding real per-user keys later is a
# code change, not a data migration.
DEFAULT_OWNER = "default"

# ---------------------------------------------------------------------------
# MODELS  (which Gemini models to use for each job)
# ---------------------------------------------------------------------------
# Turns text into vectors. embeddings.py uses this.
# NOTE: the old "text-embedding-004" model was retired from the Gemini API
# (you'll get a 404 NOT_FOUND if you use it). "gemini-embedding-001" is the
# current model. It defaults to 3072-dim vectors but supports 768/1536/3072
# via output_dimensionality — we ask for 768 in embeddings.py to match EMBED_DIM.
# (The new google-genai SDK takes the bare name, no "models/" prefix.)
EMBED_MODEL = "gemini-embedding-001"

# Length of the vectors we request from EMBED_MODEL. Must match EMBED_DIM in
# embeddings.py, which sizes the FAISS index. 768 keeps parity with the old
# text-embedding-004 setup; valid values for gemini-embedding-001 are 768/1536/3072.
EMBED_DIM = 768

# Writes the actual responses. agent.py uses this. "flash" is fast and cheap;
# swap to "gemini-3.5-pro" if you want higher-quality writing at more cost.
# NOTE: older models get retired regularly — gemini-1.5-* and even the bare
# gemini-2.5-flash alias now return 404 for newer API keys. gemini-3.5-flash is
# the current fast/cheap generation model verified working on this key. To see
# what your key can use, call client.models.list() and filter for generateContent.
GENERATION_MODEL = "gemini-3.5-flash"

# ---------------------------------------------------------------------------
# STORAGE PATHS  (where the FAISS memory is saved on disk)
# ---------------------------------------------------------------------------
# embeddings.py writes/reads these in save() and load(). Both live in a
# "store/" folder so your saved memory stays tidy and separate from code.
STORE_DIR = "store"
INDEX_PATH = os.path.join(STORE_DIR, "index.faiss")   # the FAISS vectors
TEXTS_PATH = os.path.join(STORE_DIR, "texts.json")    # the parallel chunk texts

# ---------------------------------------------------------------------------
# TUNING KNOBS  (the numbers you'll experiment with most)
# ---------------------------------------------------------------------------
# ingestion.py uses these when splitting text into chunks.
# CHUNK_SIZE   = characters per chunk. Smaller = more precise style matches
#                but less context per chunk. ~500 is a good start.
# CHUNK_OVERLAP = characters shared between neighboring chunks, so a sentence's
#                style isn't cut cleanly in half at a boundary.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# embeddings.py uses this: how many style chunks search() returns by default.
# More chunks = richer style signal but a longer prompt to Gemini. 3-5 is typical.
TOP_K = 4
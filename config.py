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
# Your .env uses `geminiAPI`, so we look up "geminiAPI" here. (Renaming it to
# GEMINI_API_KEY in .env and here is the tidier convention, but either works
# as long as the two names match.)
#
# Example .env line:
#     geminiAPI="your-key-here"
#
# The second argument to os.environ.get is a fallback ("") if it's not set.
GEMINI_API_KEY = os.environ.get("geminiAPI", "")

# Fail loudly and early if the key is missing, instead of getting a confusing
# error later from deep inside the Gemini library.
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY is not set. Set it before making API calls.")

# ---------------------------------------------------------------------------
# MODELS  (which Gemini models to use for each job)
# ---------------------------------------------------------------------------
# Turns text into vectors. embeddings.py uses this. 004 returns 768-dim vectors,
# which is the EMBED_DIM hardcoded in embeddings.py — keep them in sync.
# (The new google-genai SDK takes the bare name, no "models/" prefix.)
EMBED_MODEL = "text-embedding-004"

# Writes the actual responses. agent.py uses this. "flash" is fast and cheap;
# swap to "gemini-1.5-pro" if you want higher-quality writing at more cost.
GENERATION_MODEL = "gemini-1.5-flash"

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
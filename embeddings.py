"""
embeddings.py
=============
This file is the "memory" of the writing agent.

Its whole job: turn text into vectors (numbers that capture meaning/style),
store those vectors in a FAISS index, and — given a new piece of text —
find the stored chunks that are most similar.

How it connects to the other files:

    ingestion.py  ── gives us list[str] of chunks ──►  add_chunks()
    agent.py      ── asks "find style like this" ──►  search()  ── returns list[str]
    config.py     ── gives us the API key, model names, paths, TOP_K

Nobody else in the project touches FAISS or the embedding model directly.
If you ever swap FAISS for another vector store, this is the ONLY file
that changes — the functions below keep the same inputs and outputs.
"""

import os
import json
import numpy as np
import faiss
# New official Gemini SDK. Install with: pip install google-genai
# (The old `google.generativeai` package is deprecated.)
from google import genai
from google.genai import types

# We import settings from config.py so keys/models/paths live in ONE place.
# (If you haven't written config.py yet, see the fallback defaults below.)
try:
    import config
    GEMINI_API_KEY = config.GEMINI_API_KEY
    EMBED_MODEL = config.EMBED_MODEL          # e.g. "models/text-embedding-004"
    INDEX_PATH = config.INDEX_PATH            # where the FAISS index is saved
    TEXTS_PATH = config.TEXTS_PATH            # where the chunk texts are saved
    TOP_K = config.TOP_K                      # how many chunks to retrieve by default
except ImportError:
    # Fallback so this file can run on its own while you build the others.
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
    EMBED_MODEL = "models/text-embedding-004"
    INDEX_PATH = "store/index.faiss"
    TEXTS_PATH = "store/texts.json"
    TOP_K = 4

# In the new SDK you create ONE client object and reuse it, instead of a
# global configure() call. It holds your key and talks to the API.
client = genai.Client(api_key=GEMINI_API_KEY)

# text-embedding-004 returns vectors of length 768. FAISS needs to know this
# number up front so it can size the index correctly.
EMBED_DIM = 768


class StyleMemory:
    """
    Wraps the FAISS index PLUS the list of original chunk texts.

    Why both? FAISS only stores vectors and gives back integer positions
    (0, 1, 2, ...). It has NO idea what the original sentence was. So we keep
    a plain Python list `self.texts` in the exact same order we added vectors.
    Position 5 in the FAISS index  ==  self.texts[5].  That parallel list is
    the bridge that turns a search result back into readable text.

    Using a class keeps the index and the texts glued together, so they can
    never drift out of sync.
    """

    def __init__(self):
        # IndexFlatIP = "inner product" search. Combined with normalized
        # vectors (see _normalize below), inner product == cosine similarity,
        # which is the standard way to measure "how similar in meaning/style."
        # "Flat" means it compares against every vector exactly — simple and
        # accurate, perfect until you have tens of thousands of chunks.
        self.index = faiss.IndexFlatIP(EMBED_DIM)

        # The parallel list of original chunk strings (the bridge described above).
        self.texts: list[str] = []

    # ---------- internal helpers (leading underscore = "private, don't call from outside") ----------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """
        Turn a list of strings into a matrix of vectors using Gemini.

        Input:  ["some text", "more text"]
        Output: numpy array of shape (number_of_texts, 768), dtype float32.

        FAISS specifically wants float32, so we cast at the end.
        Used for BOTH storing chunks (add_chunks) and searching (search),
        which is why it lives in one place.
        """
        # The new SDK embeds a whole LIST in one call (faster + fewer requests).
        # task_type="SEMANTIC_SIMILARITY" tells Gemini we're comparing texts for
        # likeness, which tunes the vectors for exactly our search use case.
        response = client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )

        # response.embeddings is a list; each item has a .values attribute
        # holding the actual list of floats. Pull those into a plain 2D array.
        vectors = [e.values for e in response.embeddings]

        # Stack into one 2D array and normalize so inner-product search behaves
        # as cosine similarity.
        matrix = np.array(vectors, dtype="float32")
        return self._normalize(matrix)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """
        Scale each vector to length 1. This is what lets IndexFlatIP measure
        cosine similarity. faiss.normalize_L2 does it in place, efficiently.
        """
        faiss.normalize_L2(matrix)
        return matrix

    # ---------- public API: these are the functions other files call ----------

    def add_chunks(self, chunks: list[str]) -> int:
        """
        Store new writing samples.

        Called by ingestion.py (or api.py) right after a file is chunked:

            chunks = ingestion.ingest("my_essay.txt")
            memory.add_chunks(chunks)

        Steps:
          1. Embed the chunks into vectors.
          2. Add those vectors to the FAISS index.
          3. Append the original texts to self.texts IN THE SAME ORDER,
             so positions stay aligned.

        Returns how many chunks were added (handy for a "stored 12 chunks" message).
        """
        if not chunks:
            return 0

        vectors = self._embed(chunks)   # step 1
        self.index.add(vectors)         # step 2
        self.texts.extend(chunks)       # step 3
        return len(chunks)

    def search(self, query: str, k: int = TOP_K) -> list[str]:
        """
        Find the stored chunks most stylistically similar to `query`.

        Called by agent.py when composing a reply:

            style_chunks = memory.search(user_prompt)
            # ...then feed style_chunks to Gemini as style examples.

        Input:  a query string + how many results you want (defaults to TOP_K).
        Output: a list of the matching ORIGINAL chunk strings (not vectors,
                not numbers) — exactly what agent.py needs to build its prompt.

        This clean "string in, strings out" shape is the whole point: agent.py
        never has to know FAISS exists.
        """
        # Guard: searching an empty memory would error, so return nothing.
        if self.index.ntotal == 0:
            return []

        # Embed the query the SAME way we embedded the chunks, so they live in
        # the same vector space and are comparable. Note [query] -> a list.
        query_vector = self._embed([query])

        # FAISS returns two arrays:
        #   scores    = similarity score of each hit (higher = more similar)
        #   positions = the index positions of each hit (this is what maps to self.texts)
        scores, positions = self.index.search(query_vector, k)

        # positions is shaped (1, k) because we searched one query. Take row 0,
        # then translate each position back into its original text via self.texts.
        # (-1 can appear if fewer than k vectors exist; we skip those.)
        results = [self.texts[i] for i in positions[0] if i != -1]
        return results

    # ---------- persistence: so users don't re-upload every session ----------

    def save(self) -> None:
        """
        Write the index and the texts to disk (paths come from config.py).

        Call this after add_chunks so the memory survives a restart. We save
        TWO things because the memory is two things: the FAISS vectors AND the
        parallel text list. Saving one without the other would break the bridge.
        """
        # Make sure the folder exists (e.g. "store/") before writing into it.
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

        faiss.write_index(self.index, INDEX_PATH)              # the vectors
        with open(TEXTS_PATH, "w", encoding="utf-8") as f:     # the texts
            json.dump(self.texts, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        """
        Reload a previously saved memory. Call this once on startup (e.g. in
        api.py) so the agent already "remembers" the user's past uploads.

        If no saved files exist yet, we quietly start with an empty memory
        instead of crashing.
        """
        if os.path.exists(INDEX_PATH) and os.path.exists(TEXTS_PATH):
            self.index = faiss.read_index(INDEX_PATH)
            with open(TEXTS_PATH, "r", encoding="utf-8") as f:
                self.texts = json.load(f)
        # else: keep the fresh, empty index/texts created in __init__.


# A single shared instance the rest of the app imports and reuses.
# In other files you just write:  from embeddings import memory
# ...then memory.add_chunks(...) / memory.search(...). One memory, one source of truth.
memory = StyleMemory()


# ---------- quick manual test ----------
# Run `python embeddings.py` directly to sanity-check this file on its own,
# BEFORE wiring it into ingestion.py or agent.py. This block does nothing when
# the file is imported elsewhere — it only runs when executed directly.
if __name__ == "__main__":
    # Pretend these came out of ingestion.py.
    sample_chunks = [
        "The morning light spilled across the quiet kitchen, slow and golden.",
        "Quarterly revenue increased 12% driven by strong enterprise demand.",
        "I never trust a recipe that doesn't ask you to taste as you go.",
    ]

    memory.add_chunks(sample_chunks)
    print("Stored chunks:", memory.index.ntotal)

    hits = memory.search("write something cozy about breakfast", k=2)
    print("\nMost similar chunks to the query:")
    for h in hits:
        print(" -", h)
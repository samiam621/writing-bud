"""
Ingestion layer: raw files -> clean text -> chunks.

This file only knows how to read files and split text. It does NOT know
about embeddings or Gemini. That separation matters: if tomorrow you add
support for .rtf files, you only touch this file.
"""
"""
ingestion.py
============
This file turns a raw uploaded file into clean, bite-sized text chunks.

It deals ONLY with text — it never touches Gemini or FAISS. That separation is
deliberate: file parsing is fiddly and format-specific, so keeping it isolated
means adding PDF/Word support later can never break your search code.

How it connects to the other files:

    api.py / your script  ── calls ──►  ingest("essay.txt")  ── returns list[str]
                                                                      │
                                          embeddings.py ◄─────────────┘
                                          memory.add_chunks(chunks)

So the handoff is exactly one line:

    from ingestion import ingest
    from embeddings import memory

    chunks = ingest("essay.txt")   # list[str]
    memory.add_chunks(chunks)      # store them

`ingest()` is the only function other files need. The three helpers below
(read_file, clean_text, chunk_text) are the steps it runs internally.
"""

import os
import re

# Tuning knobs live in config.py so you tweak them in one place.
try:
    import config
    CHUNK_SIZE = config.CHUNK_SIZE        # characters per chunk
    CHUNK_OVERLAP = config.CHUNK_OVERLAP  # characters shared between neighbors
except ImportError:
    # Fallback defaults so this file runs on its own while you build the others.
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50


# ---------------------------------------------------------------------------
# STEP 1: read the file into one big string
# ---------------------------------------------------------------------------
def read_file(path: str) -> str:
    """
    Extract plain text from a file, picking the right reader by extension.

    Input:  a path like "essays/my_essay.pdf"
    Output: the file's text as one string.

    Supports .txt / .md today, plus .pdf and .docx IF the optional libraries
    are installed. Adding a new format later = adding one more `elif` here,
    and nothing else in the project changes.
    """
    # os.path.splitext splits "file.pdf" -> ("file", ".pdf"). We lowercase the
    # extension so ".PDF" and ".pdf" are treated the same.
    ext = os.path.splitext(path)[1].lower()

    if ext in (".txt", ".md"):
        # Plain text: just open and read. utf-8 handles most characters safely.
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        # PDFs need a library. Requires: pip install pypdf
        from pypdf import PdfReader
        reader = PdfReader(path)
        # Pull text out of each page and join with newlines.
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    elif ext == ".docx":
        # Word docs need a library. Requires: pip install python-docx
        import docx
        document = docx.Document(path)
        # A .docx is a list of paragraphs; join their text with newlines.
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    else:
        # Fail clearly instead of silently returning nothing.
        raise ValueError(f"Unsupported file type: {ext}. Try .txt, .md, .pdf, or .docx")


# ---------------------------------------------------------------------------
# STEP 2: clean the text
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Tidy raw text so the embeddings aren't polluted by junk whitespace.

    - Collapse runs of spaces/tabs into a single space.
    - Collapse 3+ blank lines into a single blank line (keeps paragraph breaks).
    - Strip leading/trailing whitespace.

    Small step, but messy whitespace produces noisier vectors, so it's worth it.
    """
    # \r\n (Windows newlines) -> \n so line handling is consistent.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Multiple spaces/tabs -> one space.
    text = re.sub(r"[ \t]+", " ", text)
    # 3+ newlines -> 2 (one blank line = a paragraph boundary).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# STEP 3: split the text into overlapping chunks
# ---------------------------------------------------------------------------
def chunk_text(text: str) -> list[str]:
    """
    Slice a long string into chunks of ~CHUNK_SIZE characters, with
    CHUNK_OVERLAP characters shared between neighboring chunks.

    Why overlap? If we cut cleanly every 500 chars, a sentence's style could be
    split across a boundary and lost. Overlap lets each chunk keep a bit of its
    neighbor's tail, so no phrase falls through the cracks.

    Input:  one long string.
    Output: list of chunk strings — the format embeddings.py expects.

    We move a window forward by (CHUNK_SIZE - CHUNK_OVERLAP) each step. Example
    with size 500 / overlap 50: chunk 1 = chars 0-500, chunk 2 = chars 450-950,
    chunk 3 = chars 900-1400, and so on.
    """
    if not text:
        return []

    chunks = []
    start = 0
    step = CHUNK_SIZE - CHUNK_OVERLAP  # how far the window advances each loop

    # Safety: if someone sets overlap >= size, step would be <= 0 and loop forever.
    if step <= 0:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()   # grab the window, trim its edges
        if chunk:                         # skip empty pieces
            chunks.append(chunk)
        start += step                     # slide the window forward

    return chunks


# ---------------------------------------------------------------------------
# THE ONE PUBLIC FUNCTION: read -> clean -> chunk
# ---------------------------------------------------------------------------
def ingest(path: str) -> list[str]:
    """
    The single function other files call. Ties the three steps together.

    Input:  a file path.
    Output: list[str] of clean chunks, ready to hand straight to
            embeddings.memory.add_chunks(...).

    Usage:
        chunks = ingest("essay.txt")
        memory.add_chunks(chunks)
    """
    raw = read_file(path)      # step 1
    cleaned = clean_text(raw)  # step 2
    chunks = chunk_text(cleaned)  # step 3
    return chunks


# ---------------------------------------------------------------------------
# quick manual test — runs only when you execute `python ingestion.py` directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Make a tiny sample file, ingest it, and print the chunks. No AI needed,
    # so this is a fast way to confirm chunking works before wiring in embeddings.
    sample_path = "sample.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        f.write(
            "The morning light spilled across the quiet kitchen. "
            "I never trust a recipe that doesn't ask you to taste as you go. "
            "Good writing, like good cooking, is mostly about paying attention. " * 10
        )

    result = ingest(sample_path)
    print(f"Produced {len(result)} chunks.\n")
    for i, chunk in enumerate(result):
        print(f"--- chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:120], "...\n")

    os.remove(sample_path)  # clean up the temp file
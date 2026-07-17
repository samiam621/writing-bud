"""
agent.py
========
The "brain" of the writing agent — it turns a user request into a Gemini
response that sounds like the user.

It leans on two other files and knows nothing about their internals:

    embeddings.py  ── memory.search(prompt) ──►  list[str] of style chunks
    config.py      ── the generation model name + API key

Crucially, agent.py NEVER sees FAISS. It just asks memory.search() for the
most relevant passages of the user's writing and gets back plain strings.
That clean boundary means you can rewrite your prompt strategy here without
touching storage/search, and vice versa.

Flow of one turn:
    user request
        │
        ├─► memory.search(request)         # get the user's most similar writing
        │         └─► list[str] style chunks
        │
        ├─► build_prompt(request, chunks)  # assemble the instruction for Gemini
        │
        └─► Gemini.generate_content(...)   # -> the reply, in the user's voice
"""

# New official Gemini SDK. Install with: pip install google-genai
# (The old `google.generativeai` package is deprecated.)
from google import genai
from google.genai import types

# Pull the model name + key from config. Fallbacks let this file run alone.
try:
    import config
    GENERATION_MODEL = config.GENERATION_MODEL
    GEMINI_API_KEY = config.GEMINI_API_KEY
except ImportError:
    import os
    GENERATION_MODEL = "gemini-1.5-flash"
    GEMINI_API_KEY = os.environ.get("geminiAPI", "")

# The retrieval memory. This is the ONLY link to embeddings/FAISS, and it
# speaks in plain strings — agent.py stays blissfully unaware of vectors.
from embeddings import memory

# One reusable client, created when this module loads.
client = genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# THE SYSTEM PROMPT — the agent's standing instructions (its "personality")
# ---------------------------------------------------------------------------
# This is the single most important knob for output quality. Expect to edit it
# a lot. Everything from the line marked "# --- extension ---" down is my
# completion of the prompt you started; rewrite it to taste.
SYSTEM_PROMPT = """
You are a ghostwriting assistant. Your job is to help the user write new content
that sounds authentically like them — not like an AI.
You will be given:
1. STYLE EXAMPLES: Passages from the user's own past writing
2. USER REQUEST: What they want written
Your rules:
- Mirror the user's vocabulary, sentence length, rhythm, and tone
- If they write casually, be casual. If they write formally, be formal.
- Match their
  # --- extension (edit freely below) ---
  level of detail, punctuation habits, and quirks (contractions, slang,
  em-dashes, one-line paragraphs, etc.)
- Study the STYLE EXAMPLES for VOICE only. Do NOT copy their sentences,
  phrases, or specific facts into your answer — imitate the style, write new words.
- Write ONLY the requested content. No preamble, no "Here's your draft:",
  no explanations, no sign-off unless the request asks for one.
- If the STYLE EXAMPLES are thin or unclear, lean toward a clean, natural
  version of the request rather than inventing an exaggerated voice.
"""


# ---------------------------------------------------------------------------
# STEP: assemble the full prompt sent to Gemini
# ---------------------------------------------------------------------------
def build_prompt(user_request: str, style_chunks: list[str]) -> str:
    """
    Combine the system prompt, the retrieved style examples, and the user's
    request into one string for Gemini.

    Input:
        user_request  - what the user wants written
        style_chunks  - list[str] straight from memory.search()
    Output:
        one prompt string.

    Separating this from generate() means you can print/inspect the exact
    prompt while tuning, without making an API call.

    Note: SYSTEM_PROMPT is NOT included here. In the new SDK the standing
    instructions go in their own `system_instruction` slot (see generate()),
    which the model weights more heavily than ordinary prompt text. This
    function only builds the per-request part: the examples + the request.
    """
    # Number the examples so they read as distinct samples, not one blob.
    if style_chunks:
        examples = "\n\n".join(
            f"[Example {i + 1}]\n{chunk}" for i, chunk in enumerate(style_chunks)
        )
    else:
        # No stored writing yet — say so plainly so the model doesn't hallucinate a voice.
        examples = "(No style examples available yet.)"

    # Clear section headers matching the system prompt's vocabulary.
    return (
        f"STYLE EXAMPLES:\n{examples}\n\n"
        f"USER REQUEST:\n{user_request}\n\n"
        f"Now write the requested content in the user's voice:"
    )


# ---------------------------------------------------------------------------
# THE MAIN ENTRY POINT — retrieve, build, generate
# ---------------------------------------------------------------------------
def generate(user_request: str) -> str:
    """
    Produce a reply written in the user's style.

    This is the one function api.py / your script calls:
        from agent import generate
        reply = generate("write a birthday message for my coworker")

    Steps:
      1. Ask embeddings for the most stylistically relevant chunks.
      2. Build the full prompt from those chunks + the request.
      3. Send it to Gemini and return the text.
    """
    # 1. Retrieve style. memory.search returns list[str]; agent never sees FAISS.
    style_chunks = memory.search(user_request)

    # 2. Build the per-request part of the prompt (examples + request).
    prompt = build_prompt(user_request, style_chunks)

    # 3. Generate. The standing rules go in system_instruction; the per-request
    #    text goes in contents. .text is the generated string.
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


# ---------------------------------------------------------------------------
# OPTIONAL next step: multi-turn chat with memory of the conversation
# ---------------------------------------------------------------------------
def chat(user_request: str, history: list[dict] | None = None) -> str:
    """
    Same idea as generate(), but also feeds prior turns so the conversation
    has continuity (useful once you build the chat extension).

    history is a list like:
        [{"role": "user", "content": "..."},
         {"role": "assistant", "content": "..."}]

    You can flesh this out later; generate() is enough to get the agent working.
    """
    style_chunks = memory.search(user_request)
    prompt = build_prompt(user_request, style_chunks)

    # Prepend a simple text transcript of the history so the model has context.
    if history:
        transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
        prompt = f"CONVERSATION SO FAR:\n{transcript}\n\n{prompt}"

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    return response.text


# ---------------------------------------------------------------------------
# quick manual test — run `python agent.py` after you've ingested some writing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from ingestion import ingest

    # Load some of the user's writing into memory first, then generate.
    # Replace "sample.txt" with a real file of your own writing to see the effect.
    try:
        chunks = ingest("sample.txt")
        memory.add_chunks(chunks)
        print(f"Loaded {len(chunks)} style chunks.\n")
    except FileNotFoundError:
        print("No sample.txt found — generating without style examples.\n")

    reply = generate("Write a short thank-you note to a mentor.")
    print("--- AGENT REPLY ---")
    print(reply)
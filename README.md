# Writing Buddy

Writing Buddy is a Chrome extension that writes in your voice.

## How it works

1. **Ingest** — you upload writing samples (`.txt`, `.md`, `.pdf`, `.docx`). Buddy splits them into overlapping chunks so no stylistic thread gets cut mid-sentence.
2. **Embed** — each chunk is turned into a vector (via Gemini's embedding model) and stored in a FAISS index — Buddy's long-term memory of how you write.
3. **Retrieve** — when you ask for something, Buddy searches that index for the chunks of your own writing most relevant to the request.
4. **Generate** — those chunks are handed to Gemini as style examples, with instructions to imitate the *voice* (word choice, rhythm, tone) without copying the actual sentences or facts.

The result: a draft that sounds like it came from you, grounded in things you've actually written.

## Where it runs

- **Backend**: FastAPI + FAISS + Gemini, deployed on Render
- **Frontend**: a Manifest V3 Chrome extension — select text on any page, ask Buddy to rewrite or continue it, review the draft, insert it
- **Auth model**: BYOK (Bring Your Own Key) — your Gemini API key stays in your browser and pays for your own requests; your writing samples are isolated to you

## Roadmap

- Direct write-back into text (agent acts inside where you write, not just copy/paste)
- OAuth for cross-device identity (separate from BYOK, which handles billing)

## Run locally

```bash
pip install -r requirements.txt
python main.py            # http://127.0.0.1:8000 (docs at /docs)
```

Production (Render) runs `uvicorn api:app --host 0.0.0.0 --port $PORT`.

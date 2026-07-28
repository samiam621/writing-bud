# writing-bud

FastAPI backend for the Writing Buddy Chrome extension. It stores samples of
a user's writing as embeddings (FAISS) and uses Gemini to generate new text
in their voice.

## Run locally

```bash
pip install -r requirements.txt
python main.py            # http://127.0.0.1:8000 (docs at /docs)
```

Production (Render) runs `uvicorn api:app --host 0.0.0.0 --port $PORT`.

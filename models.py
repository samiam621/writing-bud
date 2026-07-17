"""
Pydantic models describing what goes IN and OUT of our API endpoints.

Keeping these separate from main.py means FastAPI's auto-generated docs
(visit /docs once the server is running) stay clean, and you can reuse these
shapes elsewhere (e.g. in tests) without importing your whole app.
"""
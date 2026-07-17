"""
Ingestion layer: raw files -> clean text -> chunks.

This file only knows how to read files and split text. It does NOT know
about embeddings or Gemini. That separation matters: if tomorrow you add
support for .rtf files, you only touch this file.
"""

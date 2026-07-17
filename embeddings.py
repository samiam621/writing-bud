"""
Embeddings layer: turns text chunks into vectors and stores them in a FAISS
index so we can later ask "which of the user's past writing sounds most
relevant to this new prompt?"

This file owns the FAISS index and the parallel list of raw chunk text.
Nothing outside this file should touch the index directly - that keeps the
save/load logic in one place and stops bugs where the index and the text
list get out of sync.

Stores/searches by similarity
"""
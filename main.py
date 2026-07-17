
"""
main.py
=======
The entry point — the ONE file you run to start the whole backend:

    python main.py

Its only jobs are startup housekeeping and launching the web server. All the
actual routes live in api.py; all the logic lives in the other files. Keeping
this thin means "how do I start this thing?" always has an obvious answer.
"""

import uvicorn

# Load the saved writing memory ONCE, at startup, before the server accepts
# requests. Without this, the agent would forget every past upload each time
# you restart. memory.load() quietly does nothing if there's no saved file yet.
from embeddings import memory
memory.load()
print(f"Loaded memory: {memory.index.ntotal} chunks ready.")


if __name__ == "__main__":
    # "api:app" tells uvicorn: import the `app` object from api.py.
    # host 127.0.0.1 = local only (your machine). Change to "0.0.0.0" to expose
    #   it on your network once you're ready.
    # reload=True   = auto-restart when you edit a file. Great for development;
    #   turn it off in production.
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
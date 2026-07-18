
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

# NOTE: the saved writing memory is loaded by the lifespan hook in api.py,
# which runs no matter HOW the server is started — `python main.py` here, or
# `uvicorn api:app ...` on Render. (It used to be loaded in this file, which
# silently skipped it in production because Render never runs main.py.)

if __name__ == "__main__":
    # DEVELOPMENT entry point only.
    # "api:app" tells uvicorn: import the `app` object from api.py.
    # host 127.0.0.1 = local only (your machine).
    # reload=True   = auto-restart when you edit a file.
    #
    # In production (Render), don't use this file — set the start command to:
    #     uvicorn api:app --host 0.0.0.0 --port $PORT
    # (0.0.0.0 = accept outside connections; $PORT is assigned by Render.)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
"""Launch the app on REAL adapters for the real-mode load test. Backends default to
real (whisper / pymupdf / spacy); we only force GigaChat Lite to avoid the Pro 402.
Requires GIGACHAT_CREDENTIALS in .env and the [ml] extra installed.

    python experiments/serve_real.py
"""

from __future__ import annotations

import os

os.environ["GIGACHAT_MODEL"] = "GigaChat"  # Lite (Pro = 402); real credentials come from .env

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="info")

"""Launch the app on all-fake backends (no GPU / no live LLM) for the
orchestration-mode load test and for showing /metrics + Grafana without model
weights. Env is set in-process (Python keeps empty vars, unlike PowerShell) so the
Fake LLM is selected exactly like the test suite does.

    python experiments/serve_orchestration.py
"""

from __future__ import annotations

import os

os.environ["GIGACHAT_CREDENTIALS"] = ""  # -> FakeLlmAdapter (no live GigaChat / no 402)
os.environ["TRANSCRIBER_BACKEND"] = "fake"
os.environ["OCR_BACKEND"] = "fake"
os.environ["NER_BACKEND"] = "fake"
os.environ["DEMO_MODE"] = "true"
os.environ["CORPUS_PATH"] = "experiments/assets/empty_corpus"  # empty -> Fake embedding for F6

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="info")

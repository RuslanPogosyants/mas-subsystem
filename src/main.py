"""FastAPI application entry point.

In M0 only routing skeleton. ML models and background coroutines are
plugged in during M2-M3 via startup hooks.
"""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router as api_router

app = FastAPI(
    title="mas-subsystem",
    description="Multi-agent subsystem for intelligent processing of educational data",
    version="0.1.0",
)

app.include_router(api_router)


@app.get("/")
async def root() -> dict[str, str]:
    """Health check and pointer to interactive docs."""
    return {"service": "mas-subsystem", "docs": "/docs", "status": "ok"}

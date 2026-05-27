# mas-subsystem

Multi-agent subsystem for intelligent processing of heterogeneous educational and research data (audio, images, text). Prototype for an undergraduate thesis (VKR) in Applied Informatics, NCFU, 2026.

## Status

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Prototype under active development (M0 — M6). See implementation plans in `docs/`.

## Quick start

```powershell
docker compose up -d
uv venv --python 3.13
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
alembic upgrade head
pytest
```

## Architecture

Seven agents (TranscriberAgent, OCRAgent, SummarizerAgent, TestGeneratorAgent, TerminologyAgent, RecommenderAgent, CoordinatorAgent) communicate over Redis Streams under a FIPA-ACL-like protocol. See `ARCHITECTURE.md` (added in M4).

## Conventions

- Code rules: `docs/CODESTYLE.md`
- Commit rules: `CONTRIBUTING.md`

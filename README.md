# mas-subsystem

[![CI](https://github.com/RuslanPogosyants/mas-subsystem/actions/workflows/ci.yml/badge.svg)](https://github.com/RuslanPogosyants/mas-subsystem/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A multi-agent subsystem that converts heterogeneous educational materials — audio lectures, scanned PDFs, and images — into structured study artifacts: transcript/text chunks, a summary, a quiz, a terminology glossary, and related-paper citations.

Six specialised agents run concurrently inside a single `asyncio` process, coordinated by a readiness-DAG dispatcher that handles per-agent timeouts, fault isolation, non-blocking retry, and crash-recovery (in-flight tasks are rehydrated from the database on restart).

---

## Architecture

```
          ┌─────────────────── FastAPI  ──────────────────────┐
          │  POST /api/tasks   GET /api/tasks/{id}  /metrics  │
          └────────────────────────┬──────────────────────────┘
                                   │
                            Coordinator
                       (readiness-DAG dispatch)
                      ┌────────────────────────┐
                      │  F1  F2  ──►  F3  F4   │
                      │  F1  F2  ──►  F5  ──►  F6 │
                      └────────────────────────┘
                                   │
                         Redis Streams bus
              ┌──────────┬──────────┬──────────┐
              F1          F2         F3 … F6
         Transcriber    OCR      (see below)
              │
          PostgreSQL  (7 tables, alembic migrations)
```

### Agents

| ID | Agent | Input | Output |
|----|-------|-------|--------|
| F1 | **TranscriberAgent** | audio file | text chunks |
| F2 | **OcrAgent** | PDF / image | text chunks |
| F3 | **SummarizerAgent** | text chunks | structured summary (intro, key points, conclusions) |
| F4 | **TestGeneratorAgent** | summary | quiz (question/answer pairs) |
| F5 | **TerminologyAgent** | text chunks | terminology glossary (NER) |
| F6 | **RecommenderAgent** | summary + glossary | related-paper citations (embedding similarity) |

**DAG:** F3 and F5 depend on F1/F2; F4 depends on F3; F6 depends on F3+F5.

All agents subclass `AgentBase`: Redis Streams consumer-group loop, idempotent message handling, graceful refuse-on-error.

### ML adapters

Every model sits behind a Protocol with a deterministic in-process **Fake** and a lazily-imported real backend. The system defaults to the **real** backends at runtime (Whisper/PyMuPDF/spaCy); the Fakes are the test-only mechanism. To run without model weights or a GPU, set the three env vars to `fake` (see the Configuration table below). The LLM (GigaChat) uses the real adapter when `GIGACHAT_CREDENTIALS` is set and falls back to the Fake automatically when unset. The test suite always runs fully offline — fakes are forced in `tests/conftest.py`.

| Backend | Agent | Real implementation | Default | Fake env override |
|---------|-------|---------------------|---------|-------------------|
| faster-whisper | F1 | `WhisperTranscriberAdapter` | **real** | `TRANSCRIBER_BACKEND=fake` |
| PyMuPDF + EasyOCR | F2 | `PymupdfOcrAdapter` | **real** | `OCR_BACKEND=fake` |
| GigaChat | F3, F4 | `GigaChatAdapter` | real if credentials set | `GIGACHAT_CREDENTIALS=` (unset) |
| spaCy `ru_core_news_lg` | F5 | `SpacyNerAdapter` | **real** | `NER_BACKEND=fake` |
| sentence-transformers `multilingual-e5-base` | F6 | `SentenceTransformerEmbeddingAdapter` | real if corpus present | no corpus files |

---

## Tech stack

- **Python 3.13**, asyncio
- **FastAPI** + **uvicorn** (REST API)
- **Redis Streams** (message bus)
- **PostgreSQL** via **SQLAlchemy 2 async** + **alembic** (3 migrations, 7 tables)
- **pydantic-settings** (configuration)
- **Prometheus** (`/metrics`) + **Grafana** (dashboard provisioned via code)
- **loguru** (structured logging)
- **ML extras** (`pip install ".[ml]"`): faster-whisper, EasyOCR, PyMuPDF, spaCy, sentence-transformers, GigaChat SDK

---

## Quickstart

### Prerequisites

- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- Docker (for Postgres, Redis, and optionally Grafana)

### 1. Clone and create the virtual environment

```bash
git clone https://github.com/RuslanPogosyants/mas-subsystem.git
cd mas-subsystem
uv venv --python 3.13
```

### 2. Install dependencies

Base install (library and API code only; to run real ML tasks install the `ml` extra):

```bash
uv pip install -e ".[dev]"
```

With real ML backends (Whisper, PyMuPDF/EasyOCR, spaCy, sentence-transformers):

```bash
uv pip install -e ".[dev,ml]"
```

### 3. Start infrastructure

```bash
docker compose up -d          # Postgres :5432, Redis :6379
```

### 4. Apply database migrations

```bash
alembic upgrade head
```

### 5. Run the app

```bash
uvicorn src.main:app --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 6. Submit a task

```bash
curl -X POST http://localhost:8000/api/tasks \
  -F "files=@lecture.mp3" \
  -F "ops=F1" \
  -F "ops=F3" \
  -F "ops=F4"
# ops are operation codes: F1 transcript, F3 summary, F4 quiz
# → {"task_id": "task-<id>", "status": "planning"}

curl http://localhost:8000/api/tasks/task-<id>
# → {"task_id": "...", "status": "completed"}

curl http://localhost:8000/api/tasks/task-<id>/result
```

Accepted `ops` values: `F1` (transcript), `F2` (OCR), `F3` (summary), `F4` (quiz), `F5` (terminology), `F6` (citations).
Accepted file types: audio (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`), PDF, image (`.jpg`, `.png`, …).

### 7. View metrics

```
http://localhost:8000/metrics
```

### Observability stack (optional)

```bash
docker compose -f deploy/docker-compose.observability.yml up -d
```

Prometheus scrapes `:8000/metrics`; Grafana serves the **MAS — Pipeline Overview** dashboard at `http://localhost:3000` (login `admin` / `admin`, or browse anonymously).

---

## Configuration

All settings are loaded from environment variables or a `.env` file (pydantic-settings). Key flags:

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | `postgresql+asyncpg://mas:mas@localhost:5432/mas_subsystem` | Postgres connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `GIGACHAT_CREDENTIALS` | _(empty)_ | Bearer token; unset → Fake LLM |
| `GIGACHAT_MODEL` | `GigaChat-Pro` | Model name for F3/F4 |
| `TRANSCRIBER_BACKEND` | `whisper` | `whisper` (real, default) or `fake` (no model weights) |
| `OCR_BACKEND` | `pymupdf` | `pymupdf` (real, default) or `fake` (no model weights) |
| `NER_BACKEND` | `spacy` | `spacy` (real, default) or `fake` (no model weights) |
| `CORPUS_PATH` | `corpus` | Directory with `papers.jsonl` + `papers.npy` for F6 |
| `DEMO_MODE` | `false` | When `true`, F6 falls back to a built-in demo corpus if no real corpus is present; off by default so F6 refuses gracefully in production |
| `COORD_TIMEOUT_TRANSCRIBER` | `600` | Per-agent deadline (seconds) |

---

## Testing

```bash
# Full suite
pytest

# With coverage
pytest --cov=src --cov-report=term-missing

# Exclude tests requiring real containers
pytest -m "not integration and not e2e"

# Only integration tests (needs Docker)
pytest -m integration

# Static analysis
ruff check src tests
mypy src
```

### Test markers

| Marker | Meaning |
|--------|---------|
| `integration` | Requires Redis / Postgres via testcontainers |
| `e2e` | Full pipeline acceptance tests |
| `perf` | Throughput guards |
| `slow` | Requires real ML models (excluded from CI) |

---

## Quality evaluation

Run the intrinsic + LLM-as-judge evaluation harness (requires `GIGACHAT_CREDENTIALS`):

```bash
python -m src.evaluation.run <task_id> --source path/to/source.txt --out report.md
```

Produces per-agent intrinsic scores plus LLM-as-judge ratings, including a student-usefulness assessment of the quiz and terminology.

---

## Project layout

```
src/
  agents/         # F1–F6 agents, Coordinator, AgentBase, recovery
  adapters/       # ML adapter Protocols + Fake/real implementations
  api/            # FastAPI routes
  core/           # Redis bus, schemas, metrics, logging
  db/             # SQLAlchemy models, repos, session
  config.py       # pydantic-settings Settings
  main.py         # app wiring, lifespan, adapter selectors
  evaluation/     # quality harness (intrinsic + LLM-as-judge)
migrations/       # alembic migration scripts
deploy/           # docker-compose.observability.yml, Prometheus config, Grafana provisioning
tests/            # unit / contracts / integration / e2e / perf
```

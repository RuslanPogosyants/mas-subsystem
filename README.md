# mas-subsystem

Multi-agent subsystem for intelligent processing of heterogeneous educational and research data (audio, images, text). Prototype for an undergraduate thesis (VKR) in Applied Informatics, NCFU, 2026.

## Status

![CI](https://github.com/RuslanPogosyants/mas-subsystem/actions/workflows/ci.yml/badge.svg)
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

## Test status

| Layer | Green | Red | xfail | Green target |
|---|---|---|---|---|
| contracts | yes | rest_contract | — | M0 + M2 (REST routes) |
| unit | yes | — | — | M1 (algorithm implementation) |
| e2e | — | yes | yes | M2-M4 (pipeline implementation) |

Run locally:

```powershell
pytest -v
```

Run only the M0 green baseline:

```powershell
pytest tests\contracts -v
```

## Architecture

Seven agents (TranscriberAgent, OCRAgent, SummarizerAgent, TestGeneratorAgent, TerminologyAgent, RecommenderAgent, CoordinatorAgent) communicate over Redis Streams under a FIPA-ACL-like protocol. See `ARCHITECTURE.md` (added in M4).

## Observability

Prometheus scrapes `/metrics` (exposed by the app on `:8000`); Grafana serves the dashboard at `:3000`.

```powershell
docker compose -f deploy/docker-compose.observability.yml up -d
```

The `MAS — Pipeline Overview` dashboard is provisioned automatically on Grafana start (login `admin` / `admin`, or browse anonymously).

## Conventions

- Code style: [`docs/CODESTYLE.md`](docs/CODESTYLE.md)
- Commits and contributing: [`CONTRIBUTING.md`](CONTRIBUTING.md)

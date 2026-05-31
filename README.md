# mas-research

[![CI](https://github.com/RuslanPogosyants/mas-research/actions/workflows/ci.yml/badge.svg)](https://github.com/RuslanPogosyants/mas-research/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Русский** · [English version below](#english-version)

Мультиагентная подсистема, которая превращает разнородные учебные материалы — аудиолекции, сканированные PDF и изображения — в структурированные учебные артефакты: текстовые фрагменты/транскрипт, конспект, проверочный тест, глоссарий терминов и список рекомендованных источников.

Шесть специализированных агентов работают параллельно в одном `asyncio`-процессе под управлением диспетчера на основе DAG-готовности, который обеспечивает пер-агентные таймауты, изоляцию сбоев, неблокирующий повтор и восстановление после краха (незавершённые задачи восстанавливаются из базы данных при перезапуске).

---

## Статус и инженерные результаты

- **Функционально завершённый прототип**, провалидированный сквозным прогоном на реальных моделях: Whisper large-v3 (GPU), GigaChat, PyMuPDF/EasyOCR, spaCy `ru_core_news_lg`, эмбеддинги multilingual-e5.
- **Гарантии качества:** 333 автотеста (модульные / контрактные / интеграционные / сквозные / нагрузочные), покрытие 94%, `mypy --strict` и `ruff` без замечаний на каждом коммите.
- **Устойчивость:** восстановление после краха (незавершённые задачи перечитываются из PostgreSQL при перезапуске), мягкая деградация (при отказе одного компонента задача возвращает частичный результат вместо полного провала), ограничители времени на агента и на задачу, идемпотентная обработка сообщений.
- **Наблюдаемость:** метрики Prometheus на `/metrics`, структурное логирование, преднастроенный дашборд Grafana.
- **Измеренная производительность:** батч-инференс Whisper транскрибирует ~25-минутную лекцию примерно за 78 секунд на ноутбучной GPU с 8 ГБ — около 2.8x быстрее небатчевого; цикл диспетчера держит несколько тысяч переходов задач в секунду, то есть узкое место — ML-модели, а не оркестрация.
- **Дизайн под тестируемость:** каждая модель скрыта за типизированным `Protocol` с детерминированной in-process «заглушкой» (Fake), поэтому вся система запускается и полностью тестируется офлайн, а в реальном режиме по умолчанию используются настоящие бэкенды.

---

## Архитектура

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
         Transcriber    OCR      (см. ниже)
              │
          PostgreSQL  (7 таблиц, миграции alembic)
```

### Агенты

| ID | Агент | Вход | Выход |
|----|-------|------|-------|
| F1 | **TranscriberAgent** | аудиофайл | текстовые фрагменты |
| F2 | **OcrAgent** | PDF / изображение | текстовые фрагменты |
| F3 | **SummarizerAgent** | текстовые фрагменты | структурированный конспект (введение, тезисы, выводы) |
| F4 | **TestGeneratorAgent** | конспект | проверочный тест (вопрос/ответ) |
| F5 | **TerminologyAgent** | текстовые фрагменты | глоссарий терминов (NER) |
| F6 | **RecommenderAgent** | конспект + глоссарий | рекомендованные статьи (по близости эмбеддингов) |

**DAG зависимостей:** F3 и F5 зависят от F1/F2; F4 зависит от F3; F6 зависит от F3+F5.

Все агенты наследуют `AgentBase`: цикл consumer-group поверх Redis Streams, идемпотентная обработка, мягкий отказ при ошибке.

### ML-адаптеры

Каждая модель скрыта за `Protocol` с детерминированной in-process заглушкой (Fake) и лениво импортируемым реальным бэкендом. По умолчанию в рантайме используются **реальные** бэкенды (Whisper/PyMuPDF/spaCy); заглушки — это механизм только для тестов. Чтобы запустить без весов моделей и GPU, выставьте три переменные окружения в `fake` (см. таблицу «Конфигурация»). LLM (GigaChat) использует реальный адаптер при заданном `GIGACHAT_CREDENTIALS` и автоматически откатывается на Fake, если он не задан. Тестовый набор всегда исполняется офлайн — заглушки форсируются в `tests/conftest.py`.

| Бэкенд | Агент | Реальная реализация | По умолчанию | Переключение на Fake |
|--------|-------|---------------------|--------------|----------------------|
| faster-whisper | F1 | `WhisperTranscriberAdapter` | **реальный** | `TRANSCRIBER_BACKEND=fake` |
| PyMuPDF + EasyOCR | F2 | `PymupdfOcrAdapter` | **реальный** | `OCR_BACKEND=fake` |
| GigaChat | F3, F4 | `GigaChatAdapter` | реальный при заданных ключах | `GIGACHAT_CREDENTIALS=` (пусто) |
| spaCy `ru_core_news_lg` | F5 | `SpacyNerAdapter` | **реальный** | `NER_BACKEND=fake` |
| sentence-transformers `multilingual-e5-base` | F6 | `SentenceTransformerEmbeddingAdapter` | реальный при наличии корпуса | нет файлов корпуса |

---

## Стек технологий

- **Python 3.13**, asyncio
- **FastAPI** + **uvicorn** (REST API)
- **Redis Streams** (шина сообщений)
- **PostgreSQL** через **SQLAlchemy 2 async** + **alembic** (3 миграции, 7 таблиц)
- **pydantic-settings** (конфигурация)
- **Prometheus** (`/metrics`) + **Grafana** (дашборд как код)
- **loguru** (структурное логирование)
- **ML-расширения** (`pip install ".[ml]"`): faster-whisper, EasyOCR, PyMuPDF, spaCy, sentence-transformers, GigaChat SDK

---

## Быстрый старт

### Требования
- Python 3.13
- [uv](https://github.com/astral-sh/uv)
- Docker (для Postgres, Redis и опционально Grafana)

### 1. Клонирование и виртуальное окружение
```bash
git clone https://github.com/RuslanPogosyants/mas-research.git
cd mas-research
uv venv --python 3.13
```

### 2. Установка зависимостей
Базовая установка (код библиотеки и API; для реальных ML-задач добавьте расширение `ml`):
```bash
uv pip install -e ".[dev]"
```
С реальными ML-бэкендами (Whisper, PyMuPDF/EasyOCR, spaCy, sentence-transformers):
```bash
uv pip install -e ".[dev,ml]"
```

### 3. Запуск инфраструктуры
```bash
docker compose up -d          # Postgres :5432, Redis :6379
```

### 4. Применение миграций
```bash
alembic upgrade head
```

### 5. Запуск приложения
```bash
uvicorn src.main:app --reload
```
API доступен на `http://localhost:8000`, интерактивная документация — `http://localhost:8000/docs`.

> **Базовая установка (без ML-моделей):** установка `.[dev]` без расширения `ml` использует реальные бэкенды по умолчанию, что приведёт к `ModuleNotFoundError` в момент выполнения задачи. Либо поставьте `.[dev,ml]` и предоставьте веса/ключи, либо переключите бэкенды:
> ```bash
> TRANSCRIBER_BACKEND=fake OCR_BACKEND=fake NER_BACKEND=fake uvicorn src.main:app --reload
> ```
> Оставьте `GIGACHAT_CREDENTIALS` пустым, чтобы LLM работал на in-process заглушке.

### 6. Отправка задачи
```bash
curl -X POST http://localhost:8000/api/tasks \
  -F "files=@lecture.mp3" \
  -F "ops=F1" \
  -F "ops=F3" \
  -F "ops=F4"
# ops — коды операций: F1 транскрипт, F3 конспект, F4 тест
# → {"task_id": "task-<id>", "status": "planning"}

curl http://localhost:8000/api/tasks/task-<id>
# → {"task_id": "...", "status": "completed"}

curl http://localhost:8000/api/tasks/task-<id>/result
```
Допустимые значения `ops`: `F1` (транскрипт), `F2` (OCR), `F3` (конспект), `F4` (тест), `F5` (термины), `F6` (источники).
Допустимые типы файлов: аудио (`.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`), PDF, изображения (`.jpg`, `.png`, …).

### 7. Метрики
```
http://localhost:8000/metrics
```

### Стек наблюдаемости (опционально)
```bash
docker compose -f deploy/docker-compose.observability.yml up -d
```
Prometheus снимает `:8000/metrics`; Grafana отдаёт дашборд **MAS — Pipeline Overview** на `http://localhost:3000` (логин `admin` / `admin`, либо анонимно).

---

## Конфигурация

Все настройки читаются из переменных окружения или файла `.env` (pydantic-settings). Ключевые флаги:

| Переменная | По умолчанию | Назначение |
|------------|--------------|------------|
| `DATABASE_URL` | `postgresql+asyncpg://mas:mas@localhost:5432/mas_subsystem` | Подключение к Postgres |
| `REDIS_URL` | `redis://localhost:6379/0` | Подключение к Redis |
| `GIGACHAT_CREDENTIALS` | _(пусто)_ | Токен; пусто → Fake LLM |
| `GIGACHAT_MODEL` | `GigaChat-Pro` | Модель для F3/F4 |
| `TRANSCRIBER_BACKEND` | `whisper` | `whisper` (реальный) или `fake` |
| `OCR_BACKEND` | `pymupdf` | `pymupdf` (реальный) или `fake` |
| `NER_BACKEND` | `spacy` | `spacy` (реальный) или `fake` |
| `CORPUS_PATH` | `corpus` | Каталог с `papers.jsonl` + `papers.npy` для F6 |
| `DEMO_MODE` | `false` | При `true` F6 использует встроенный демо-корпус, если нет реального; по умолчанию выключено, чтобы F6 корректно отказывал в проде |
| `COORD_TIMEOUT_TRANSCRIBER` | `600` | Дедлайн агента (секунды) |

---

## Тестирование
```bash
# Полный набор
pytest

# С покрытием
pytest --cov=src --cov-report=term-missing

# Без тестов, требующих контейнеров
pytest -m "not integration and not e2e"

# Только интеграционные (нужен Docker)
pytest -m integration

# Статический анализ
ruff check src tests
mypy src
```

| Маркер | Значение |
|--------|----------|
| `integration` | Требует Redis / Postgres (testcontainers) |
| `e2e` | Сквозные приёмочные тесты |
| `perf` | Гарды пропускной способности |
| `slow` | Требует реальные ML-модели (исключены из CI) |

---

## Оценка качества
Харнесс intrinsic-метрик + LLM-as-judge (требует `GIGACHAT_CREDENTIALS`):
```bash
python -m src.evaluation.run <task_id> --source path/to/source.txt --out report.md
```
Выдаёт intrinsic-оценки по каждой функции плюс оценки LLM-судьи, включая оценку полезности теста и терминов для студента.

---

## Структура проекта
```
src/
  agents/         # агенты F1–F6, Coordinator, AgentBase, recovery
  adapters/       # Protocol'ы ML-адаптеров + Fake/реальные реализации
  api/            # маршруты FastAPI
  core/           # шина Redis, схемы, метрики, логирование
  db/             # модели SQLAlchemy, репозитории, сессия
  config.py       # настройки pydantic-settings
  main.py         # сборка приложения, lifespan, выбор адаптеров
  evaluation/     # харнесс качества (intrinsic + LLM-as-judge)
migrations/       # миграции alembic
deploy/           # docker-compose наблюдаемости, конфиг Prometheus, провижининг Grafana
tests/            # unit / contracts / integration / e2e / perf
```

---
<a id="english-version"></a>

## English version

A multi-agent subsystem that converts heterogeneous educational materials — audio lectures, scanned PDFs, and images — into structured study artifacts: transcript/text chunks, a summary, a quiz, a terminology glossary, and related-paper citations. Six specialised agents run concurrently inside a single `asyncio` process, coordinated by a readiness-DAG dispatcher that handles per-agent timeouts, fault isolation, non-blocking retry, and crash-recovery (in-flight tasks are rehydrated from the database on restart).

### Engineering highlights
- **Functionally complete prototype**, validated end-to-end on real models: Whisper large-v3 (GPU), GigaChat, PyMuPDF/EasyOCR, spaCy `ru_core_news_lg`, and multilingual-e5 embeddings.
- **Quality gates:** 333 automated tests (unit / contract / integration / end-to-end / performance), 94% line coverage, `mypy --strict` and `ruff` clean on every commit.
- **Resilience:** crash-recovery (in-flight tasks rehydrated from PostgreSQL on restart), graceful degradation (partial results instead of failure when one component declines), bounded per-agent and per-task deadlines, idempotent message handling.
- **Observability:** Prometheus metrics at `/metrics`, structured logging, a provisioned Grafana dashboard.
- **Measured performance:** batched Whisper inference transcribes a ~25-minute lecture in roughly 78 seconds on an 8 GB laptop GPU — about 2.8x faster than non-batched; the dispatch loop sustains several thousand task transitions per second, so the ML models, not the orchestration, are the throughput bound.
- **Design for testability:** every model sits behind a typed `Protocol` with a deterministic in-process fake, so the whole system runs and is fully tested offline while defaulting to the real backends at runtime.

### Agents
F1 `TranscriberAgent` (audio → text chunks), F2 `OcrAgent` (PDF/image → text chunks), F3 `SummarizerAgent` (chunks → structured summary), F4 `TestGeneratorAgent` (summary → quiz), F5 `TerminologyAgent` (chunks → glossary, NER), F6 `RecommenderAgent` (summary + glossary → related-paper citations). DAG: F3/F5 depend on F1/F2; F4 on F3; F6 on F3+F5.

### ML adapters
Every model sits behind a `Protocol` with a deterministic in-process **Fake** and a lazily-imported real backend. The system defaults to the **real** backends at runtime (Whisper/PyMuPDF/spaCy); the Fakes are the test-only mechanism. To run without model weights or a GPU, set `TRANSCRIBER_BACKEND=fake OCR_BACKEND=fake NER_BACKEND=fake` and leave `GIGACHAT_CREDENTIALS` unset. The test suite always runs fully offline (fakes are forced in `tests/conftest.py`).

### Quickstart
```bash
git clone https://github.com/RuslanPogosyants/mas-research.git
cd mas-research
uv venv --python 3.13
uv pip install -e ".[dev,ml]"     # or ".[dev]" + *_BACKEND=fake to run model-free
docker compose up -d              # Postgres + Redis
alembic upgrade head
uvicorn src.main:app --reload     # API at http://localhost:8000, docs at /docs
```
Submit a task: `POST /api/tasks` with `files=@lecture.mp3` and `ops=F1,F3,F4` (codes F1–F6). Poll `GET /api/tasks/{id}` and fetch `GET /api/tasks/{id}/result`. Metrics at `/metrics`.

### Tech stack
Python 3.13 · asyncio · FastAPI/uvicorn · Redis Streams · PostgreSQL (SQLAlchemy 2 async + alembic) · pydantic-settings · Prometheus + Grafana · loguru · optional ML extras (faster-whisper, EasyOCR, PyMuPDF, spaCy, sentence-transformers, GigaChat SDK).

### Testing
`pytest` (full suite), `pytest --cov=src`, `ruff check src tests`, `mypy src`. Markers: `integration` (Redis/Postgres via testcontainers), `e2e`, `perf`, `slow` (real ML models, excluded from CI).

### Quality evaluation
`python -m src.evaluation.run <task_id> --source path/to/source.txt --out report.md` — intrinsic metrics plus LLM-as-judge ratings, including a student-usefulness assessment of the quiz and terminology.

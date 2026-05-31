# Эмпирическая оценка качества и производительности

Каталог содержит воспроизводимые эксперименты для количественной оценки подсистемы
на **публичных русскоязычных бенчмарках** и под **нагрузкой**. Метрики считаются на
**реальных** адаптерах (не Fake): транскрипция — Whisper large-v3, конспект —
GigaChat, термины — spaCy. Fake-адаптеры остаются только как механизм для офлайн-CI
и юнит-тестов; эксперименты используют реальные модели.

## Состав

| Скрипт | Что измеряет | Датасет / нагрузка |
|---|---|---|
| `bench_wer.py` | WER транскрипции (F1, Whisper) | FLEURS `ru_ru` (test, первые 50) |
| `bench_rouge.py` | ROUGE конспекта (F3, GigaChat Lite) | Gazeta (test, первые 20) |
| `bench_terms.py` | Precision/Recall/F1 терминов (F5, spaCy) | Gazeta (test, первые 15) |
| `loadtest.py` + `serve_orchestration.py` | p50/p95/p99 + насыщение, режим оркестрации (fake) | синтетическая PDF-задача |
| `loadtest_real.py` + `serve_real.py` | p50/p95/p99 + насыщение, реальный режим | реальный PDF, ops F2+F3+F5 |

Библиотека метрик: `src/evaluation/{wer,rouge,term_prf}.py` (покрыта юнит-тестами
`tests/unit/test_eval_metrics.py`). Результаты — в `experiments/results/`.

## Требования

- Python 3.13, менеджер пакетов `uv`.
- Запущенные хранилища для нагрузочных тестов: `docker compose up -d postgres redis prometheus grafana`.
- Для ROUGE и реального нагрузочного теста — `GIGACHAT_CREDENTIALS` в `.env`.
- Для WER — GPU (Whisper large-v3 через CTranslate2; на 8 ГБ VRAM используется `int8_float16`) и **`ffmpeg` в PATH** (декодирование аудио).
- Версии зафиксированы в `uv.lock` (179 пакетов) → `uv sync` ставит точные версии; таблица ниже — снимок ключевых.

```bash
uv sync --extra ml --extra dev --extra eval   # ставит точные версии из uv.lock
uv run python -m spacy download ru_core_news_lg
uv run python -m spacy download en_core_web_sm   # для NER англоязычных документов
```

## Зафиксированные версии (на момент прогона)

| Компонент | Версия |
|---|---|
| Python | 3.13.5 |
| faster-whisper | 1.2.1 (модель Whisper large-v3) |
| spaCy | 3.8.14 (`ru_core_news_lg` 3.8.0, `en_core_web_sm` 3.8.0) |
| sentence-transformers | 3.4.1 (`intfloat/multilingual-e5-base`) |
| gigachat | 0.2.1 (модель **GigaChat Lite**) |
| jiwer | 4.0.0 |
| rouge-score | 0.1.2 (+ Unicode-токенизатор, см. `src/evaluation/rouge.py`) |
| datasets | 4.8.5 |
| torch | 2.12.0 (CPU; Whisper использует GPU через CTranslate2) |
| GPU | NVIDIA RTX 4060 Laptop, 8 ГБ, драйвер 591.74 |

## Детерминизм и воспроизводимость

- Все бенчмарки берут **первые N** примеров из split (без шифла) — фиксированная выборка.
- Whisper и spaCy детерминированы → WER и метрики терминов воспроизводятся точно.
- **GigaChat недетерминирован** (генеративная LLM): ROUGE и реальные латентности F3
  варьируются от прогона к прогону в пределах полосы; приведён один зафиксированный
  прогон. Для строгой воспроизводимости ROUGE сохранены тексты гипотез F3 в
  `results/rouge-results.json` (поле `per_sample[].hypothesis`).

## Команды и полученные числа

```bash
# 1. WER транскрипции (нужны GPU + ffmpeg)
uv run python experiments/bench_wer.py
#   -> corpus_wer=0.052; mean=0.057 ± 0.082, 95% ДИ [0.034, 0.080]; median=0.040
#      (50 клипов FLEURS ru_ru, Whisper large-v3)

# 2. ROUGE конспекта (нужен GIGACHAT_CREDENTIALS)
uv run python experiments/bench_rouge.py
#   -> ROUGE (лемматизир., macro-avg) R1/R2/RL = 0.280/0.100/0.173; raw R1=0.214
#      R1 = 0.280 ± 0.073, 95% ДИ [0.245, 0.316]; 16/20 успешно (Lite, zero-shot)
#      трактовать как нижнюю границу (рассогласование формата), не как соревнование с SOTA

# 3. Precision/Recall/F1 терминов
uv run python experiments/bench_terms.py
#   -> macro P/R/F1 = 0.279/0.193/0.224; F1 = 0.224 ± 0.081, 95% ДИ [0.183, 0.264]
#      нижняя граница по лексическому пересечению (прокси-эталон из человеческих конспектов)

# 4. Нагрузка, режим оркестрации (fake-бэкенды)
uv run python experiments/serve_orchestration.py   # терминал 1
uv run python experiments/loadtest.py              # терминал 2
#   -> насыщение ~8 параллельных, пик ~17 задач/с, субсекундная латентность

# 5. Нагрузка, реальный режим (реальные адаптеры + GigaChat Lite)
uv run python experiments/serve_real.py            # терминал 1
uv run python experiments/loadtest_real.py         # терминал 2 (можно передать путь к PDF)
#   -> насыщение ~2 параллельных; латентность определяется F3/GigaChat (~3.3 с/документ)
```

## Оговорки

- ROUGE против Gazeta — индикативен: F3 выдаёт **структурированный** конспект
  (введение/тезисы/выводы), эталон — свободный абстракт; рассогласование формата
  занижает ROUGE независимо от качества. Лемматизированный вариант сравним с
  литературой (статья по Gazeta репортует ROUGE по леммам).
- Стандартного публичного RU-бенчмарка по извлечению терминов нет; эталон для F5 —
  прокси из человеческих конспектов Gazeta (см. `bench_terms.py`).
- Нагрузочные тесты — один прогон на одной локальной машине (клиент и сервер делят CPU).

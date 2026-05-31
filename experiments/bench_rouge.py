"""ROUGE benchmark: real F3 summarizer (GigaChat Lite) on Gazeta, vs reference summaries.

Reuses the real SummarizerAgent map-reduce logic. Writes
experiments/results/rouge-results.json. NB: F3 emits a structured
intro/thesis/conclusion summary; ROUGE vs free-form news abstracts is indicative
(format mismatch lowers ROUGE independent of quality) and is best read as a
lower-bound / sanity floor rather than a head-to-head with fine-tuned SOTA.
corpus_rouge is a macro-average of per-pair F-measures (not a true corpus metric).

Reports BOTH raw ROUGE and lemmatized ROUGE (spaCy ru_core_news_lg). The Gazeta
paper (Gusev, 2020) reports lemmatized ROUGE, because Russian inflection makes raw
token overlap systematically understate quality; the lemmatized variant is the
literature-comparable number.

Run from the repository root (requires GIGACHAT_CREDENTIALS in .env, [ml] extra):
    python experiments/bench_rouge.py
"""

from __future__ import annotations

import asyncio
import json
import os

os.environ["GIGACHAT_MODEL"] = "GigaChat"  # Lite (Pro = 402)

from pathlib import Path

import spacy
from datasets import load_dataset
from src.adapters.gigachat import GigaChatAdapter
from src.agents.summarizer import SummarizerAgent
from src.config import get_settings
from src.evaluation.rouge import corpus_rouge, rouge_scores
from tests.support.fake_bus import FakeBus

_N = 20
_OUT = Path("experiments/results/rouge-results.json")
_NLP = spacy.load("ru_core_news_lg", disable=["parser", "ner"])


def _lemmatize(text: str) -> str:
    return " ".join(token.lemma_.lower() for token in _NLP(text) if not token.is_space)


async def main() -> None:  # noqa: PLR0915
    settings = get_settings()
    print(f"model={settings.gigachat_model} creds_set={bool(settings.gigachat_credentials)}", flush=True)
    agent = SummarizerAgent(bus=FakeBus(), llm=GigaChatAdapter(settings))
    ds = load_dataset("IlyaGusev/gazeta", split="test", streaming=True)
    raw_pairs: list[tuple[str, str]] = []
    lemma_pairs: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    skipped = 0
    iterator = iter(ds)
    for i in range(_N):
        sample = next(iterator)
        reference = str(sample["summary"])
        try:
            raw = await agent._summarize(str(sample["text"]))
        except Exception as error:
            skipped += 1
            rows.append({"i": i, "error": type(error).__name__})
            print(f"[{i + 1}/{_N}] ERROR {type(error).__name__}", flush=True)
            continue
        if raw is None:
            skipped += 1
            rows.append({"i": i, "skipped": True})
            print(f"[{i + 1}/{_N}] skipped (llm declined)", flush=True)
            continue
        hypothesis = " ".join([raw.introduction, raw.key_points, raw.conclusions])
        scores_raw = rouge_scores(reference, hypothesis)
        ref_lemma, hyp_lemma = _lemmatize(reference), _lemmatize(hypothesis)
        scores_lemma = rouge_scores(ref_lemma, hyp_lemma)
        raw_pairs.append((reference, hypothesis))
        lemma_pairs.append((ref_lemma, hyp_lemma))
        rows.append({"i": i, "raw": scores_raw, "lemmatized": scores_lemma, "hypothesis": hypothesis})
        print(f"[{i + 1}/{_N}] raw_r1={scores_raw['rouge1']} lemma_r1={scores_lemma['rouge1']}", flush=True)
    result = {
        "dataset": "IlyaGusev/gazeta test (first 20, deterministic)",
        "n": _N,
        "succeeded": len(raw_pairs),
        "skipped": skipped,
        "model": "GigaChat Lite via real F3 SummarizerAgent (map-reduce)",
        "corpus_rouge_raw": corpus_rouge(raw_pairs),
        "corpus_rouge_lemmatized": corpus_rouge(lemma_pairs),
        "caveat": (
            "F3 emits structured intro/thesis/conclusion; ROUGE vs free-form abstracts is indicative. "
            "Lemmatized ROUGE (spaCy) is the literature-comparable variant (Gazeta paper reports lemmatized)."
        ),
        "per_sample": rows,
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DONE raw={result['corpus_rouge_raw']} lemma={result['corpus_rouge_lemmatized']} "
        f"succeeded={len(raw_pairs)} skipped={skipped} -> {_OUT}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

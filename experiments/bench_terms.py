"""Term P/R/F1 benchmark: real F5 (spaCy NER + bigrams) on Gazeta articles vs a
proxy gold = content-noun lemmas from the human reference summary.

No standard RU term-extraction benchmark exists, so gold is derived from the
human-written reference summary (the terms a human deemed summary-worthy). Matching
is token-level (multi-word F5 terms split into lemma tokens) to avoid format
mismatch. Writes experiments/results/terms-results.json. spaCy-only (CPU), no GigaChat.

Run from the repository root:
    python experiments/bench_terms.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path

import spacy
from datasets import load_dataset
from src.adapters.spacy_ner import SpacyNerAdapter
from src.agents.terminology import TerminologyAgent
from src.evaluation.term_prf import term_prf
from tests.support.fake_bus import FakeBus

_N = 15
_MIN_LEN = 4
_TOP_N = 10
_OUT = Path("experiments/results/terms-results.json")


def _gold_terms(nlp: object, summary: str) -> list[str]:
    doc = nlp(summary)  # type: ignore[operator]
    return list(
        {t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN") and len(t.lemma_) >= _MIN_LEN and not t.is_stop}
    )


def _pred_tokens(term_lemmas: list[str]) -> list[str]:
    return list({tok for lemma in term_lemmas for tok in lemma.split() if len(tok) >= _MIN_LEN})


async def main() -> None:
    nlp = spacy.load("ru_core_news_lg")
    agent = TerminologyAgent(bus=FakeBus(), ner=SpacyNerAdapter())
    ds = load_dataset("IlyaGusev/gazeta", split="test", streaming=True)
    rows: list[dict[str, object]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    iterator = iter(ds)
    for i in range(_N):
        sample = next(iterator)
        gold = _gold_terms(nlp, str(sample["summary"]))
        terms = await agent._extract_terms([(f"c{i}", str(sample["text"]))], _TOP_N)
        pred = _pred_tokens([t.lemma for t in terms])
        scores = term_prf(pred, gold)
        rows.append({"i": i, "n_pred": len(pred), "n_gold": len(gold), **scores})
        precisions.append(scores["precision"])
        recalls.append(scores["recall"])
        f1s.append(scores["f1"])
        print(f"[{i + 1}/{_N}] P={scores['precision']} R={scores['recall']} F1={scores['f1']}", flush=True)
    result = {
        "dataset": "IlyaGusev/gazeta test (first 15, deterministic)",
        "n": _N,
        "gold": "content-noun (NOUN/PROPN) lemmas from human reference summary, len>=4, non-stop",
        "pred": "F5 top-10 term lemmas (multi-word split to tokens), len>=4",
        "matching": "token-level set P/R/F1",
        "macro_precision": round(statistics.mean(precisions), 4),
        "macro_recall": round(statistics.mean(recalls), 4),
        "macro_f1": round(statistics.mean(f1s), 4),
        "caveat": "No standard RU term-extraction benchmark; gold is a proxy from human reference summaries.",
        "per_sample": rows,
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE macro P/R/F1 = {result['macro_precision']}/{result['macro_recall']}/{result['macro_f1']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

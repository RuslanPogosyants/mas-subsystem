"""WER benchmark: real Whisper (F1 adapter) on FLEURS ru_ru, vs reference transcripts.

Deterministic first-N of the test split (no shuffle) for reproducibility. Writes
experiments/results/wer-results.json. Requires GPU + the [ml] and [eval] extras.
Audio is decoded by faster-whisper/CTranslate2, which needs **ffmpeg on PATH**.

Run from the repository root:
    python experiments/bench_wer.py
"""

from __future__ import annotations

import asyncio
import json
import statistics
import tempfile
from pathlib import Path

from datasets import Audio, load_dataset
from src.adapters.whisper_transcriber import WhisperTranscriberAdapter
from src.evaluation.wer import corpus_wer, word_error_rate

_N = 50
_OUT = Path("experiments/results/wer-results.json")


async def main() -> None:
    ds = load_dataset("google/fleurs", "ru_ru", split="test", streaming=True).cast_column("audio", Audio(decode=False))
    adapter = WhisperTranscriberAdapter(model_size="large-v3", device="cuda", compute_type="int8_float16")
    pairs: list[tuple[str, str]] = []
    rows: list[dict[str, object]] = []
    iterator = iter(ds)
    for i in range(_N):
        sample = next(iterator)
        reference = str(sample["transcription"])
        audio_bytes = sample["audio"]["bytes"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(audio_bytes)
            path = handle.name
        try:
            chunks = await adapter.transcribe(file_path=path, language="ru")
        finally:
            Path(path).unlink(missing_ok=True)
        hypothesis = " ".join(chunk.content for chunk in chunks)
        wer = word_error_rate(reference, hypothesis)
        pairs.append((reference, hypothesis))
        rows.append({"i": i, "wer": round(wer, 4), "ref_words": len(reference.split())})
        print(f"[{i + 1}/{_N}] WER={wer:.3f}", flush=True)
    result = {
        "dataset": "google/fleurs ru_ru test (first 50, deterministic)",
        "n": _N,
        "model": "Whisper large-v3, int8_float16, CUDA, vad_filter+batched",
        "corpus_wer": round(corpus_wer(pairs), 4),
        "mean_wer": round(statistics.mean(float(r["wer"]) for r in rows), 4),
        "median_wer": round(statistics.median(float(r["wer"]) for r in rows), 4),
        "per_sample": rows,
    }
    _OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"DONE corpus_wer={result['corpus_wer']} mean_wer={result['mean_wer']} -> {_OUT}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())

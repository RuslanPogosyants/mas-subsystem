"""Set-based term Precision / Recall / F1 metric.

Terms are normalised (lowercased and stripped) before comparison, and
duplicates within each list are collapsed to a set.

Edge-case conventions:
- Empty ``predicted`` → precision = 0.0, recall = 0.0, f1 = 0.0.
- Empty ``gold``      → precision = 0.0, recall = 0.0, f1 = 0.0.
- Both empty          → all 0.0.

Rationale: when there are no predictions there is no meaningful precision,
and when there is no gold standard there is no meaningful recall; F1 is
therefore 0.0 in both degenerate cases to avoid misleading scores.

All returned values are rounded to 4 decimal places.
"""

from __future__ import annotations

_ZERO_RESULT: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def _normalize_term(term: str) -> str:
    return term.strip().lower()


def term_prf(predicted: list[str], gold: list[str]) -> dict[str, float]:
    """Set-based Precision, Recall, F1 for predicted vs. gold term lists.

    Args:
        predicted: Terms predicted by the system (may contain duplicates).
        gold: Ground-truth terms (may contain duplicates).

    Returns:
        ``{"precision": p, "recall": r, "f1": f}`` with values in [0, 1]
        rounded to 4 decimal places.
    """
    pred_set = {_normalize_term(t) for t in predicted}
    gold_set = {_normalize_term(t) for t in gold}

    if not pred_set or not gold_set:
        return dict(_ZERO_RESULT)

    intersection_count = len(pred_set & gold_set)
    precision = intersection_count / len(pred_set)
    recall = intersection_count / len(gold_set)

    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }

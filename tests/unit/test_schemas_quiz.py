"""Well-formedness predicate for QuizQuestion."""

from __future__ import annotations

import pytest
from src.core.schemas import QuizQuestion


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"question": "q", "type": "open_answer"}, True),
        ({"question": "  ", "type": "open_answer"}, False),
        ({"question": "q", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 1}, True),
        ({"question": "q", "type": "single_choice", "choices": ["A", "B"], "answer_idx": None}, False),
        ({"question": "q", "type": "single_choice", "choices": ["A", "B"], "answer_idx": 5}, False),
        ({"question": "q", "type": "single_choice", "choices": ["A"], "answer_idx": 0}, False),
        ({"question": "q", "type": "multi_choice", "choices": ["A", "B"], "answer_indices": [0, 1]}, True),
        ({"question": "q", "type": "multi_choice", "choices": ["A", "B"], "answer_indices": []}, False),
        ({"question": "q", "type": "multi_choice", "choices": ["A", "B"], "answer_indices": [2]}, False),
    ],
)
def test_is_well_formed(kwargs: dict, expected: bool) -> None:
    assert QuizQuestion(**kwargs).is_well_formed() is expected

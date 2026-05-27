"""Contract tests for ResultArtifact (VKR listing 2.5 plus S8 partial)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.core.schemas import (
    Citation,
    FailedOperation,
    Operation,
    QuizQuestion,
    ResultArtifact,
    ResultPayload,
    ResultStats,
    Summary,
    SummarySection,
    TaskStatus,
    Term,
)


def _make_summary() -> Summary:
    return Summary(
        summary_id="sum-7c41",
        sections=[
            SummarySection(type="introduction", text="Sorting..."),
            SummarySection(type="thesis", text="Algorithms split..."),
            SummarySection(type="conclusion", text="Choice depends..."),
        ],
    )


def _make_full_payload() -> ResultPayload:
    return ResultPayload(
        summary=_make_summary(),
        terms=[Term(term="algorithm", lemma="algorithm", frequency=9, category="method")],
        quiz=[
            QuizQuestion(
                question="Bubble sort worst case complexity?",
                type="single_choice",
                choices=["O(n^2)", "O(n log n)", "O(n)"],
                answer_idx=0,
            )
        ],
        citations=[Citation(title="Algorithms", year=2013, relevance_score=0.89)],
    )


class TestHappyPathArtifact:
    def test_completed_artifact_has_all_operations(self) -> None:
        artifact = ResultArtifact(
            task_id="task-7c41",
            status=TaskStatus.COMPLETED,
            operations=list(Operation),
            result=_make_full_payload(),
            stats=ResultStats(duration_sec=187.0, agents_called=7, messages_exchanged=14),
        )
        assert artifact.status == TaskStatus.COMPLETED
        assert len(artifact.operations) == 6
        assert artifact.degraded == []
        assert artifact.stats.failed_operations == []

    def test_completed_artifact_stats_match_listing(self) -> None:
        artifact = ResultArtifact(
            task_id="task-7c41",
            status=TaskStatus.COMPLETED,
            operations=list(Operation),
            result=_make_full_payload(),
            stats=ResultStats(duration_sec=187.0, agents_called=7, messages_exchanged=14),
        )
        assert artifact.stats.duration_sec == 187.0
        assert artifact.stats.agents_called == 7
        assert artifact.stats.messages_exchanged == 14


class TestPartialArtifactS8:
    def test_partial_artifact_marks_degraded_f6(self) -> None:
        artifact = ResultArtifact(
            task_id="task-7c42",
            status=TaskStatus.PARTIAL_READY,
            operations=[
                Operation.F1_TRANSCRIBE,
                Operation.F2_OCR,
                Operation.F3_SUMMARIZE,
                Operation.F4_TEST,
                Operation.F5_TERMS,
            ],
            result=ResultPayload(
                summary=_make_summary(),
                terms=[Term(term="algorithm", lemma="algorithm", frequency=9, category="method")],
                quiz=[],
                citations=[],
            ),
            degraded=[Operation.F6_RECOMMEND],
            stats=ResultStats(
                duration_sec=169.0,
                agents_called=6,
                messages_exchanged=13,
                failed_operations=[
                    FailedOperation(
                        op=Operation.F6_RECOMMEND,
                        agent="RecommenderAgent",
                        reason="refuse: force_refuse flag enabled",
                        retries=2,
                        elapsed_sec=6.2,
                    )
                ],
            ),
        )
        assert artifact.status == TaskStatus.PARTIAL_READY
        assert Operation.F6_RECOMMEND not in artifact.operations
        assert artifact.degraded == [Operation.F6_RECOMMEND]
        assert artifact.result.citations == []
        assert len(artifact.stats.failed_operations) == 1
        assert artifact.stats.failed_operations[0].retries == 2

    def test_partial_artifact_summary_terms_quiz_still_present(self) -> None:
        artifact = ResultArtifact(
            task_id="task-7c42",
            status=TaskStatus.PARTIAL_READY,
            operations=[
                Operation.F1_TRANSCRIBE,
                Operation.F3_SUMMARIZE,
                Operation.F4_TEST,
                Operation.F5_TERMS,
            ],
            result=ResultPayload(
                summary=_make_summary(),
                terms=[Term(term="x", lemma="x", frequency=1, category="method")],
                quiz=[QuizQuestion(question="?", type="single_choice", choices=["a", "b"], answer_idx=0)],
                citations=[],
            ),
            degraded=[Operation.F6_RECOMMEND],
            stats=ResultStats(
                duration_sec=100.0,
                agents_called=5,
                messages_exchanged=10,
                failed_operations=[
                    FailedOperation(
                        op=Operation.F6_RECOMMEND,
                        agent="RecommenderAgent",
                        reason="timeout",
                        retries=2,
                        elapsed_sec=50.0,
                    )
                ],
            ),
        )
        assert artifact.result.summary is not None
        assert len(artifact.result.terms) >= 1
        assert len(artifact.result.quiz) >= 1


class TestArtifactExtraFieldsForbidden:
    def test_unknown_top_level_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ResultArtifact.model_validate(
                {
                    "task_id": "task-x",
                    "status": "completed",
                    "operations": [],
                    "result": {},
                    "stats": {
                        "duration_sec": 0,
                        "agents_called": 0,
                        "messages_exchanged": 0,
                    },
                    "bonus_field": "oops",
                }
            )

"""Unit test for the Markdown/JSON evaluation report builder."""

from __future__ import annotations

from src.evaluation.report import build_report


def test_build_report_contains_sections_and_scores() -> None:
    report = build_report(
        task_id="task-1",
        timing={"duration_sec": 12.3, "agents_called": 7},
        intrinsic={"summary": {"compression_ratio": 0.1}, "terms": {"count": 10}},
        judge={"summary": {"faithfulness": 4, "coverage": 5, "coherence": 4}, "quiz": None},
    )
    assert "task-1" in report.markdown
    assert "duration_sec" in report.markdown
    assert "faithfulness" in report.markdown
    assert report.data["task_id"] == "task-1"
    assert report.data["judge"]["summary"]["coverage"] == 5

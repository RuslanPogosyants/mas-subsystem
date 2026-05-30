"""Evaluation report builder: renders intrinsic + judge + timing to Markdown and JSON dict."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Report:
    markdown: str
    data: dict[str, Any]


def _render_dict(d: dict[str, Any], indent: int = 0) -> str:
    prefix = "  " * indent
    lines: list[str] = []
    for key in sorted(d):
        val = d[key]
        if isinstance(val, dict):
            lines.append(f"{prefix}- **{key}**:")
            lines.append(_render_dict(val, indent + 1))
        elif val is None:
            lines.append(f"{prefix}- **{key}**: —")
        else:
            lines.append(f"{prefix}- **{key}**: {val}")
    return "\n".join(lines)


def build_report(
    *,
    task_id: str,
    timing: dict[str, Any],
    intrinsic: dict[str, Any],
    judge: dict[str, Any],
) -> Report:
    lines: list[str] = [
        f"# Evaluation report: `{task_id}`",
        "",
        "## Timing",
        _render_dict(timing),
        "",
        "## Intrinsic metrics",
        _render_dict(intrinsic),
        "",
        "## Judge scores",
        _render_dict(judge),
        "",
    ]
    markdown = "\n".join(lines)
    data: dict[str, Any] = {
        "task_id": task_id,
        "timing": timing,
        "intrinsic": intrinsic,
        "judge": judge,
    }
    return Report(markdown=markdown, data=data)


if __name__ == "__main__":
    sample = build_report(
        task_id="demo",
        timing={"duration_sec": 5.0},
        intrinsic={"summary": {"compression_ratio": 0.12}},
        judge={"summary": {"faithfulness": 4}},
    )
    print(sample.markdown)
    print(json.dumps(sample.data, ensure_ascii=False, indent=2))

"""Validate commit message: allowed type prefix and 6-9 word description."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Final

ALLOWED_TYPES: Final[tuple[str, ...]] = (
    "feat",
    "fix",
    "refactor",
    "test",
    "docs",
    "chore",
    "ci",
)
MIN_WORDS: Final[int] = 6
MAX_WORDS: Final[int] = 9
EXPECTED_ARG_COUNT: Final[int] = 2
EXIT_USAGE_ERROR: Final[int] = 2
SUBJECT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(?P<type>\w+): (?P<desc>.+)$")


def validate(subject: str) -> str | None:
    """Return error message or None if subject is valid."""
    match = SUBJECT_PATTERN.match(subject)
    if match is None:
        return f"subject must match '<type>: <description>', got: {subject!r}"

    commit_type = match.group("type")
    description = match.group("desc")

    if commit_type not in ALLOWED_TYPES:
        return f"type must be one of {ALLOWED_TYPES}, got {commit_type!r}"

    word_count = len(description.split())
    if not MIN_WORDS <= word_count <= MAX_WORDS:
        return f"description must be {MIN_WORDS}-{MAX_WORDS} words, " f"got {word_count}: {description!r}"

    return None


def main() -> int:
    if len(sys.argv) != EXPECTED_ARG_COUNT:
        sys.stderr.write("usage: check_commit_msg.py <path-to-commit-msg-file>\n")
        return EXIT_USAGE_ERROR

    msg_path = Path(sys.argv[1])
    content = msg_path.read_text(encoding="utf-8")
    first_line = content.splitlines()[0].strip() if content else ""

    error = validate(first_line)
    if error is not None:
        sys.stderr.write(f"commit-msg FAIL: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""E2E: in-flight task resumes after app restart. RED-xfail until M4."""

from __future__ import annotations

import pytest


@pytest.mark.e2e
class TestRecoveryAfterRestart:
    def test_in_flight_task_resumes_after_app_restart(self) -> None:
        """Verifies design spec section 8.7: tasks survive coordinator restart.

        Requires a switchable ASGI app fixture that simulates kill and restart.
        Implemented in M4.
        """
        pytest.xfail("M4: recovery test requires app restart fixture")

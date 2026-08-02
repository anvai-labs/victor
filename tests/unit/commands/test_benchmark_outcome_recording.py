# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for per-task benchmark outcome recording (FEP-0012 Gap A).

A ctrl+c / crash during ``run_evaluation`` previously lost EVERY task's outcome
because recording was deferred to after the full run. The fix records each task
inside ``on_progress``; these tests cover the extracted helper directly.
"""

from types import SimpleNamespace
from unittest.mock import patch

from victor.ui.commands.benchmark import _record_benchmark_task_outcome


def test_passing_task_records_success_with_full_score():
    tr = SimpleNamespace(session_id="s1", tests_total=10, tests_passed=10)
    with patch("victor.agent.decisions.outcome.record_session_outcome") as rec:
        _record_benchmark_task_outcome(tr)
    rec.assert_called_once_with("s1", success=True, quality_score=1.0)


def test_failing_task_records_failure_with_partial_score():
    tr = SimpleNamespace(session_id="s2", tests_total=10, tests_passed=4)
    with patch("victor.agent.decisions.outcome.record_session_outcome") as rec:
        _record_benchmark_task_outcome(tr)
    rec.assert_called_once_with("s2", success=False, quality_score=0.4)


def test_zero_tests_is_failure_not_success():
    # No tests at all must NOT count as success (success requires _total > 0).
    tr = SimpleNamespace(session_id="s3", tests_total=0, tests_passed=0)
    with patch("victor.agent.decisions.outcome.record_session_outcome") as rec:
        _record_benchmark_task_outcome(tr)
    rec.assert_called_once_with("s3", success=False, quality_score=0.0)


def test_task_without_session_id_is_skipped():
    tr = SimpleNamespace(session_id="", tests_total=5, tests_passed=5)
    with patch("victor.agent.decisions.outcome.record_session_outcome") as rec:
        _record_benchmark_task_outcome(tr)
    rec.assert_not_called()

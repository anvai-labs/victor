# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the online per-turn auditor (EVR-6, FEP-0008 Phase C)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from victor.framework.evaluation_nodes import EvaluationDecision, EvaluationResult
from victor.framework.per_turn_auditor import (
    AuditVerdict,
    PerTurnAuditor,
    PerTurnAuditorConfig,
    resolve_per_turn_auditor_enabled,
)


def _turn(content: str = "", tools: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_results=tools or [])


def _complete(score: float = 0.9) -> EvaluationResult:
    return EvaluationResult(decision=EvaluationDecision.COMPLETE, score=score)


def _auditor(enabled: bool = True, max_alarms: int = 2) -> PerTurnAuditor:
    return PerTurnAuditor(PerTurnAuditorConfig(enabled=enabled, max_alarms=max_alarms))


# ── audit_turn signal ─────────────────────────────────────────────────────


def test_degenerate_turn_alarms() -> None:
    sig = _auditor().audit_turn(_turn(content="", tools=[{"name": "x", "success": False}]))
    assert sig.verdict is AuditVerdict.ALARM


def test_turn_with_content_continues() -> None:
    sig = _auditor().audit_turn(_turn(content="Here is the answer."))
    assert sig.verdict is AuditVerdict.CONTINUE


def test_turn_with_successful_tool_continues() -> None:
    sig = _auditor().audit_turn(_turn(content="", tools=[{"name": "write", "success": True}]))
    assert sig.verdict is AuditVerdict.CONTINUE


# ── apply() gating ────────────────────────────────────────────────────────


def test_disabled_is_a_strict_noop() -> None:
    auditor = _auditor(enabled=False)
    ev = _complete()
    out = auditor.apply(ev, _turn())  # degenerate, but disabled
    assert out is ev
    assert out.should_complete


def test_non_complete_passes_through() -> None:
    auditor = _auditor()
    ev = EvaluationResult(decision=EvaluationDecision.RETRY, score=0.3)
    out = auditor.apply(ev, _turn())  # degenerate, but not a COMPLETE
    assert out.should_retry


def test_degenerate_complete_downgraded_to_retry() -> None:
    auditor = _auditor()
    out = auditor.apply(_complete(0.9), _turn())
    assert out.should_retry
    assert out.score <= 0.4
    assert out.metadata["per_turn_auditor"]["verdict"] == "alarm"


def test_healthy_complete_passes_through() -> None:
    auditor = _auditor()
    out = auditor.apply(_complete(), _turn(content="done"))
    assert out.should_complete
    assert out.metadata["per_turn_auditor"]["verdict"] == "continue"


def test_downgrades_are_bounded_then_allowed() -> None:
    auditor = _auditor(max_alarms=2)
    assert auditor.apply(_complete(), _turn()).should_retry  # alarm 1
    assert auditor.apply(_complete(), _turn()).should_retry  # alarm 2
    exhausted = auditor.apply(_complete(), _turn())  # budget spent → allow
    assert exhausted.should_complete
    assert exhausted.metadata["per_turn_auditor_exhausted"] is True


def test_reset_clears_alarm_budget() -> None:
    auditor = _auditor(max_alarms=1)
    assert auditor.apply(_complete(), _turn()).should_retry  # alarm 1
    assert auditor.apply(_complete(), _turn()).should_complete  # exhausted
    auditor.reset()
    assert auditor.apply(_complete(), _turn()).should_retry  # budget restored


# ── flag resolver ─────────────────────────────────────────────────────────


def test_resolver_default_off() -> None:
    assert resolve_per_turn_auditor_enabled(SimpleNamespace(agent=None)) is False


def test_resolver_reads_settings() -> None:
    s = SimpleNamespace(agent=SimpleNamespace(per_turn_auditor=True))
    assert resolve_per_turn_auditor_enabled(s) is True


def test_resolver_env_overrides_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VICTOR_PER_TURN_AUDITOR", "1")
    assert (
        resolve_per_turn_auditor_enabled(
            SimpleNamespace(agent=SimpleNamespace(per_turn_auditor=False))
        )
        is True
    )
    monkeypatch.setenv("VICTOR_PER_TURN_AUDITOR", "off")
    assert (
        resolve_per_turn_auditor_enabled(
            SimpleNamespace(agent=SimpleNamespace(per_turn_auditor=True))
        )
        is False
    )

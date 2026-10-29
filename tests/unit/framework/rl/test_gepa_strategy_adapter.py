# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
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

"""GEPAServiceStrategy: trace evidence assembled for GEPA reflection."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from victor.framework.rl.gepa_strategy_adapter import GEPAServiceStrategy
from victor.framework.rl.learners.trace_analysis import FAILURE_HINTS


def _trace(success, failures=None, tool_calls=3, tokens=100):
    return SimpleNamespace(
        success=success,
        tool_failures=failures or {},
        tool_call_details=[],
        tool_calls=tool_calls,
        tokens_used=tokens,
        task_type="action",
        provider="anthropic",
        completion_score=1.0 if success else 0.2,
    )


def _strategy():
    tier_manager = MagicMock()
    service = tier_manager.get_service.return_value
    service.reflect.side_effect = lambda summary, section, text: summary
    return GEPAServiceStrategy(tier_manager), service


def test_reflect_sends_gap_report_profile_and_heuristics_to_the_service():
    strategy, service = _strategy()
    traces = [
        _trace(False, {"edit_mismatch": 5}),
        _trace(True),
        _trace(False, {"edit_mismatch": 2}),
    ]

    summary = strategy.reflect(traces, "tools", "current text", unused_kwarg=1)

    service.reflect.assert_called_once()
    _, section, text = service.reflect.call_args.args
    assert (section, text) == ("tools", "current text")
    assert "CAPABILITY GAP ANALYSIS" in summary
    assert "DATA DISTRIBUTION PROFILE" in summary
    assert "Aggregate (3 sessions)" in summary
    assert "Success rate: 1/3" in summary


def test_reflect_without_failures_has_no_gap_report():
    strategy, _ = _strategy()
    summary = strategy.reflect([_trace(True), _trace(True)], "tools", "t")
    assert "CAPABILITY GAP ANALYSIS" not in summary
    assert "Top failures" not in summary


def test_heuristic_summary_ranks_failures_and_attaches_known_hints():
    known = next(iter(FAILURE_HINTS))
    traces = [_trace(False, {known: 3, "unmapped_failure": 1}), _trace(False, {known: 2})]

    text = GEPAServiceStrategy._build_heuristic_summary(traces)

    lines = text.splitlines()
    top = lines.index("- Top failures:")
    assert lines[top + 1].startswith(f"  {known}: 5 → Hint: {FAILURE_HINTS[known]}")
    assert lines[top + 2] == "  unmapped_failure: 1"


def test_heuristic_summary_counts_list_valued_tool_calls():
    text = GEPAServiceStrategy._build_heuristic_summary([_trace(True, tool_calls=["a", "b"])])
    assert "Avg tool calls: 2.0" in text

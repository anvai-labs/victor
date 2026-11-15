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

"""RLM-3 Phase 1: structured error diagnostics for the graph tools.

The classifier must (a) assign every failure mode observed in the tool's error
paths a stable category, (b) never raise on malformed input — the classifier
becoming a new failure mode would defeat its purpose, and (c) stamp the
diagnostics at the single ``_graph_tool_entry`` seam without changing what
searches return.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from victor_coding.tools.graph_diagnostics import (
    CATEGORY_DEPENDENCY_MISSING,
    CATEGORY_EXECUTION_ERROR,
    CATEGORY_NO_INDEX,
    CATEGORY_NO_RESULTS,
    CATEGORY_NONE,
    CATEGORY_QUERY_FORMAT,
    CATEGORY_THRESHOLD_FILTERED,
    CATEGORY_TIMEOUT,
    CATEGORY_UNKNOWN,
    CATEGORY_UNRESOLVED_NODE,
    RL_DIAGNOSTICS_KEY,
    attach_graph_rl_diagnostics,
    classify_graph_result,
)


def _error_result(error: str, **extra: Any) -> Dict[str, Any]:
    result: Dict[str, Any] = {"success": False, "mode": "search", "error": error}
    result.update(extra)
    return result


class TestClassifyGraphResult:
    """Positive: each observed failure mode maps to its category."""

    @pytest.mark.parametrize(
        "error",
        [
            "Project graph database is unavailable/empty. Run: graph(...) to build the graph index first.",
            "database is empty",
            "project database is empty",
        ],
    )
    def test_no_index(self, error: str) -> None:
        assert classify_graph_result(_error_result(error)) == CATEGORY_NO_INDEX

    def test_dependency_missing(self) -> None:
        assert (
            classify_graph_result(_error_result("No module named 'x'", unavailable=True))
            == CATEGORY_DEPENDENCY_MISSING
        )

    def test_unresolved_node(self) -> None:
        assert (
            classify_graph_result(_error_result("Could not resolve graph node 'Foo'"))
            == CATEGORY_UNRESOLVED_NODE
        )

    def test_query_format_unsupported_mode(self) -> None:
        assert (
            classify_graph_result(_error_result("Unsupported graph mode: everything"))
            == CATEGORY_QUERY_FORMAT
        )

    def test_query_format_missing_required_arg(self) -> None:
        # Exact shape produced by _build_node_required_error.
        assert (
            classify_graph_result(
                _error_result("The 'neighbors' mode requires a 'node' parameter to identify it.")
            )
            == CATEGORY_QUERY_FORMAT
        )

    def test_requires_substring_alone_is_not_query_format(self) -> None:
        # Negative: loose substring matching would misclassify this.
        assert (
            classify_graph_result(_error_result("operation requires a database lock"))
            == CATEGORY_EXECUTION_ERROR
        )

    def test_timeout(self) -> None:
        assert classify_graph_result(_error_result("Operation timed out")) == CATEGORY_TIMEOUT

    def test_execution_error(self) -> None:
        assert classify_graph_result(_error_result("boom")) == CATEGORY_EXECUTION_ERROR

    def test_threshold_filtered(self) -> None:
        result = {
            "success": True,
            "focus_node": {"id": "n1"},
            "potential_relationships": [],
            "threshold_used": 0.5,
            "total_discovered": 0,
        }
        assert classify_graph_result(result) == CATEGORY_THRESHOLD_FILTERED

    def test_semantic_success_with_results(self) -> None:
        result = {
            "success": True,
            "potential_relationships": [{"id": "n2", "similarity": 0.7}],
            "threshold_used": 0.5,
            "total_discovered": 1,
        }
        assert classify_graph_result(result) == CATEGORY_NONE

    def test_no_results(self) -> None:
        assert classify_graph_result({"success": True, "matches": []}) == CATEGORY_NO_RESULTS

    def test_success_with_results(self) -> None:
        assert classify_graph_result({"success": True, "matches": [{"id": "n1"}]}) == CATEGORY_NONE


class TestClassifyGraphResultDefensive:
    """Negative: malformed input never raises and never mislabels as none."""

    def test_non_dict_is_unknown(self) -> None:
        assert classify_graph_result("### ❌ broken") == CATEGORY_UNKNOWN
        assert classify_graph_result(None) == CATEGORY_UNKNOWN
        assert classify_graph_result([{"success": True}]) == CATEGORY_UNKNOWN

    def test_error_without_text(self) -> None:
        # success=False with no error string: a failure we cannot explain.
        assert classify_graph_result({"success": False}) == CATEGORY_EXECUTION_ERROR

    def test_non_string_error(self) -> None:
        assert classify_graph_result({"success": False, "error": 42}) == CATEGORY_EXECUTION_ERROR

    def test_weird_counts_do_not_raise(self) -> None:
        assert classify_graph_result({"success": True, "matches": "banana"}) == CATEGORY_NONE
        assert classify_graph_result({"success": True, "total_discovered": None}) == CATEGORY_NONE

    def test_empty_dict_is_none_shaped(self) -> None:
        # No success flag at all (schema result, bounding stats): treat as
        # success-shaped, must not raise.
        assert classify_graph_result({"modes": ["search"]}) == CATEGORY_NONE


class TestAttachGraphRlDiagnostics:
    def test_stamps_namespace_key(self) -> None:
        result = attach_graph_rl_diagnostics({"success": True, "matches": []}, 12.34)
        assert result[RL_DIAGNOSTICS_KEY] == {
            "error_category": CATEGORY_NO_RESULTS,
            "duration_ms": 12.3,
            "result_count": 0,
        }

    def test_overwrites_stale_diagnostics(self) -> None:
        result = {"success": True, "matches": [{"id": 1}], RL_DIAGNOSTICS_KEY: {"stale": True}}
        attach_graph_rl_diagnostics(result, 5.0)
        assert "stale" not in result[RL_DIAGNOSTICS_KEY]
        assert result[RL_DIAGNOSTICS_KEY]["error_category"] == CATEGORY_NONE

    def test_non_dict_passthrough_unchanged(self) -> None:
        assert attach_graph_rl_diagnostics("plain", 1.0) == "plain"

    def test_duration_rounding(self) -> None:
        result = attach_graph_rl_diagnostics({"success": True}, 100.0)
        assert result[RL_DIAGNOSTICS_KEY]["duration_ms"] == 100.0


class TestGraphToolSeam:
    """The ``_graph_tool_entry`` seam stamps diagnostics on every return path."""

    @pytest.mark.asyncio
    async def test_seam_stamps_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from victor_coding.tools import graph_tool

        async def fake_impl(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {"success": True, "mode": "search", "matches": [{"id": "n1"}, {"id": "n2"}]}

        monkeypatch.setattr(graph_tool, "_graph_impl", fake_impl)
        result = await graph_tool._graph_tool_entry(mode="search", query="x")
        assert result["success"] is True
        assert result["matches"] == [{"id": "n1"}, {"id": "n2"}]
        assert result[RL_DIAGNOSTICS_KEY]["error_category"] == CATEGORY_NONE
        assert result[RL_DIAGNOSTICS_KEY]["result_count"] == 2
        assert result[RL_DIAGNOSTICS_KEY]["duration_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_seam_stamps_empty_database_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from victor_coding.tools import graph_tool

        async def fake_impl(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "success": False,
                "mode": "search",
                "error": "Project graph database is unavailable/empty. Run reindex.",
            }

        monkeypatch.setattr(graph_tool, "_graph_impl", fake_impl)
        result = await graph_tool._graph_tool_entry(mode="search", query="x")
        assert result[RL_DIAGNOSTICS_KEY]["error_category"] == CATEGORY_NO_INDEX

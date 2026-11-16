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

"""Structured error diagnostics for the graph tools (RLM-3, Phase 1).

RL analysis (2026-11) measured ``graph_search`` at 36.4% success and ``graph``
at 87.6%, but failures carried no structured error information, so the failure
modes were undiagnosable. This module classifies every graph tool result into a
small error taxonomy and publishes it through the per-execution RL signal
channel (``victor.tools.rl_signal``), which the tool executor drains into
``RLEvent.metadata`` (RLM-2 wiring):

- at the tool seam (``_graph_tool_entry``) so every graph mode and every
  focused tool (``graph_search``, ``graph_query``, ...) is covered;
- defensively — a missing/malformed field classifies as ``unknown``/``none``,
  never raises (the classifier must not become a new failure mode);
- with zero change to what searches return: the result dict is NOT modified
  (an earlier revision stamped a result key; that leaked into model-visible
  serialized output).

Categories:
- ``none``: success with results.
- ``no_index``: project graph database empty/unavailable (recoverable via
  ``reindex=True``).
- ``dependency_missing``: the indexing provider/vertical is unavailable.
- ``unresolved_node``: a required node argument could not be resolved.
- ``query_format``: malformed/unsupported query or mode/arguments.
- ``threshold_filtered``: semantic search succeeded but returned zero
  relationships at the configured threshold.
- ``no_results``: success-shaped search with an empty result set (non-semantic).
- ``timeout``: adaptive/asyncio timeout.
- ``execution_error``: any other failure.
- ``unknown``: defensive fallback when the result shape is unrecognized.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from victor.tools.rl_signal import attach_tool_outcome_metadata

CATEGORY_NONE = "none"
CATEGORY_NO_INDEX = "no_index"
CATEGORY_DEPENDENCY_MISSING = "dependency_missing"
CATEGORY_UNRESOLVED_NODE = "unresolved_node"
CATEGORY_QUERY_FORMAT = "query_format"
CATEGORY_THRESHOLD_FILTERED = "threshold_filtered"
CATEGORY_NO_RESULTS = "no_results"
CATEGORY_TIMEOUT = "timeout"
CATEGORY_EXECUTION_ERROR = "execution_error"
CATEGORY_UNKNOWN = "unknown"

_RESULT_LIST_KEYS = (
    "matches",
    "nodes",
    "edges",
    "relationships",
    "potential_relationships",
    "results",
    "components",
    "paths",
)

_EMPTY_DATABASE_MARKERS = (
    "database is empty",
    "database is unavailable",
    "project database is empty",
)
_TIMEOUT_MARKERS = (
    "timed out",
    "timeout",
    "asyncio.timeouterror",
    "TimeoutError",
)


def _result_count(result: Dict[str, Any]) -> Optional[int]:
    """Best-effort count of returned entities; ``None`` when not derivable."""
    for key in _RESULT_LIST_KEYS:
        value = result.get(key)
        if isinstance(value, list):
            return len(value)
    for key in ("total_discovered", "count", "total"):
        value = result.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def classify_graph_result(result: Any) -> str:
    """Classify a graph tool result into an error category.

    Structural signals win (``success``/``unavailable`` flags, threshold
    metadata); message-prefix checks are the fallback. Never raises.
    """
    try:
        if not isinstance(result, dict):
            return "unknown"

        if result.get("unavailable"):
            return "dependency_missing"

        error = result.get("error")
        error_text = str(error) if error else ""

        if result.get("success") is False:
            lowered = error_text.lower()
            if any(marker in lowered for marker in _EMPTY_DATABASE_MARKERS):
                return "no_index"
            if error_text.startswith("Could not resolve graph node"):
                return "unresolved_node"
            # Exact builder shapes — loose substring matching here would
            # misclassify unrelated errors that happen to contain "requires".
            if error_text.startswith("Unsupported graph mode") or (
                error_text.startswith("The '") and "requires a 'node' parameter" in lowered
            ):
                return "query_format"
            if any(marker in lowered for marker in _TIMEOUT_MARKERS):
                return "timeout"
            return "execution_error"

        # Success-shaped results: distinguish "found nothing" from "found".
        if "threshold_used" in result:
            discovered = result.get("total_discovered")
            if isinstance(discovered, int) and discovered == 0:
                return "threshold_filtered"
            return "none"
        count = _result_count(result)
        if count == 0:
            return "no_results"
        return "none"
    except Exception:  # classifier must never become a failure mode
        return "unknown"


def attach_graph_rl_diagnostics(
    result: Any, duration_ms: float, *, tool_name: Optional[str] = None
) -> Any:
    """Publish classification + timing via the RL signal channel.

    Returns ``result`` unchanged — the result dict is never modified (it is
    serialized into model-visible output). Attaching is a no-op when the
    executor has not seeded the channel (direct calls, tests).
    """
    try:
        category = classify_graph_result(result)
        fields: Dict[str, Any] = {
            "error_category": category,
            "duration_ms": round(float(duration_ms), 1),
        }
        if tool_name:
            fields["tool_name"] = tool_name
        if isinstance(result, dict):
            count = _result_count(result)
            if count is not None:
                fields["result_count"] = count
        attach_tool_outcome_metadata(**fields)
    except Exception:
        pass  # diagnostics must never become a failure mode
    return result

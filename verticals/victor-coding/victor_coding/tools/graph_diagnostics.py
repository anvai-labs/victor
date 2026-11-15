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
small error taxonomy and stamps it onto the result dict under a namespaced
``rl`` key:

- at the tool seam (``_graph_tool_entry``) so every graph mode and every
  focused tool (``graph_search``, ``graph_query``, ...) is covered;
- defensively — a missing/malformed field classifies as ``unknown``/``none``,
  never raises (the classifier must not become a new failure mode);
- with zero behavior change to what searches return beyond the added
  diagnostics key.

The ``rl`` payload is designed to flow into ``RLEvent.metadata`` once the
tool-executor emit path gains a result-metadata passthrough (tracked by
RLM-2); until then it serves result-level diagnosis and testing.

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

RL_DIAGNOSTICS_KEY = "rl"
"""Namespaced result key carrying graph tool diagnostics."""

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
            return CATEGORY_UNKNOWN

        if result.get("unavailable"):
            return CATEGORY_DEPENDENCY_MISSING

        error = result.get("error")
        error_text = str(error) if error else ""

        if result.get("success") is False:
            lowered = error_text.lower()
            if any(marker in lowered for marker in _EMPTY_DATABASE_MARKERS):
                return CATEGORY_NO_INDEX
            if error_text.startswith("Could not resolve graph node"):
                return CATEGORY_UNRESOLVED_NODE
            # Exact builder shapes — loose substring matching here would
            # misclassify unrelated errors that happen to contain "requires".
            if error_text.startswith("Unsupported graph mode") or (
                error_text.startswith("The '") and "requires a 'node' parameter" in lowered
            ):
                return CATEGORY_QUERY_FORMAT
            if any(marker in lowered for marker in _TIMEOUT_MARKERS):
                return CATEGORY_TIMEOUT
            return CATEGORY_EXECUTION_ERROR

        # Success-shaped results: distinguish "found nothing" from "found".
        if "threshold_used" in result:
            discovered = result.get("total_discovered")
            if isinstance(discovered, int) and discovered == 0:
                return CATEGORY_THRESHOLD_FILTERED
            return CATEGORY_NONE
        count = _result_count(result)
        if count == 0:
            return CATEGORY_NO_RESULTS
        return CATEGORY_NONE
    except Exception:  # classifier must never become a failure mode
        return CATEGORY_UNKNOWN


def attach_graph_rl_diagnostics(
    result: Any, duration_ms: float, *, tool_name: Optional[str] = None
) -> Any:
    """Stamp classification + timing onto a graph tool result dict.

    Non-dict results (defensive: bounding/serialization edge cases) are
    returned unchanged. Existing diagnostics are overwritten so the value
    always describes the current call (results may be cached downstream).
    """
    try:
        category = classify_graph_result(result)
        if isinstance(result, dict):
            diagnostics: Dict[str, Any] = {
                "error_category": category,
                "duration_ms": round(float(duration_ms), 1),
            }
            if tool_name:
                diagnostics["tool_name"] = tool_name
            count = _result_count(result)
            if count is not None:
                diagnostics["result_count"] = count
            result[RL_DIAGNOSTICS_KEY] = diagnostics
        return result
    except Exception:
        return result

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

"""RLM-2: shell outcome classification, the RL signal channel, and the
shell() wrapper attachment.

The RL data analysis (docs/analysis/rl-data-summary-2026-11-15.md) showed
shell at 44% success with zero diagnostic metadata — every failure row
carried no exit code, no error class. These tests pin the classification
taxonomy, the executor<->tool side channel, and that shell() populates it.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from victor.tools.bash import classify_shell_outcome, shell
from victor.tools.rl_signal import (
    attach_tool_outcome_metadata,
    drain_tool_outcome_metadata,
    seed_tool_outcome_metadata,
)

# --- Classifier ---------------------------------------------------------------


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"success": True, "return_code": 0, "stdout": "ok"}, "none"),
        ({"success": True, "return_code": 0}, "none"),
        (
            {"success": False, "return_code": -1, "error": "Command timed out after 5 seconds"},
            "timeout",
        ),
        (
            {
                "success": False,
                "return_code": -1,
                "error": "Dangerous command blocked: rm -rf /",
                "stdout": "",
                "stderr": "",
            },
            "blocked",
        ),
        (
            {
                "success": False,
                "return_code": -1,
                "error": "Filesystem-wide searches are not allowed.",
            },
            "blocked",
        ),
        (
            {"success": False, "return_code": -1, "error": "Missing required parameter: cmd"},
            "validation",
        ),
        (
            {
                "success": False,
                "return_code": -1,
                "error": "'foo <bar>' is a documentation placeholder, not a command.",
            },
            "validation",
        ),
        (
            {"success": False, "return_code": -1, "error": "Working directory does not exist: /x"},
            "validation",
        ),
        (
            {
                "success": False,
                "return_code": 126,
                "error": "Command failed with exit code 126: ./script.sh",
                "stderr": "permission denied",
            },
            "permission",
        ),
        (
            {
                "success": False,
                "return_code": 1,
                "error": "Command failed with exit code 1: ls /nope",
                "stderr": "ls: /nope: Permission denied",
            },
            "permission",
        ),
        (
            {
                "success": False,
                "return_code": 127,
                "error": "Command failed with exit code 127: foo",
                "stderr": "foo: command not found",
            },
            "not_found",
        ),
        (
            {
                "success": False,
                "return_code": 2,
                "error": "Command failed with exit code 2: sh -c",
                "stderr": "syntax error near unexpected token",
            },
            "syntax",
        ),
        (
            {
                "success": False,
                "return_code": 1,
                "error": "Command failed with exit code 1: grep -q x file",
                "stderr": "",
            },
            "exit_code",
        ),
        # Negative: failed with no distinguishing signal — must not guess.
        ({"success": False, "return_code": 0, "error": ""}, "unknown"),
        ({"success": False}, "unknown"),
    ],
)
def test_classify_shell_outcome(result: Dict[str, Any], expected: str):
    assert classify_shell_outcome(result) == expected


@pytest.mark.parametrize("bad", [None, "string", 42, [], {"success": "yes-not-bool"}])
def test_classify_shell_outcome_never_raises(bad: Any):
    """Negative: malformed inputs classify as 'unknown'/'none', never raise."""
    category = classify_shell_outcome(bad)
    assert category in {"unknown", "none", "exit_code"}


# --- RL signal channel --------------------------------------------------------


def test_channel_seed_attach_drain_roundtrip():
    seed_tool_outcome_metadata()
    attach_tool_outcome_metadata(error_category="timeout", return_code=-1)
    attach_tool_outcome_metadata(duration_ms=12.5)  # second attach merges
    assert drain_tool_outcome_metadata() == {
        "error_category": "timeout",
        "return_code": -1,
        "duration_ms": 12.5,
    }


def test_channel_drain_clears_state():
    seed_tool_outcome_metadata()
    attach_tool_outcome_metadata(error_category="exit_code")
    drain_tool_outcome_metadata()
    # Negative: a stale payload must never leak into a later drain.
    assert drain_tool_outcome_metadata() == {}


def test_channel_attach_without_seed_is_noop():
    drain_tool_outcome_metadata()  # ensure clean state
    attach_tool_outcome_metadata(error_category="blocked")  # must not raise
    assert drain_tool_outcome_metadata() == {}


async def test_shell_wrapper_attaches_outcome_metadata():
    """The decorated public shell() attaches classified metadata (real run)."""
    seed_tool_outcome_metadata()
    result = await shell(cmd="exit 3")
    assert result["success"] is False
    assert result["return_code"] == 3
    metadata = drain_tool_outcome_metadata()
    assert metadata["error_category"] == "exit_code"
    assert metadata["return_code"] == 3
    assert metadata["timed_out"] is False
    assert "duration_ms" in metadata
    assert metadata["duration_ms"] >= 0


def test_shell_signature_visible_through_decorator():
    """The @tool decorator must still see the real signature (functools.wraps)."""
    import inspect

    params = list(inspect.signature(shell).parameters)
    assert params[0] == "cmd"
    assert "cwd" in params and "_exec_ctx" in params

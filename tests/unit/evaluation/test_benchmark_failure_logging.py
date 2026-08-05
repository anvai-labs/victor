# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for benchmark failure-diagnosis logging (agent_adapter + harness).

Covers the gaps that made benchmark failures undiagnosable:
- `_record_tool_failure` now logs the full error text (was only hashed).
- `_file_modifying_tools()` recognizes the real tool names ``edit``/``write``
  (was only legacy aliases → files_modified always False after an edit).
  (#368 replaced the `_FILE_MODIFYING_TOOLS` constant with this derivation:
  CORE AccessMode.WRITE metadata unioned with a curated fallback.)
"""

import logging
from unittest.mock import MagicMock

from victor.evaluation.agent_adapter import (
    VictorAgentAdapter,
    _FILE_MODIFYING_TOOLS_FALLBACK,
    _file_modifying_tools,
)


def _make_adapter() -> VictorAgentAdapter:
    orch = MagicMock()
    orch._on_tool_start_callback = None
    orch._on_tool_complete_callback = None
    orch.reset_conversation = MagicMock()
    return VictorAgentAdapter(orch)


def test_file_modifying_tools_recognize_edit_and_write():
    """The real tool names 'edit' and 'write' must be tracked as file-modifying.

    Regression guard: previously only legacy aliases ('file_write', 'file_edit',
    'edit_file', 'patch') were recognized, so an `edit`/`write` tool call left
    files_modified=False even after a successful edit.
    """
    tools = _file_modifying_tools()
    assert "edit" in tools
    assert "write" in tools
    # legacy aliases still recognized
    assert "file_write" in tools
    assert "patch" in tools
    # Read-only tools must NOT be in the curated fallback. (The derived set
    # unions registry AccessMode.WRITE tools, so assert the curated contract —
    # a registry-populated environment could legitimately add new WRITE tools.)
    assert "read" not in _FILE_MODIFYING_TOOLS_FALLBACK
    assert "shell" not in _FILE_MODIFYING_TOOLS_FALLBACK
    # And in this test environment the derived set stays read-only-free too.
    assert "read" not in tools
    assert "shell" not in tools


def test_record_tool_failure_logs_the_error_text(caplog):
    """The full (truncated) error text must be logged on every tool failure.

    Previously _record_tool_failure only hashed the error for the circuit
    breaker, so the failure cause was invisible in logs.
    """
    adapter = _make_adapter()
    with caplog.at_level(logging.WARNING, logger="victor.evaluation.agent_adapter"):
        adapter._record_tool_failure("edit", "commit failed: FileNotFoundError: backups/x.bak")

    assert any(
        "edit" in rec.message and "FileNotFoundError" in rec.message for rec in caplog.records
    ), [rec.message for rec in caplog.records]


def test_record_tool_failure_still_circuit_breaks(caplog):
    """Logging is additive — the circuit-breaker behavior is preserved."""
    adapter = _make_adapter()
    threshold = adapter._tool_failure_threshold
    with caplog.at_level(logging.WARNING, logger="victor.evaluation.agent_adapter"):
        for _ in range(threshold):
            adapter._record_tool_failure("graph", "import error: missing module")
    # tool auto-disabled after threshold consecutive same-error failures
    assert "graph" in adapter._disabled_tools

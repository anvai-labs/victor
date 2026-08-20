# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for the edit tool's replace_lines op + per-op partial application.

Improvement 3 (line-number editing): ``replace_lines`` bypasses old_str matching
entirely — the agent specifies WHERE (from its last read) + WHAT (new content).
Improvement 1 (per-op partial): when one op in a same-file batch fails, the
correct ops still apply (not all-or-nothing per file).
"""

from victor.tools.file_editor_tool import _resolve_replace_lines


def test_replace_lines_basic():
    """Replace lines 2-3 (1-indexed inclusive) with new content."""
    content = "line1\nline2\nline3\nline4"
    result = _resolve_replace_lines(content, 2, 3, "REPLACED", "f.py")
    assert result["ok"] is True
    assert result["new_content"] == "line1\nREPLACED\nline4"


def test_replace_lines_single_line():
    """Replace a single line (line_start == line_end)."""
    content = "a\nb\nc"
    result = _resolve_replace_lines(content, 2, 2, "B", "f.py")
    assert result["ok"] is True
    assert result["new_content"] == "a\nB\nc"


def test_replace_lines_multi_line_replacement():
    """Replace 1 line with multiple lines."""
    content = "x\ny\nz"
    result = _resolve_replace_lines(content, 2, 2, "p\nq\nr", "f.py")
    assert result["ok"] is True
    assert result["new_content"] == "x\np\nq\nr\nz"


def test_replace_lines_delete_range():
    """Replace a range with empty string (deletion — no leftover empty line)."""
    content = "a\nb\nc\nd"
    result = _resolve_replace_lines(content, 2, 3, "", "f.py")
    assert result["ok"] is True
    assert result["new_content"] == "a\nd"


def test_replace_lines_invalid_range_start_zero():
    """Line numbers are 1-indexed — line_start=0 is invalid."""
    result = _resolve_replace_lines("a\nb", 0, 1, "x", "f.py")
    assert result["ok"] is False
    assert "invalid range" in result["reason"]


def test_replace_lines_invalid_range_end_past_file():
    """line_end beyond the file length is invalid."""
    result = _resolve_replace_lines("a\nb", 1, 10, "x", "f.py")
    assert result["ok"] is False
    assert "invalid range" in result["reason"]


def test_replace_lines_end_before_start():
    """line_end < line_start is invalid."""
    result = _resolve_replace_lines("a\nb\nc", 3, 1, "x", "f.py")
    assert result["ok"] is False
    assert "invalid range" in result["reason"]


def test_replace_lines_preserves_surrounding_content():
    """Lines before and after the range are preserved exactly."""
    content = "keep1\nkeep2\nold1\nold2\nold3\nkeep3\nkeep4"
    result = _resolve_replace_lines(content, 3, 5, "NEW", "f.py")
    assert result["ok"] is True
    assert result["new_content"] == "keep1\nkeep2\nNEW\nkeep3\nkeep4"

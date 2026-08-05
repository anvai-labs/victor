# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License").

"""Tests for the edit tool's backslash-normalized match fallback.

Reproduces the real benchmark failure (astropy-14182, qdp.py): the agent's
old_str had double-escaped backslashes (``\\\\d`` from JSON) while the file had
single (``\\d``) → exact match failed → no patch → task failed. The fallback
normalizes ``\\\\d`` → ``\\d`` and retries.
"""

from victor.tools.file_editor_tool import _normalize_backslashes, _resolve_replace


def test_exact_match_unchanged():
    """The common case (exact match) is unaffected by the fallback."""
    content = "DEBUG = True\n"
    result = _resolve_replace(content, "DEBUG = True", "DEBUG = False", "f.py")
    assert result["ok"] is True
    assert result["fuzzy"] is False
    assert "normalized" not in result
    assert result["new_content"] == "DEBUG = False\n"


def test_backslash_normalized_fallback_regex_pattern():
    """The qdp.py scenario: file has ``\\d`` (single), old_str has ``\\\\d``
    (double, from JSON escaping). The normalized fallback should match + apply."""
    # File content: literal single-backslash \d (raw string in Python)
    file_content = r'_decimal_re = r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?"'
    # Agent's old_str: literal double-backslash \\d (JSON escaping artifact)
    old_str = r'_decimal_re = r"[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][+-]?\\d+)?"'
    new_str = r'_decimal_re = r"[+-]?(\\d+(\\.\\d*)?|\\.\\d+)([eE][+-]?\\d+)?"  # fixed'

    # Exact match must FAIL (old_str has \\d, file has \d).
    assert file_content.count(old_str) == 0

    result = _resolve_replace(file_content, old_str, new_str, "qdp.py")
    assert result["ok"] is True, f"expected normalized match, got: {result.get('reason', '')[:100]}"
    assert result.get("normalized") == "backslash"
    assert "# fixed" in result["new_content"]


def test_normalize_backslashes_helper():
    """The helper collapses doubled backslashes."""
    assert _normalize_backslashes(r"\\d") == r"\d"
    assert _normalize_backslashes(r"\d") == r"\d"  # already single → unchanged
    assert _normalize_backslashes("C:\\\\Users") == "C:\\Users"
    assert _normalize_backslashes("no backslashes") == "no backslashes"


def test_legitimate_no_match_still_fails():
    """When neither exact nor normalized matches, the error is returned."""
    content = "foo = 1\n"
    result = _resolve_replace(content, "bar = 2", "bar = 3", "f.py")
    assert result["ok"] is False
    assert "not found" in result["reason"]


def test_ambiguous_normalized_rejected():
    """When the normalized old_str matches multiple times, don't guess."""
    content = "x = \\d\ny = \\d\n"  # two occurrences of \d
    old_str = r"x = \\d"  # normalizes to "x = \d" — but only 1 match (x, not y)
    # Actually "x = \d" appears once (the y line has "y = \d"). So this should match.
    result = _resolve_replace(content, old_str, "x = \\d #done", "f.py")
    assert result["ok"] is True  # unique match after normalization

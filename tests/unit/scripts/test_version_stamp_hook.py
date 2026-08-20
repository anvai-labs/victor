# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for the MkDocs version-stamp hook (single-source docs version from VERSION).

The hook substitutes only the literal ``{{ victor_version }}`` token, leaving every other
``{{ ... }}`` (cookiecutter trees, Cypher queries, workflow prompts) untouched — that targeted
behaviour is what makes it safe where a full mkdocs-macros/Jinja pass would not be.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_HOOK_PATH = _REPO_ROOT / "mkdocs_hooks" / "version_stamp.py"


def _load_hook():
    spec = importlib.util.spec_from_file_location("victor_version_stamp_hook", _HOOK_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hook_file_exists():
    assert _HOOK_PATH.is_file(), f"version hook missing at {_HOOK_PATH}"


def test_version_matches_version_file():
    mod = _load_hook()
    expected = (_REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert mod._VERSION == expected


def test_token_is_substituted():
    mod = _load_hook()
    out = mod.on_page_markdown("**Version**: {{ victor_version }} | rest")
    assert "{{ victor_version }}" not in out
    assert mod._VERSION in out


def test_other_braces_left_untouched():
    mod = _load_hook()
    # cookiecutter / Cypher / workflow-prompt style braces a full Jinja pass would mangle
    samples = [
        "{{cookiecutter.vertical_name}}/",
        "MATCH (c:Class {{name: '{method_name}'}})",
        'prompt: "Analyze: {{input}}"',
    ]
    for s in samples:
        assert mod.on_page_markdown(s) == s


def test_pages_without_token_pass_through_verbatim():
    mod = _load_hook()
    s = "no token here\n\n```python\nx = 1\n```\n"
    assert mod.on_page_markdown(s) == s

# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MkDocs hook: single-source the docs version stamp from the ``VERSION`` file.

Replaces the literal token ``{{ victor_version }}`` in page markdown with the victor-ai release
version (read once from the repo-root ``VERSION``) at build time.

This is a *targeted string replace*, not a Jinja pass (i.e. not mkdocs-macros): the docs contain
~40 literal ``{{ ... }}`` examples (cookiecutter trees, Cypher queries, workflow prompts) that a
full macro/Jinja pass would mangle, so we deliberately substitute only this one token and leave
everything else untouched. Result: bump ``VERSION`` and every doc stamp updates — one source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

_VERSION = (Path(__file__).resolve().parents[1] / "VERSION").read_text(encoding="utf-8").strip()
_TOKEN = "{{ victor_version }}"


def on_page_markdown(markdown: str, **_: Any) -> str:
    """Substitute the version token; pass other markdown through verbatim."""
    if _TOKEN in markdown:
        return markdown.replace(_TOKEN, _VERSION)
    return markdown

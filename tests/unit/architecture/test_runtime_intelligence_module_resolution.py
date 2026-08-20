# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Architecture guardrails for runtime-intelligence module resolution."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LEGACY_IMPORT = "from victor.agent.intelligent_pipeline import"
ALLOWED_LEGACY_FILE = REPO_ROOT / "victor/agent/intelligent_pipeline.py"


def test_production_code_uses_canonical_runtime_intelligence_module() -> None:
    """Production code should not import the deprecated shim module."""
    offenders: list[str] = []
    for path in (REPO_ROOT / "victor").rglob("*.py"):
        if path == ALLOWED_LEGACY_FILE:
            continue
        if LEGACY_IMPORT in path.read_text():
            offenders.append(str(path.relative_to(REPO_ROOT)))

    assert offenders == []

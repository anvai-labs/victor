#!/usr/bin/env python3
# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fail CI when docs contradict code-derived facts or the declared doc canon.

The docs are hand-written (no mkdocstrings/auto-gen), so factual claims — the release version,
the provider count, the tool-module count, the vertical count — drift silently. This gate makes
the alignment self-enforcing: it derives what it can from code (version from ``VERSION``, provider
count from the provider modules) and pins the rest to a single declared canon, then fails if any
doc states a contradicting number.

Pure stdlib so it runs in the lightweight ``ci-fast`` gate with no install. Scope is deliberately
narrow (claim-specific phrasings) to avoid false positives on incidental prose.

Update path when the code legitimately changes:
- version  -> bump ``VERSION`` (this gate then requires the doc stamps to match)
- providers -> automatic (derived from ``victor/providers/*_provider.py``)
- tool modules / verticals -> update ``CANON_TOOL_MODULES`` / ``CANON_VERTICALS`` below
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]

# Curated counts that are not cleanly derivable from the file tree (the tool "modules" and the
# vertical set are intentional groupings). Single source of truth — keep in sync with CLAUDE.md.
CANON_TOOL_MODULES = 34
CANON_VERTICALS = 9

# Beyond docs/, also scan the root instruction files that carry canon counts
# (CLAUDE.md, AGENTS.md, .victor/init.md). These are hand-written and drift the
# same way the docs do, so a provider/tool/vertical claim made here must stay
# aligned too.
DOC_GLOB = "docs/**/*.md"
EXTRA_FILES = (
    "mkdocs.yml",
    "docs/conf.py",
    "CLAUDE.md",
    "AGENTS.md",
    ".victor/init.md",
)

# The release version is checked ONLY in the canonical pages that carry it. Spec/FEP docs (e.g.
# vertical-package-spec.md) legitimately have their own independent **Version** stamp.
CANONICAL_VERSION_DOCS = frozenset(
    {
        "docs/index.md",
        "docs/features.md",
        "docs/architecture.md",
        "docs/tech-stack.md",
        "docs/architecture/BLUEPRINT.md",
        "docs/conf.py",
    }
)

_VERSION_STAMP = re.compile(r"\*\*Version\*\*:\s*(\d+\.\d+\.\d+)")
_CONF_RELEASE = re.compile(r'release\s*=\s*"(\d+\.\d+\.\d+)"')
# Public-facing minimum-version claims.  This deliberately does not match a
# CI test matrix such as "Python 3.11, 3.12"—only a stated minimum such as
# "Python 3.11+" or "requires Python >=3.11".
_PYTHON_MINIMUM = re.compile(
    r"(?:Python|python)\s*(?:version\s*)?(?:>=\s*)?(\d+\.\d+)(?:\+|\s+or\s+higher)",
    re.IGNORECASE,
)
# Claim-specific provider phrasings (avoids matching incidental "3 providers" prose).
_PROVIDERS = re.compile(
    r"(\d+)\+?\s+LLM\s+providers?\b"
    r"|(\d+)\+?\s+provider\s+adapters?\b"
    r"|supports\s+(\d+)\+?\s+(?:different\s+)?providers?\b",
    re.IGNORECASE,
)
_TOOLS = re.compile(r"(\d+)\s+tool\s+modules?\b", re.IGNORECASE)
# Verticals are checked ONLY on the precise canonical claim "N verticals (5 domain + 4 utility)".
# Bare "N verticals" is too noisy to enforce (legitimate contextual counts: "5/5 verticals
# migrated", "13 vertical-specific tests", "TD-2 Vertical Cleanup", "5 external verticals").
_VERTICALS_CANON = re.compile(r"(\d+)\s+verticals?\s*\(\s*5\s+domain", re.IGNORECASE)


def expected_version() -> str:
    return (ROOT / "VERSION").read_text(encoding="utf-8").strip()


def expected_python_minimum() -> str:
    """Return the package's declared minimum Python version."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requires_python = project["project"]["requires-python"]
    match = re.fullmatch(r">=\s*(\d+\.\d+)", requires_python)
    if match is None:
        raise ValueError(f"Unsupported requires-python format: {requires_python!r}")
    return match.group(1)


def expected_provider_count() -> int:
    """Number of concrete provider adapters (excludes the base class + the shared compat shim)."""
    providers_dir = ROOT / "victor" / "providers"
    files = [
        p
        for p in providers_dir.glob("*_provider.py")
        if p.stem != "base_provider" and "compat" not in p.stem
    ]
    return len(files)


def _first_int(match: re.Match) -> int:
    return int(next(g for g in match.groups() if g is not None))


def scan_text(
    label: str,
    text: str,
    version: str,
    provider_count: int,
    python_minimum: str | None = None,
) -> List[str]:
    """Return drift violations for one file's text. Pure — unit-testable without the tree."""
    errors: List[str] = []
    check_version = label in CANONICAL_VERSION_DOCS
    python_minimum = python_minimum or expected_python_minimum()
    for i, line in enumerate(text.splitlines(), 1):
        if check_version:
            for m in _VERSION_STAMP.finditer(line):
                if m.group(1) != version:
                    errors.append(f"{label}:{i} version stamp {m.group(1)} != {version}")
            for m in _CONF_RELEASE.finditer(line):
                if m.group(1) != version:
                    errors.append(f"{label}:{i} release {m.group(1)} != {version}")
        for m in _PROVIDERS.finditer(line):
            if _first_int(m) != provider_count:
                errors.append(f"{label}:{i} provider count {_first_int(m)} != {provider_count}")
        for m in _TOOLS.finditer(line):
            if int(m.group(1)) != CANON_TOOL_MODULES:
                errors.append(f"{label}:{i} tool modules {m.group(1)} != {CANON_TOOL_MODULES}")
        for m in _VERTICALS_CANON.finditer(line):
            if int(m.group(1)) != CANON_VERTICALS:
                errors.append(f"{label}:{i} verticals {m.group(1)} != {CANON_VERTICALS}")
        for m in _PYTHON_MINIMUM.finditer(line):
            if m.group(1) != python_minimum:
                errors.append(f"{label}:{i} Python minimum {m.group(1)} != {python_minimum}")
    return errors


def _doc_files() -> List[Path]:
    seen = {p for p in ROOT.glob(DOC_GLOB) if p.is_file()}
    seen.update(ROOT / f for f in EXTRA_FILES if (ROOT / f).exists())
    return sorted(seen)


def main() -> int:
    version = expected_version()
    providers = expected_provider_count()
    python_minimum = expected_python_minimum()
    errors: List[str] = []
    for path in _doc_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        errors.extend(
            scan_text(
                str(path.relative_to(ROOT)),
                text,
                version,
                providers,
                python_minimum,
            )
        )

    if errors:
        print("❌ Docs drift detected (docs contradict code / declared canon):")
        for e in errors:
            print(f"  - {e}")
        print(
            f"\nExpected: version={version}, providers={providers}, "
            f"tool_modules={CANON_TOOL_MODULES}, verticals={CANON_VERTICALS}, "
            f"python_minimum={python_minimum}"
        )
        print("Fix the docs, or update VERSION / CANON_* if the code legitimately changed.")
        return 1

    print(
        f"✓ Docs aligned: version={version}, providers={providers}, "
        f"tool_modules={CANON_TOOL_MODULES}, verticals={CANON_VERTICALS}, "
        f"python_minimum={python_minimum}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

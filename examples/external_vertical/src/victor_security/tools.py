# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
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

"""Custom tool shipped by the contract-only security vertical example.

The tool is duck-typed against the shape the Victor runtime accepts from
plugins: a ``name`` string, a ``description`` string, a ``parameters``
JSON-schema dict, and an async ``execute(**kwargs)``. No ``victor.*`` or
``victor_contracts`` runtime import is required to author it, which keeps
this module usable with only ``victor-contracts`` installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from re import Pattern
from typing import Any

# Defensive-security patterns for detecting hardcoded secrets. Pure Python
# regexes only - the example intentionally has no third-party dependencies.
_SECRET_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    ("aws_access_key_id", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("slack_token", re.compile(r"\bxox[abpors]-[A-Za-z0-9-]{10,}\b")),
    (
        "private_key_block",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "hardcoded_credential",
        re.compile(r"""(?ix)
            \b(password|passwd|secret|api[_-]?key|auth[_-]?token|access[_-]?token)\b
            \s*[:=]\s*
            ['"][^'"\s]{8,}['"]
            """),
    ),
)

_MAX_VISIBLE_CHARS = 4


def _mask(match_text: str) -> str:
    """Mask matched secret material so findings never leak the secret itself."""

    visible = match_text[:_MAX_VISIBLE_CHARS]
    return f"{visible}{'*' * max(len(match_text) - _MAX_VISIBLE_CHARS, 4)}"


def _scan_text(text: str, source: str) -> list[dict[str, Any]]:
    """Scan a block of text and return masked findings."""

    findings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for pattern_name, pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(line):
                findings.append(
                    {
                        "pattern": pattern_name,
                        "source": source,
                        "line": line_number,
                        "match": _mask(match.group(0)),
                    }
                )
    return findings


class SecretPatternScanTool:
    """Scan text or files for hardcoded secret patterns (defensive security).

    Detects likely secret material (cloud keys, tokens, private key blocks,
    hardcoded credentials) so it can be removed or rotated. Matched secret
    values are masked in the output.
    """

    @property
    def name(self) -> str:
        return "secret_pattern_scan"

    @property
    def description(self) -> str:
        return (
            "Scan provided text and/or files for hardcoded secrets such as "
            "cloud access keys, API tokens, private key blocks, and "
            "credential assignments. Findings are reported with masked "
            "matches for safe remediation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Raw text to scan for secret patterns.",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "File paths to scan for secret patterns.",
                },
            },
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Scan the provided text and/or files for secret patterns.

        Args:
            **kwargs: ``text`` (str) and/or ``paths`` (list of file paths).

        Returns:
            Dict with ``success``, ``findings`` (masked matches), and
            ``scanned_sources``. Unreadable paths are reported in ``errors``
            rather than raising.
        """

        text = kwargs.get("text")
        paths = kwargs.get("paths") or []

        findings: list[dict[str, Any]] = []
        scanned_sources: list[str] = []
        errors: list[str] = []

        if text:
            findings.extend(_scan_text(text, source="<text>"))
            scanned_sources.append("<text>")

        for raw_path in paths:
            path = Path(raw_path)
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                errors.append(f"{raw_path}: {exc}")
                continue
            findings.extend(_scan_text(content, source=str(path)))
            scanned_sources.append(str(path))

        return {
            "success": True,
            "findings": findings,
            "finding_count": len(findings),
            "scanned_sources": scanned_sources,
            "errors": errors,
        }


__all__ = ["SecretPatternScanTool"]

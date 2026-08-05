# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
# SPDX-License-Identifier: Apache-2.0

"""Shared search result types for integrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CodeSearchResult:
    """Result from a code search operation.

    For line-level code search results with file context.
    """

    file: str
    line: int
    content: str
    score: float
    context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "line": self.line,
            "content": self.content,
            "score": self.score,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeSearchResult":
        return cls(
            file=data["file"],
            line=data["line"],
            content=data["content"],
            score=data["score"],
            context=data.get("context", ""),
        )


__all__ = ["CodeSearchResult"]

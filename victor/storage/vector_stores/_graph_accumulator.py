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

"""Shared graph-guided seed accumulator (jscpd exact-duplicate extraction).

``ChangeImpactAccumulator`` and ``IssueLocalizationAccumulator`` were
token-identical modulo names and scoring constants (jscpd: 7 blocks, 134
lines). This module holds the one implementation; the two public classes
subclass it and pin their tuning knobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

_QUOTED_HINT_RE = re.compile(r"[`'\"]([^`'\"]{2,80})[`'\"]")
_DOTTED_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_CAMEL_HINT_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")
_SNAKE_HINT_RE = re.compile(r"\b[a-z]+_[a-z0-9_]{2,}\b")
_FILE_HINT_RE = re.compile(r"\b[\w./-]+\.(?:py|js|ts|tsx|jsx|java|go|rs|cpp|c|h)\b")


def extract_issue_hints(issue_description: str, *, limit: int = 12) -> List[str]:
    """Extract repository-relevant lexical hints from a natural-language issue."""
    candidates: List[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        normalized = value.strip()
        key = normalized.lower()
        if len(normalized) < 3 or key in seen:
            return
        if any(
            key in existing.lower() and len(existing) > len(normalized) for existing in candidates
        ):
            return
        seen.add(key)
        candidates.append(normalized)

    for pattern in (
        _QUOTED_HINT_RE,
        _DOTTED_SYMBOL_RE,
        _FILE_HINT_RE,
        _CAMEL_HINT_RE,
        _SNAKE_HINT_RE,
    ):
        for match in pattern.finditer(issue_description):
            if match.groups():
                _add(next(group for group in match.groups() if group))
            else:
                _add(match.group(0))
            if len(candidates) >= limit:
                return candidates
    return candidates


def _match_hints(
    hints: List[str],
    *,
    file_path: str,
    symbol_name: Optional[str],
    content: str,
    metadata: Dict[str, Any],
) -> List[str]:
    haystacks = [
        file_path,
        symbol_name or "",
        content,
        str(metadata.get("qualified_name", "")),
        str(metadata.get("name", "")),
    ]
    combined = "\n".join(haystacks).lower()
    return [hint for hint in hints if hint.lower() in combined]


def _dedupe_graph_rows(rows: Iterable[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: List[Dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("id", "")),
            str(row.get("file_path", "")),
            str(row.get("name", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(dict(row))
        if len(deduped) >= limit:
            break
    return deduped


@dataclass
class GraphSeedCandidate:
    """One file-level candidate accumulated from seeds and graph evidence."""

    file_path: str
    symbol_name: Optional[str] = None
    content: str = ""
    line_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)
    seed_score: float = 0.0
    graph_score: float = 0.0
    support_count: int = 0
    matched_hints: set[str] = field(default_factory=set)
    reasons: List[str] = field(default_factory=list)
    graph_context: Dict[str, Any] = field(
        default_factory=lambda: {"callers": [], "callees": [], "related_files": []}
    )


class GraphSeedAccumulator:
    """Accumulate semantic seeds and graph evidence into ranked candidates.

    Subclasses pin the tuning knobs (weights, boosts) and the naming bits
    (id prefix, metadata key, seed reason, relation word); the algorithm is
    this class alone.
    """

    id_prefix: str = "seed"
    metadata_key: str = "seed"
    seed_reason: str = "semantic_seed"
    relation_word: str = "related to"
    context_caller_weight: float = 0.08
    context_callee_weight: float = 0.04
    relation_caller_weight: float = 0.28
    relation_other_weight: float = 0.16
    hint_boost_unit: float = 0.05
    hint_boost_cap: float = 0.15
    support_boost_unit: float = 0.04
    support_boost_cap: float = 0.16

    def __init__(self, description: str, *, context_limit: int = 3) -> None:
        self.description = description
        self.context_limit = max(int(context_limit), 1)
        self.hints = extract_issue_hints(description)
        self._candidates: Dict[str, GraphSeedCandidate] = {}

    def add_seed(self, row: Dict[str, Any]) -> None:
        file_path = str(row.get("file_path") or row.get("metadata", {}).get("file_path") or "")
        if not file_path:
            return

        candidate = self._candidates.setdefault(file_path, GraphSeedCandidate(file_path=file_path))
        score = float(row.get("score", 0.0) or 0.0)
        metadata = dict(row.get("metadata", {}) or {})
        content = str(row.get("content", "") or "")
        symbol_name = (
            row.get("symbol_name") or metadata.get("qualified_name") or metadata.get("name")
        )
        line_number = (
            row.get("line_number") or metadata.get("start_line") or metadata.get("line_number")
        )

        if score >= candidate.seed_score:
            candidate.symbol_name = symbol_name
            candidate.content = content or candidate.content
            candidate.line_number = line_number or candidate.line_number
            candidate.metadata.update(metadata)

        candidate.seed_score = max(candidate.seed_score, score)
        candidate.sources.update(str(source) for source in row.get("sources", []) if source)
        if not candidate.sources:
            candidate.sources.add("semantic")
        candidate.support_count += 1
        candidate.matched_hints.update(
            _match_hints(
                self.hints,
                file_path=file_path,
                symbol_name=symbol_name,
                content=content,
                metadata=metadata,
            )
        )
        self._append_reason(candidate, self.seed_reason)

    def attach_graph_context(
        self,
        file_path: str,
        *,
        callers: Iterable[Dict[str, Any]],
        callees: Iterable[Dict[str, Any]],
    ) -> None:
        candidate = self._candidates.get(file_path)
        if candidate is None:
            return

        deduped_callers = _dedupe_graph_rows(callers, limit=self.context_limit)
        deduped_callees = _dedupe_graph_rows(callees, limit=self.context_limit)
        candidate.graph_context["callers"] = deduped_callers
        candidate.graph_context["callees"] = deduped_callees
        candidate.graph_context["related_files"] = sorted(
            {
                row.get("file_path")
                for row in [*deduped_callers, *deduped_callees]
                if row.get("file_path")
            }
        )
        candidate.graph_score += self.context_caller_weight * len(
            deduped_callers
        ) + self.context_callee_weight * len(deduped_callees)
        if deduped_callers or deduped_callees:
            self._append_reason(candidate, "graph_context")

    def add_graph_neighbors(
        self,
        *,
        seed_file_path: str,
        seed_symbol: Optional[str],
        seed_score: float,
        relation: str,
        neighbors: Iterable[Dict[str, Any]],
    ) -> None:
        relation_weight = (
            self.relation_caller_weight if relation == "callers" else self.relation_other_weight
        )

        for rank, neighbor in enumerate(neighbors, start=1):
            neighbor_file = str(neighbor.get("file_path") or "")
            if not neighbor_file:
                continue

            contribution = seed_score * relation_weight / rank
            candidate = self._candidates.setdefault(
                neighbor_file,
                GraphSeedCandidate(file_path=neighbor_file),
            )
            candidate.sources.add("graph")
            candidate.graph_score += contribution
            candidate.support_count += 1
            candidate.symbol_name = candidate.symbol_name or neighbor.get("name")
            candidate.line_number = candidate.line_number or neighbor.get("line_start")
            candidate.metadata.setdefault("related_symbol", neighbor.get("name"))
            if not candidate.content:
                relation_summary = seed_symbol or seed_file_path
                candidate.content = (
                    f"{neighbor.get('name') or neighbor_file} "
                    f"({relation} {self.relation_word} {relation_summary})"
                )
            candidate.matched_hints.update(
                _match_hints(
                    self.hints,
                    file_path=neighbor_file,
                    symbol_name=str(neighbor.get("name") or ""),
                    content=candidate.content,
                    metadata=dict(neighbor.get("metadata", {}) or {}),
                )
            )
            reason_target = seed_symbol or seed_file_path
            self._append_reason(candidate, f"{relation}:{reason_target}")

    def finalize(self, *, top_k: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        for candidate in self._candidates.values():
            hint_boost = min(
                self.hint_boost_unit * len(candidate.matched_hints), self.hint_boost_cap
            )
            support_boost = min(
                self.support_boost_unit * max(candidate.support_count - 1, 0),
                self.support_boost_cap,
            )
            score = round(
                candidate.seed_score + candidate.graph_score + hint_boost + support_boost,
                4,
            )

            metadata = dict(candidate.metadata)
            metadata[self.metadata_key] = {
                "seed_score": round(candidate.seed_score, 4),
                "graph_score": round(candidate.graph_score, 4),
                "matched_hints": sorted(candidate.matched_hints),
                "support_count": candidate.support_count,
                "reasons": list(candidate.reasons),
                "candidate_type": ("seed" if candidate.seed_score > 0 else "graph_related"),
            }

            results.append(
                {
                    "id": f"{self.id_prefix}:{candidate.file_path}",
                    "file_path": candidate.file_path,
                    "symbol_name": candidate.symbol_name,
                    "content": candidate.content,
                    "score": score,
                    "line_number": candidate.line_number,
                    "sources": sorted(candidate.sources),
                    "metadata": metadata,
                    "graph_context": candidate.graph_context,
                }
            )

        results.sort(
            key=lambda row: (
                row.get("score", 0.0),
                row.get("metadata", {}).get(self.metadata_key, {}).get("graph_score", 0.0),
                row.get("metadata", {}).get(self.metadata_key, {}).get("seed_score", 0.0),
                row.get("file_path", ""),
            ),
            reverse=True,
        )
        return results[:top_k]

    def _append_reason(self, candidate: GraphSeedCandidate, reason: str) -> None:
        if reason not in candidate.reasons:
            candidate.reasons.append(reason)

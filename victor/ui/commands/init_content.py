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

"""Helpers for init.md post-processing and summary metrics."""

from __future__ import annotations

import re
from typing import Optional

_ARCHITECTURE_SECTION_TITLES = ("architecture patterns", "architecture")
_ARCHITECTURE_EVIDENCE_SECTION_TITLES = (
    "architecture evidence",
    "graph-validated architecture signals",
)
_TOP_LEVEL_HEADING_RE = re.compile(r"^##\s+(.*?)\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\S")
_DEFAULT_INSERT_BEFORE = (
    "development commands",
    "dependencies",
    "configuration",
    "codebase scale",
    "important notes",
)
_QUALITY_SECTION_TITLES = (
    "repository working agreements",
    "repository guidelines",
    "development guidelines",
    "agent instructions",
)

_QUALITY_BASELINE_LINES = [
    "- **Follow the existing architecture first**: Before adding new abstractions, search for similar code and extend the smallest existing layer that fits.",
    "- **Respect repository boundaries**: Keep framework/runtime concerns, extension surfaces, generated artifacts, tests, docs, and subprojects in their established locations.",
    "- **Preserve user work in git**: Check `git status` before edits, do not revert unrelated changes, keep commits scoped, and prefer conventional commit messages when committing.",
    "- **Avoid generated-output churn**: Do not hand-edit generated directories or local caches unless the task explicitly targets generated artifacts.",
    "- **Match local naming and style**: Use the repository's existing module, class, function, file, and test naming conventions instead of introducing parallel vocabulary.",
    "- **Validate close to the change**: Run the smallest meaningful lint/test/build command for touched code and record any validation that could not be run.",
    "- **Document public behavior changes**: Update docs or examples when commands, configuration, public APIs, workflows, providers, or user-visible behavior changes.",
]


def _find_top_level_section_bounds(
    lines: list[str], titles: tuple[str, ...]
) -> Optional[tuple[int, int]]:
    lower_titles = {title.lower() for title in titles}
    start_idx: Optional[int] = None

    for idx, line in enumerate(lines):
        match = _TOP_LEVEL_HEADING_RE.match(line.strip())
        if not match:
            continue

        heading = match.group(1).strip().lower()
        if start_idx is None:
            if heading in lower_titles:
                start_idx = idx
            continue

        return start_idx, idx

    if start_idx is None:
        return None
    return start_idx, len(lines)


def _normalize_list_item(line: str) -> str:
    return _LIST_ITEM_RE.sub("", line.strip(), count=1).lower()


def count_architecture_patterns(content: str) -> int:
    """Count list items inside the Architecture section of init.md content."""
    lines = content.splitlines()
    bounds = _find_top_level_section_bounds(lines, _ARCHITECTURE_SECTION_TITLES)
    if bounds is None:
        return 0

    start_idx, end_idx = bounds
    return sum(1 for line in lines[start_idx + 1 : end_idx] if _LIST_ITEM_RE.match(line))


def _build_graph_fallback_patterns(graph_context: Optional[dict]) -> list[str]:
    if not graph_context:
        return []

    patterns = graph_context.get("patterns", {})
    stats = graph_context.get("stats", {})
    fallback: list[str] = []

    if patterns.get("protocol", 0) > 0:
        fallback.append(
            f"**Protocol/interface contracts**: {patterns['protocol']} implementation "
            "relationships indicate explicit abstraction boundaries."
        )
    if patterns.get("inheritance", 0) > 0:
        fallback.append(
            f"**Inheritance backbone**: {patterns['inheritance']} inheritance relationships "
            "highlight shared base types and extension points."
        )
    if graph_context.get("has_ccg") and stats.get("ccg_edges", 0) > 0:
        fallback.append(
            f"**Statement-level control/data flow**: {stats['ccg_edges']} CFG/CDG/DDG edges "
            "provide graph-level evidence for execution and data dependencies."
        )

    return fallback


def _build_architecture_evidence_lines(graph_context: Optional[dict]) -> list[str]:
    if not graph_context:
        return []

    stats = graph_context.get("stats", {})
    patterns = graph_context.get("patterns", {})
    complexity = graph_context.get("complexity", {})
    lines: list[str] = []

    total_nodes = stats.get("total_nodes", 0)
    total_edges = stats.get("total_edges", 0)
    if total_nodes or total_edges:
        lines.append(
            f"- **Graph scale**: `{total_nodes:,}` nodes and `{total_edges:,}` edges were "
            "available for architecture analysis."
        )

    if patterns.get("protocol", 0):
        lines.append(
            f"- **Protocol/interface evidence**: `{patterns['protocol']}` `IMPLEMENTS` "
            "relationships show explicit abstraction contracts."
        )

    if patterns.get("inheritance", 0):
        lines.append(
            f"- **Inheritance evidence**: `{patterns['inheritance']}` `INHERITS` relationships "
            "show shared extension backbones."
        )

    ccg_edges = stats.get("ccg_edges", 0)
    if graph_context.get("has_ccg") and ccg_edges:
        lines.append(
            f"- **Statement-level flow evidence**: `{ccg_edges:,}` CFG/CDG/DDG edges "
            f"(branching ratio `{complexity.get('avg_branching', 0):.2f}`) back the control- "
            "and data-flow claims with graph data."
        )

    return lines


def ensure_architecture_patterns_section(content: str, graph_context: Optional[dict]) -> str:
    """Ensure init.md has a minimally useful Architecture Patterns section.

    If the LLM omits or under-produces the section, use graph-derived fallback
    signals so the final document still contains concrete architectural guidance.
    """
    fallback_patterns = _build_graph_fallback_patterns(graph_context)
    if not fallback_patterns:
        return content

    lines = content.splitlines()
    bounds = _find_top_level_section_bounds(lines, _ARCHITECTURE_SECTION_TITLES)

    if bounds is None:
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            match = _TOP_LEVEL_HEADING_RE.match(line.strip())
            if match and match.group(1).strip().lower() in _DEFAULT_INSERT_BEFORE:
                insert_at = idx
                break

        block = ["## Architecture Patterns", ""]
        block.extend(f"- {pattern}" for pattern in fallback_patterns)
        block.append("")
        new_lines = lines[:insert_at] + block + lines[insert_at:]
        return "\n".join(new_lines).rstrip() + "\n"

    start_idx, end_idx = bounds
    existing_count = count_architecture_patterns(content)
    if existing_count >= 3:
        return content

    existing_items = {
        _normalize_list_item(line)
        for line in lines[start_idx + 1 : end_idx]
        if _LIST_ITEM_RE.match(line)
    }

    missing_lines = [
        f"- {pattern}" for pattern in fallback_patterns if pattern.lower() not in existing_items
    ]
    if not missing_lines:
        return content

    insert_at = end_idx
    while insert_at > start_idx + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1

    new_lines = lines[:insert_at] + missing_lines + [""] + lines[insert_at:]
    return "\n".join(new_lines).rstrip() + "\n"


def ensure_architecture_evidence_section(content: str, graph_context: Optional[dict]) -> str:
    """Insert or refresh a graph-backed Architecture Evidence section."""
    evidence_lines = _build_architecture_evidence_lines(graph_context)
    if not evidence_lines:
        return content

    lines = content.splitlines()
    evidence_bounds = _find_top_level_section_bounds(lines, _ARCHITECTURE_EVIDENCE_SECTION_TITLES)
    architecture_bounds = _find_top_level_section_bounds(lines, _ARCHITECTURE_SECTION_TITLES)

    block = ["## Architecture Evidence", ""]
    block.extend(evidence_lines)
    block.append("")

    if evidence_bounds is not None:
        start_idx, end_idx = evidence_bounds
        new_lines = lines[:start_idx] + block + lines[end_idx:]
        return "\n".join(new_lines).rstrip() + "\n"

    insert_at = len(lines)
    if architecture_bounds is not None:
        _start_idx, insert_at = architecture_bounds
    else:
        for idx, line in enumerate(lines):
            match = _TOP_LEVEL_HEADING_RE.match(line.strip())
            if match and match.group(1).strip().lower() in _DEFAULT_INSERT_BEFORE:
                insert_at = idx
                break

    new_lines = lines[:insert_at] + block + lines[insert_at:]
    return "\n".join(new_lines).rstrip() + "\n"


def ensure_quality_baseline_section(content: str) -> str:
    """Ensure init.md includes durable repository-quality guidance.

    The LLM/code analyzer owns project-specific discovery, but this repo owns
    the baseline working agreement that agents should carry into the system
    prompt.  Keep it generic so it is useful across languages and frameworks.
    """
    lines = content.splitlines()
    bounds = _find_top_level_section_bounds(lines, _QUALITY_SECTION_TITLES)

    if bounds is None:
        insert_at = len(lines)
        for idx, line in enumerate(lines):
            match = _TOP_LEVEL_HEADING_RE.match(line.strip())
            if match and match.group(1).strip().lower() in _DEFAULT_INSERT_BEFORE:
                insert_at = idx
                break

        block = ["## Repository Working Agreements", ""]
        block.extend(_QUALITY_BASELINE_LINES)
        block.append("")
        new_lines = lines[:insert_at] + block + lines[insert_at:]
        return "\n".join(new_lines).rstrip() + "\n"

    start_idx, end_idx = bounds
    existing_norm = {
        _normalize_list_item(line)
        for line in lines[start_idx + 1 : end_idx]
        if _LIST_ITEM_RE.match(line)
    }
    missing = [
        line for line in _QUALITY_BASELINE_LINES if _normalize_list_item(line) not in existing_norm
    ]
    if not missing:
        return content

    insert_at = end_idx
    while insert_at > start_idx + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    new_lines = lines[:insert_at] + missing + [""] + lines[insert_at:]
    return "\n".join(new_lines).rstrip() + "\n"


__all__ = [
    "count_architecture_patterns",
    "ensure_architecture_patterns_section",
    "ensure_architecture_evidence_section",
    "ensure_quality_baseline_section",
]

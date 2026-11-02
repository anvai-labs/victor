"""Bounded, offline source regions with host-file line provenance.

Only script blocks and explicitly selected documentation fences are analyzed.
Template expressions are outside this contract. Virtual compiler coordinates
are never assumed to coincide with the host document's SCIP coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re

EMBEDDED_SUFFIXES = frozenset({".vue", ".svelte", ".astro"})
DOC_SUFFIXES = frozenset({".md", ".markdown", ".mdown"})
MAX_REGIONS = 256
_ALIASES = {
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "tsx",
    "py": "python",
    "rs": "rust",
    "c++": "cpp",
    "c#": "csharp",
    "cs": "csharp",
    "sh": "bash",
    "shell": "bash",
    "kt": "kotlin",
}


@dataclass(frozen=True)
class SourceRegion:
    source: str
    language: str
    line_offset: int = 0
    kind: str = "source"
    index: int = 0


def source_regions(source: str, suffix: str) -> tuple[list[SourceRegion], dict[str, int]]:
    """Return regions and coverage counters; never load external script sources."""
    from victor.core.graph_rag.language_parsers import LANGUAGE_MODULES

    regions: list[SourceRegion] = []
    stats: dict[str, int] = {}

    def count(key: str) -> None:
        stats[key] = stats.get(key, 0) + 1

    def add(content: str, language: str, line: int, kind: str) -> None:
        if len(regions) >= MAX_REGIONS:
            stats["region_limit_files"] = 1
            return
        if language not in LANGUAGE_MODULES:
            count("unsupported_regions")
            return
        regions.append(SourceRegion(content, language, line, kind, len(regions)))

    if suffix in DOC_SUFFIXES:
        try:
            from markdown_it import MarkdownIt
        except ImportError:
            return [], {"unavailable_region_extractors": 1}
        for token in MarkdownIt("commonmark").parse(source):
            if token.type != "fence":
                continue
            info = token.info.strip().split()
            if not info:
                count("untyped_documentation_regions")
                continue
            language = _ALIASES.get(info[0].lower(), info[0].lower())
            # Unknown prose/data fences are intentional exclusions, not a
            # failed compiler frontend. Report them without pretending to parse.
            if language not in LANGUAGE_MODULES:
                count("ignored_documentation_regions")
                continue
            add(token.content, language, token.map[0] + 1, "documentation")
        return regions, stats

    html_source = source
    frontmatter = re.match(r"\A\ufeff?---[ \t]*\r?\n", source) if suffix == ".astro" else None
    if frontmatter is not None:
        start = frontmatter.end()
        end = re.search(r"^---[ \t]*\r?$", source[start:], re.MULTILINE)
        if end is None:
            return [], {"malformed_regions": 1}
        stop = end.start() + start
        add(source[start:stop], "typescript", 1, "frontmatter")
        html_source = (
            "".join("\n" if c == "\n" else " " for c in source[: stop + len(end[0])])
            + source[stop + len(end[0]) :]
        )

    class Scripts(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=False)
            self.active: tuple[int, int, str] | None = None
            self.line_starts = [0, *(m.end() for m in re.finditer("\n", html_source))]
            self.depth = 0
            self.templates = 0

        def source_offset(self) -> int:
            row, column = self.getpos()
            return self.line_starts[row - 1] + column

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag != "script":
                if tag == "template":
                    self.templates += 1
                if tag not in {
                    "br",
                    "hr",
                    "img",
                    "input",
                    "meta",
                    "link",
                    "source",
                    "wbr",
                    "area",
                    "base",
                    "embed",
                    "param",
                    "track",
                }:
                    self.depth += 1
                return
            if self.depth and (suffix != ".astro" or self.templates):
                count("ignored_nested_script_regions")
                return
            attributes = dict(attrs)
            if len(attributes) != len(attrs):
                count("unsupported_regions")
                return
            if "src" in attributes:
                count("unresolved_external_regions")
                return
            script_type = (attributes.get("type") or "").lower()
            if script_type in {
                "application/json",
                "application/ld+json",
                "importmap",
                "speculationrules",
            }:
                count("ignored_data_regions")
                return
            if script_type not in {
                "",
                "module",
                "text/javascript",
                "application/javascript",
                "text/typescript",
            }:
                count("unsupported_regions")
                return
            language = (
                attributes.get("lang")
                or ("typescript" if script_type == "text/typescript" else "javascript")
            ).lower()
            language = _ALIASES.get(language, language)
            if language not in {"javascript", "typescript", "tsx"}:
                count("unsupported_regions")
                return
            start = self.source_offset() + len(self.get_starttag_text())
            self.active = (start, html_source.count("\n", 0, start), language)

        def handle_endtag(self, tag: str) -> None:
            if tag == "script":
                if self.active is not None:
                    start, line, language = self.active
                    add(source[start : self.source_offset()], language, line, "script")
                    self.active = None
            else:
                self.depth = max(0, self.depth - 1)
                if tag == "template":
                    self.templates = max(0, self.templates - 1)

        def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            if tag == "script":
                self.handle_starttag(tag, attrs)
                if self.active is not None:
                    count("malformed_regions")
                    self.active = None

    parser = Scripts()
    try:
        parser.feed(html_source)
        parser.close()
    except (AssertionError, ValueError):
        count("malformed_regions")
    if parser.active is not None:
        count("malformed_regions")
    return regions, stats

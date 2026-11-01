"""Rich markdown helpers with diagram/image hooks."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable, List, Tuple

from rich.console import Console, ConsoleOptions, Group, RenderableType, RenderResult
from rich.markdown import CodeBlock as RichCodeBlock
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


class _CompactCodeBlock(RichCodeBlock):
    """Keep narrow-pane code formatting scoped to Victor renderables."""

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield Syntax(
            str(self.text).rstrip(),
            self.lexer_name,
            theme=self.theme,
            word_wrap=True,
            padding=(0, 1),
        )


class _VictorMarkdown(Markdown):
    elements = {**Markdown.elements, "fence": _CompactCodeBlock, "code_block": _CompactCodeBlock}


_DIAGRAM_LANGS = {"mermaid"}
_TOKEN_PATTERN = re.compile(
    r"```(?P<lang>[a-zA-Z0-9_-]+)\s*\n(?P<code>[\s\S]*?)```"
    r"|!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)"
    r"|^(?:>\s*)?\[!(?P<admon_type>NOTE|TIP|IMPORTANT|WARNING|CAUTION)\][ \t]*\n(?P<admon_body>(?:^(?:>\s*)?.*\n?)*?)(?=\n^(?:[^>]|$)|\Z)",
    re.MULTILINE,
)

# Pattern to detect Rich markup tags that could cause parsing errors
# Matches: [tag], [/tag], [tag=value], [tag=value1,value2]
_RICH_MARKUP_PATTERN = re.compile(r"\[/?[\w]+(?:=[^\]]+)?(?:,\s*[^\]]+)*\]")

# Phase 2: Long content threshold for collapsible sections
_LONG_CONTENT_LINES = 50
_LONG_CODE_BLOCK_LINES = 30


def _escape_rich_markup_from_text(text: str) -> str:
    """Escape Rich markup tags that could cause parsing errors.

    This function escapes Rich-specific markup tags like [bold], [/], [red], etc.
    while preserving markdown syntax. It's designed to handle cases where LLM
    output contains strings that look like Rich markup but aren't intended to be.

    Note: This is a best-effort escape. It escapes square brackets that look like
    Rich markup tags, which is safer than trying to parse and validate them.

    Args:
        text: The text to escape

    Returns:
        Text with Rich markup tags escaped by replacing [ with \\[
    """

    # Escape Rich markup tags by replacing [ with \\[
    # This is a simple approach that works for most cases
    # We escape patterns that look like Rich tags: [word], [/word], [tag=value]
    def _escape_match(match: re.Match) -> str:
        return match.group(0).replace("[", "\\[")

    return _RICH_MARKUP_PATTERN.sub(_escape_match, text)


def _markdown_block(text: str) -> Markdown:
    """Consistently style markdown output with a lighter base.

    ``rich.markdown.Markdown`` does not interpret Rich inline markup (it parses
    Markdown, not Rich ``[tags]``), so the text is passed through unescaped.
    Pre-escaping here was both redundant and counterproductive: it matched the
    ``[text]`` portion of markdown links and broke ``[text](url)`` rendering.
    ``_escape_rich_markup_from_text`` is retained as a utility for any
    markup-aware Rich path (e.g. ``console.print(Text(...))``) that may need it.
    """
    return _VictorMarkdown(text, style="markdown.text", justify="left")


def find_safe_split(content: str) -> int:
    """Return the char offset where the in-progress markdown TAIL begins.

    ``content[:offset]`` is the stable HEAD (complete markdown blocks) and
    ``content[offset:]`` is the active TAIL (the in-progress last block). The
    split is placed just after the last blank-line block boundary that is NOT
    inside an open fenced code block, so the HEAD can be rendered once and
    cached while only the TAIL is re-rendered each streaming tick.

    Returns 0 when inside an open fence (don't split a code block mid-stream) or
    when no safe boundary exists yet (the first block is still streaming).
    """
    last_safe = 0
    in_fence = False
    pos = 0
    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if in_fence:
            # A fence line (``` or ~~~) closes the block.
            if stripped[:3] in ("```", "~~~"):
                in_fence = False
        elif stripped[:3] in ("```", "~~~"):
            # Opening a fenced code block — no safe split inside it.
            in_fence = True
        elif stripped == "":
            # Blank line outside a fence = a completed block boundary.
            last_safe = pos + len(line)
        pos += len(line)
    return last_safe


def render_markdown_with_hooks(content: str) -> RenderableType:
    """Render markdown content with diagram/image hooks.

    Splits markdown into chunks so diagram code blocks and inline images can:
    - Render Mermaid snippets as ASCII trees for quick preview
    - Show textual placeholders for image links (since Rich can't display images)

    Falls back to plain text if Rich markup parsing fails, preventing rendering
    errors from breaking the agentic loop.
    """
    if not content.strip():
        return _markdown_block("")

    try:
        parts: List[RenderableType] = []
        cursor = 0
        for match in _TOKEN_PATTERN.finditer(content):
            start, end = match.span()
            if start > cursor:
                chunk = content[cursor:start]
                if chunk.strip():
                    parts.append(_markdown_block(chunk))

            admon_type = match.group("admon_type")
            lang = match.group("lang")

            if admon_type:
                admon_body = match.group("admon_body") or ""
                clean_body = re.sub(r"^>?\s*", "", admon_body, flags=re.MULTILINE)
                parts.append(_render_admonition(admon_type, clean_body))
            elif lang:
                lang = lang.lower()
                code = (match.group("code") or "").strip()
                if lang in _DIAGRAM_LANGS:
                    parts.append(_render_diagram(lang, code))
                else:
                    parts.append(_markdown_block(match.group(0)))
            else:
                alt = match.group("alt") or "image"
                src = match.group("src") or ""
                parts.append(_render_image_placeholder(alt, src))

            cursor = end

        remaining = content[cursor:]
        if remaining.strip():
            parts.append(_markdown_block(remaining))

        if not parts:
            return _markdown_block(content)
        if len(parts) == 1:
            return parts[0]
        return Group(*parts)

    except Exception as e:
        # If Rich rendering fails, fall back to plain text to prevent
        # rendering errors from breaking the agentic loop
        import logging

        logger = logging.getLogger(__name__)
        error_context = get_render_error_context(e, content)
        logger.warning(
            "Rich rendering failed, falling back to plain text: %s",
            error_context,
        )
        # Return as plain text (Rich Text object, not Markdown)
        from rich.text import Text
        from rich.panel import Panel

        # Show helpful error panel
        error_panel = Panel(
            Text(error_context, style="dim"),
            title="[yellow]⚠ Markdown Rendering Error[/]",
            border_style="yellow",
            padding=(0, 1),
        )
        # Return error panel plus fallback text
        return Group(error_panel, Text(content, style="dim"))


def _render_diagram(lang: str, code: str) -> RenderableType:
    if lang == "mermaid":
        return _render_mermaid(code)
    return Panel(
        Syntax(code, lang or "text", theme="monokai", word_wrap=True),
        title=f"{lang.title()} diagram",
        border_style="cyan",
    )


def _render_admonition(admon_type: str, body: str) -> RenderableType:
    colors = {
        "NOTE": "blue",
        "TIP": "green",
        "IMPORTANT": "magenta",
        "WARNING": "yellow",
        "CAUTION": "red",
    }
    icons = {
        "NOTE": "ℹ️",
        "TIP": "💡",
        "IMPORTANT": "✨",
        "WARNING": "⚠️",
        "CAUTION": "🛑",
    }
    color = colors.get(admon_type, "blue")
    icon = icons.get(admon_type, "ℹ️")

    from rich import box

    return Panel(
        _markdown_block(body),
        title=f"[{color}]{icon} {admon_type.title()}[/]",
        title_align="left",
        border_style=color,
        box=box.ROUNDED,
        padding=(0, 2),
    )


def _render_image_placeholder(alt_text: str, source: str) -> RenderableType:
    description = Text.assemble(
        ("Image: ", "bold cyan"),
        alt_text.strip() or "diagram",
        ("\nSource: ", "bold"),
        source,
    )
    return Panel(description, border_style="magenta")


def _render_mermaid(code: str) -> RenderableType:
    direction = _detect_direction(code)
    edges = _parse_mermaid_edges(code)
    if not edges:
        return Panel(
            Syntax(code or "", "mermaid", theme="monokai", word_wrap=True),
            title="Mermaid diagram (raw)",
            border_style="cyan",
        )

    adjacency = defaultdict(list)
    indegree = defaultdict(int)
    nodes = set()

    for src, dst, label in edges:
        adjacency[src].append((dst, label))
        indegree[dst] += 1
        nodes.add(src)
        nodes.add(dst)
        indegree.setdefault(src, 0)

    roots = [node for node in nodes if indegree[node] == 0] or sorted(nodes)

    forest = []
    for root in roots:
        tree = Tree(Text(f"{root}", style="bold"))
        _build_mermaid_tree(tree, root, adjacency, visited=set())
        forest.append(tree)

    title = f"Mermaid ({direction})"
    body: Iterable[RenderableType]
    if len(forest) == 1:
        body = [forest[0]]
    else:
        body = [Text("Multiple roots", style="bold dim")] + forest
    return Panel(Group(*body), title=title, border_style="cyan")


def _build_mermaid_tree(
    tree: Tree,
    node: str,
    adjacency: dict[str, List[Tuple[str, str | None]]],
    visited: set[str],
) -> None:
    if node in visited:
        tree.add(Text("(cycle)", style="dim red"))
        return
    visited.add(node)
    for dest, label in adjacency.get(node, []):
        edge_text = f"{label}: {dest}" if label else dest
        child = tree.add(edge_text)
        _build_mermaid_tree(child, dest, adjacency, visited.copy())


def _parse_mermaid_edges(code: str) -> List[Tuple[str, str, str | None]]:
    edges: List[Tuple[str, str, str | None]] = []
    for line in code.splitlines():
        text = line.strip()
        if not text or text.startswith("%") or "graph " in text.lower():
            continue
        if "-->" not in text:
            continue
        label = None

        if "-->|" in text:
            src_part, tail = text.split("-->|", 1)
            if "|" in tail:
                label, dest_part = tail.split("|", 1)
            else:
                dest_part = tail
        else:
            src_part, dest_part = text.split("-->", 1)

        src = _normalize_mermaid_node(src_part)
        dest = _normalize_mermaid_node(dest_part)
        if not src or not dest:
            continue
        edges.append((src, dest, (label or "").strip() or None))
    return edges


def _normalize_mermaid_node(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    # Remove trailing arrow modifiers or text (e.g., -.->)
    token = token.rstrip(".-")
    # Extract label inside brackets/braces/parentheses if present
    for opener, closer in ("[]", "()", "{}", "<>", "##"):
        o, c = opener[0], closer[0]
        if o in token and c in token:
            start = token.find(o) + 1
            end = token.find(c, start)
            if end > start:
                label = token[start:end].strip()
                if label:
                    return label
    # Strip identifier prefix like A or node1
    ident = re.match(r"[A-Za-z0-9_]+", token)
    if ident:
        remainder = token[ident.end() :].strip()
        if remainder.startswith("[") and "]" in remainder:
            inner = remainder[1 : remainder.find("]")]
            if inner.strip():
                return inner.strip()
        return ident.group(0)
    return token


def _detect_direction(code: str) -> str:
    for line in code.splitlines():
        text = line.strip().lower()
        if text.startswith("graph "):
            return text.split()[1].upper()
    return "TD"


# Phase 2: Enhanced markdown rendering features


def render_collapsible_section(
    title: str,
    content: str,
    lines_visible: int = 10,
    collapsed: bool = True,
) -> RenderableType:
    """Render a collapsible section for long content.

    Args:
        title: Section title
        content: Content to display
        lines_visible: Number of lines to show when collapsed
        collapsed: Whether to start collapsed

    Returns:
        Panel with truncated content hint
    """
    content_lines = content.splitlines()
    total_lines = len(content_lines)

    if total_lines <= lines_visible:
        # No need to collapse
        return _markdown_block(content)

    visible_lines = content_lines[:lines_visible]
    visible_content = "\n".join(visible_lines)
    remaining = total_lines - lines_visible

    hint = f"[dim italic]... {remaining} more line{'s' if remaining != 1 else ''} • expand to see full content[/]"

    from rich import box

    return Panel(
        Group(_markdown_block(visible_content), Text(hint)),
        title=title,
        title_align="left",
        border_style="blue",
        box=box.MINIMAL,
        padding=(0, 2),
    )


def render_enhanced_table(
    headers: List[str],
    rows: List[List[str]],
    title: str = "",
    style: str = "cyan",
) -> RenderableType:
    """Render an enhanced table with better styling.

    Args:
        headers: Table headers
        rows: Table rows (list of lists)
        title: Optional table title
        style: Color style for headers

    Returns:
        Rich Table with enhanced styling
    """
    table = Table(
        title=title,
        title_style=f"bold {style}",
        box=None,  # Use default box
        show_header=True,
        header_style=f"bold {style}",
        border_style="dim",
        padding=(0, 1),
        expand=False,  # Don't expand beyond content width
    )

    for header in headers:
        # Allow wrapping for all columns to prevent overflow
        table.add_column(header, no_wrap=False, overflow="fold")

    for row in rows:
        table.add_row(*[str(cell) for cell in row])

    return table


def should_collapse_content(content: str, content_type: str = "text") -> bool:
    """Determine if content should be collapsed.

    Args:
        content: Content to check
        content_type: Type of content (text, code, etc.)

    Returns:
        True if content is long enough to collapse
    """
    line_count = len(content.splitlines())
    threshold = _LONG_CODE_BLOCK_LINES if content_type == "code" else _LONG_CONTENT_LINES
    return line_count > threshold


def get_render_error_context(error: Exception, content: str) -> str:
    """Get helpful context for rendering errors.

    Args:
        error: The exception that occurred
        content: Content that failed to render

    Returns:
        Helpful error message with context
    """
    error_type = type(error).__name__
    error_msg = str(error)[:200]

    content_preview = content[:100].replace("\n", "\\n")
    if len(content) > 100:
        content_preview += "..."

    return (
        f"[{error_type}] {error_msg}\n\n"
        f"[dim]Content preview:[/]\n{content_preview}\n\n"
        f"[dim yellow]Tip: If this continues, try simplifying the markdown or avoiding special characters.[/]"
    )

"""Shared rendering utilities for stream renderers.

This module provides common rendering functions used across different
renderer implementations to ensure consistent visual output.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import re
from typing import Any, Optional, TypedDict

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

logger = logging.getLogger(__name__)

_TOOL_NAME_SEPARATOR_PATTERN = re.compile(r"[\s_\-]+")

_REASONING_PREFIX_MARKERS = (
    "💭 Thinking...",
    "💭",
    "🤔 Thinking...",
    "🤔",
    "Thinking...",
)

_THINKING_STATUS_PREFIXES = ("💭", "🤔", "◌")

# Canonical field list for a tool-result payload. This is the single source of
# truth shared by the producer (``ChunkGenerator.generate_tool_result_chunk``),
# the ``StreamRenderer.on_tool_result`` protocol, the Chainlit event mapping, and
# ``extract_tool_result_payload`` below — keep them in sync via this tuple.
TOOL_RESULT_FIELDS = (
    "name",
    "success",
    "elapsed",
    "arguments",
    "error",
    "follow_up_suggestions",
    "result",
    "original_result",
    "was_pruned",
)


class ToolResultPayload(TypedDict, total=False):
    """Normalized, flat tool-result payload carried on a stream event.

    Every key is optional; consumers read with ``.get(...)`` and sensible
    defaults. ``result`` is the (possibly truncated/placeholder) preview output;
    ``original_result`` is the full output for surfaces that render it directly
    (e.g. the web UI) or expand it on demand (the Rich CLI).
    """

    name: str
    success: bool
    elapsed: float
    arguments: dict[str, Any]
    error: Optional[str]
    follow_up_suggestions: Optional[list[dict[str, Any]]]
    result: Any
    original_result: Any
    was_pruned: bool


class StreamDeltaNormalizer:
    """Normalize provider stream chunks to append-only deltas.

    Some providers stream true deltas while others resend cumulative snapshots.
    Renderers should receive append-only text so UI surfaces do not duplicate
    already-rendered content.
    """

    _TAIL_WINDOW = 4096
    _MIN_OVERLAP_CHARS = 8

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._last_raw = ""
        self._emitted_tail = ""

    def consume(self, text: str) -> str:
        """Return only the unseen portion of ``text``."""
        if not text:
            return ""

        if text == self._last_raw:
            logger.debug("StreamDeltaNormalizer: exact duplicate, returning empty")
            return ""

        if self._last_raw and text.startswith(self._last_raw):
            delta = text[len(self._last_raw) :]
            self._last_raw = text
            self._extend_tail(delta)
            logger.debug(
                "StreamDeltaNormalizer: accumulated content, "
                "last_raw_len=%d, new_text_len=%d, delta_len=%d",
                len(self._last_raw) - len(delta),
                len(text),
                len(delta),
            )
            return delta

        overlap = self._find_overlap(text)
        self._last_raw = text
        if overlap and (overlap >= self._MIN_OVERLAP_CHARS or overlap * 2 >= len(text)):
            delta = text[overlap:]
            self._extend_tail(delta)
            logger.debug(
                "StreamDeltaNormalizer: found overlap=%d, text_len=%d, delta_len=%d",
                overlap,
                len(text),
                len(delta),
            )
            return delta

        self._extend_tail(text)
        logger.debug(
            "StreamDeltaNormalizer: no overlap, returning full text (%d chars)",
            len(text),
        )
        return text

    def _extend_tail(self, text: str) -> None:
        if not text:
            return
        self._emitted_tail = (self._emitted_tail + text)[-self._TAIL_WINDOW :]

    # Cap the overlap search to avoid O(n×m) scans on large payloads.
    _OVERLAP_SEARCH_CAP = 256

    def _find_overlap(self, text: str) -> int:
        max_len = min(len(self._emitted_tail), len(text), self._OVERLAP_SEARCH_CAP)
        for size in range(max_len, 0, -1):
            if self._emitted_tail[-size:] == text[:size]:
                return size
        return 0


def normalize_reasoning_content(text: str) -> str:
    """Strip provider-added reasoning banners from reasoning stream content."""
    normalized = text.lstrip()
    for marker in _REASONING_PREFIX_MARKERS:
        if normalized.startswith(marker):
            return normalized[len(marker) :].lstrip()
    return normalized


def is_thinking_status_message(message: str) -> bool:
    """Return True for generic thinking-status messages that should use thinking UI."""
    normalized = message.strip()
    for prefix in _THINKING_STATUS_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :].strip()
            break

    lowered = normalized.lower()
    return lowered in {"thinking", "thinking...", "thinking…"}


def format_tool_display_name(name: str) -> str:
    """Format a canonical tool identifier for lower-camel display in the UI.

    This keeps protocol and registry names unchanged while making renderer output
    easier to scan for humans.

    Examples:
        "code_search" -> "codeSearch"
        "git-status" -> "gitStatus"
        "CodeSearch" -> "codeSearch"
    """
    normalized = str(name).strip()
    if not normalized:
        return "unknown"

    if _TOOL_NAME_SEPARATOR_PATTERN.search(normalized):
        parts = [part for part in _TOOL_NAME_SEPARATOR_PATTERN.split(normalized) if part]
        if not parts:
            return normalized
        first = parts[0].lower()
        rest = [part[:1].upper() + part[1:] for part in parts[1:]]
        return first + "".join(rest)

    if normalized.isupper():
        return normalized.lower()

    if normalized[:1].isupper():
        return normalized[:1].lower() + normalized[1:]

    return normalized


def _coerce_mapping_dict(value: Any) -> dict[str, Any]:
    """Return a shallow dict for mapping-like values."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def extract_tool_call_arguments(event: Any) -> dict[str, Any]:
    """Extract tool-call arguments from either Agent or VictorClient events."""
    arguments = getattr(event, "arguments", None)
    if isinstance(arguments, Mapping):
        return dict(arguments)

    metadata = _coerce_mapping_dict(getattr(event, "metadata", None))
    metadata_arguments = metadata.get("arguments")
    if isinstance(metadata_arguments, Mapping):
        return dict(metadata_arguments)

    tool_start = metadata.get("tool_start")
    if isinstance(tool_start, Mapping):
        nested_arguments = tool_start.get("arguments")
        if isinstance(nested_arguments, Mapping):
            return dict(nested_arguments)

    return {}


def extract_tool_result_payload(event: Any) -> dict[str, Any]:
    """Normalize tool-result payloads across Agent and VictorClient streams."""
    metadata = _coerce_mapping_dict(getattr(event, "metadata", None))
    nested_payload = metadata.get("tool_result")
    result_payload = getattr(event, "result", None)

    payload: dict[str, Any] = {}
    if isinstance(nested_payload, Mapping):
        payload.update(dict(nested_payload))

    top_level_payload = {key: metadata[key] for key in TOOL_RESULT_FIELDS if key in metadata}
    payload.update(top_level_payload)

    if isinstance(result_payload, Mapping):
        payload.update(dict(result_payload))
    elif result_payload is not None and "result" not in payload:
        payload["result"] = result_payload

    payload.setdefault("name", getattr(event, "tool_name", None) or "unknown")
    payload.setdefault("success", getattr(event, "success", True))
    payload.setdefault("arguments", extract_tool_call_arguments(event))

    return payload


def _coerce_arguments_mapping(arguments: Any) -> dict[str, Any]:
    """Best-effort coercion of tool arguments to a mapping for display.

    Models sometimes deliver tool arguments as a raw JSON string (or another
    non-mapping shape). Display formatting must never crash on that: in
    session modality-doc-review-fixes-b4e87728 a str reached
    ``format_tool_args`` and the resulting ``AttributeError: 'str' object has
    no attribute 'items'`` replaced the real provider error in ``victor run``
    output.
    """
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        import json

        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, Mapping):
            return dict(parsed)
        return {"args": text}
    if arguments is None:
        return {}
    return {"args": str(arguments)}


def format_tool_args(arguments: Any, max_width: int = 80) -> str:
    """Format tool arguments for compact display.

    Shared utility used by all renderers for consistent arg formatting.

    Args:
        arguments: Tool arguments mapping (non-mapping shapes are coerced)
        max_width: Maximum total width for args string

    Returns:
        Formatted args string like "path='/file.py', limit=100"
    """
    arguments = _coerce_arguments_mapping(arguments)
    if not arguments:
        return ""
    parts = []
    total_len = 0
    for k, v in arguments.items():
        if isinstance(v, str):
            display = v if len(v) <= 60 else v[:57] + "..."
            part = f"{k}='{display}'"
        elif isinstance(v, (int, float, bool)):
            part = f"{k}={v}"
        elif v is None:
            continue
        elif isinstance(v, list):
            # Summarize list args meaningfully (e.g. edit's ops=[{type,path,...}])
            if v and isinstance(v[0], dict):
                first = v[0]
                op_type = first.get("type", "")
                op_path = first.get("path", "")
                summary = f"{op_type}:{op_path}" if op_type and op_path else str(len(v))
                part = f"{k}=[{summary}]" if len(v) == 1 else f"{k}=[{summary} +{len(v)-1}]"
            else:
                part = f"{k}=[{len(v)}]"
        else:
            part = f"{k}=..."
        if total_len + len(part) > max_width and parts:
            parts.append("...")
            break
        parts.append(part)
        total_len += len(part) + 2
    return ", ".join(parts)


def format_duration(elapsed: float) -> str:
    """Format elapsed seconds for compact CLI display."""
    if elapsed <= 0:
        return "0ms"
    if elapsed < 1:
        return f"{elapsed * 1000:.0f}ms"
    if elapsed < 10:
        return f"{elapsed:.1f}s"
    return f"{elapsed:.0f}s"


def format_tool_args_bash_style(arguments: Any, max_args: int = 5) -> str:
    """Format tool arguments in bash CLI style (--key=value).

    Args:
        arguments: Tool arguments mapping (non-mapping shapes are coerced)
        max_args: Maximum number of arguments to show (rest truncated)

    Returns:
        Formatted args string like "--path='file.py' --limit=100"
    """
    arguments = _coerce_arguments_mapping(arguments)
    if not arguments:
        return ""

    parts = []
    for i, (k, v) in enumerate(arguments.items()):
        if i >= max_args:
            parts.append("...")
            break

        if isinstance(v, str):
            # Escape single quotes in string values
            escaped = v.replace("'", "\\'") if "'" in v else v
            # Truncate long strings
            display = escaped if len(escaped) <= 60 else escaped[:57] + "..."
            part = f"--{k}='{display}'"
        elif isinstance(v, bool):
            # Boolean flags: --flag for True, omitted for False
            if v:
                part = f"--{k}"
            else:
                continue
        elif isinstance(v, (int, float)):
            part = f"--{k}={v}"
        elif v is None:
            continue
        elif isinstance(v, list):
            # List args: show count or first element
            if v and isinstance(v[0], dict):
                first = v[0]
                op_type = first.get("type", "")
                op_path = first.get("path", "")
                summary = f"{op_type}:{op_path}" if op_type and op_path else str(len(v))
                part = f"--{k}=[{summary}]" if len(v) == 1 else f"--{k}=[{summary} +{len(v)-1}]"
            else:
                part = f"--{k}=[{len(v)}]"
        else:
            part = f"--{k}=..."

        parts.append(part)

    return " ".join(parts)


def format_bash_command_invocation(tool_name: str, arguments: dict[str, Any]) -> str:
    """Format a complete bash-style command invocation.

    Args:
        tool_name: Name of the tool/command
        arguments: Tool arguments dict

    Returns:
        Formatted command like "$ code_search --query='auth' --path='src/'"
    """
    args_str = format_tool_args_bash_style(arguments)
    if args_str:
        return f"[dim]$[/] [bold cyan]{tool_name}[/] [dim]{args_str}[/]"
    else:
        return f"[dim]$[/] [bold cyan]{tool_name}[/]"


def render_file_preview(console: Console, path: str, content: str) -> None:
    """Render a file content preview panel.

    Shared utility for displaying file contents with syntax highlighting.

    Args:
        console: Rich Console to render to
        path: File path for title and syntax detection
        content: File content to display
    """
    ext = path.split(".")[-1] if "." in path else "txt"
    syntax = Syntax(content, ext, theme="monokai", line_numbers=False)
    console.print(Panel(syntax, title=f"[dim]{path}[/]", border_style="dim"))


def _append_inline_diff(body: "Text", old: str, new: str) -> None:
    """Append a `-`/`+` line pair with word-level change highlighting.

    Unchanged words are dim; removed words (in the `-` line) are bold red and
    inserted words (in the `+` line) are bold green — so a small edit reads as a
    few highlighted words rather than two fully-colored lines.
    """
    import difflib

    sm = difflib.SequenceMatcher(a=old.split(), b=new.split())
    # Removed line: context dim red, removed words bold red.
    body.append("-", style="red")
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        style = "red" if tag == "equal" else "bold red"
        if tag in ("equal", "delete", "replace"):
            for word in sm.a[i1:i2]:
                body.append(" " + word, style=style)
    body.append("\n")
    # Added line: context dim green, inserted words bold green.
    body.append("+", style="green")
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        style = "green" if tag == "equal" else "bold green"
        if tag in ("equal", "insert", "replace"):
            for word in sm.b[j1:j2]:
                body.append(" " + word, style=style)
    body.append("\n")


def render_edit_preview(console: Console, path: str, diff: str) -> None:
    """Render an edit diff preview as a compact unified diff.

    Lines are colored: +additions green, -deletions red, @@hunks cyan,
    file headers dim, context dim. Modified lines (a ``-`` run directly followed
    by a ``+`` run of equal length) are rendered with **word-level** change
    highlighting; pure additions/deletions stay line-level. Diff is wrapped in a
    Panel keyed by path for visual grouping.

    Args:
        console: Rich Console to render to
        path: File path being edited (panel title)
        diff: Unified-diff content
    """
    from rich.panel import Panel
    from rich.text import Text

    body = Text()
    lines = diff.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.startswith("@@"):
            body.append(line + "\n", style="cyan")
            i += 1
        elif line.startswith("+++") or line.startswith("---"):
            body.append(line + "\n", style="dim bold")
            i += 1
        elif line.startswith("-"):
            # Collect a run of removals, then any immediately-following additions.
            removed = []
            while i < n and lines[i].startswith("-") and not lines[i].startswith("--"):
                removed.append(lines[i][1:])
                i += 1
            added = []
            while i < n and lines[i].startswith("+") and not lines[i].startswith("++"):
                added.append(lines[i][1:])
                i += 1
            if removed and added and len(removed) == len(added):
                # Modification: word-level inline diff per paired line.
                for old, new in zip(removed, added):
                    _append_inline_diff(body, old, new)
            else:
                # Pure deletion / addition, or mismatched counts: line-level.
                for old in removed:
                    body.append("-" + old + "\n", style="red")
                for new in added:
                    body.append("+" + new + "\n", style="green")
        elif line.startswith("+"):
            body.append(line + "\n", style="green")
            i += 1
        else:
            body.append(line + "\n", style="dim")
            i += 1

    console.print(
        Panel(
            body,
            title=f"[dim]edit {path}[/]",
            border_style="dim",
            padding=(0, 1),
            expand=False,
        )
    )


def render_thinking_indicator(console: Console) -> None:
    """Render the thinking start indicator with enhanced visual emphasis.

    Args:
        console: Rich Console to render to
    """
    from victor.config.theme_settings import get_theme_settings

    theme_settings = get_theme_settings()

    # Use high contrast colors if enabled
    if theme_settings.high_contrast:
        # A rule separates the reasoning section from prior output for readability
        console.rule(style="bold cyan")
        console.print("🤔 [bold cyan]Thinking...[/]")
    else:
        # Subtle theme-based indicator
        console.rule(style="thinking.text")
        console.print("🤔 [thinking.indicator]Thinking...[/]", style="thinking.text")


def render_thinking_text(console: Console, text: str) -> None:
    """Render thinking content as dimmed/italic text.

    Args:
        console: Rich Console to render to
        text: Thinking text to display
    """
    styled = Text(text, style="thinking.text")
    console.print(styled, end="")


def render_content_badge(console: Console, content_type: str, icon: str = "") -> None:
    """Render a small badge for content type identification.

    Args:
        console: Rich Console to render to
        content_type: Type of content (thinking, tool, response, error)
        icon: Optional icon to display
    """
    from victor.config.theme_settings import get_theme_settings

    theme_settings = get_theme_settings()

    # Use brighter colors for high contrast mode
    if theme_settings.high_contrast:
        badges = {
            "thinking": ("[bold cyan]◌[/]", "[bold cyan]THINKING[/]"),
            "tool": ("[bold blue]⟳[/]", "[bold blue]TOOL[/]"),
            "response": ("[bold green]→[/]", "[bold green]RESPONSE[/]"),
            "error": ("[bold red]✗[/]", "[bold red]ERROR[/]"),
        }
    else:
        badges = {
            "thinking": ("[cyan]◌[/]", "[dim]THINKING[/]"),
            "tool": ("[blue]⟳[/]", "[dim]TOOL[/]"),
            "response": ("[green]→[/]", "[dim]RESPONSE[/]"),
            "error": ("[red]✗[/]", "[dim]ERROR[/]"),
        }

    prefix, label = badges.get(content_type, ("•", content_type.upper()))
    console.print(f"{prefix} {label}")


def render_status_message(console: Console, message: str) -> None:
    """Render a compact status line with lightweight semantic styling."""
    from victor.config.theme_settings import get_theme_settings

    theme_settings = get_theme_settings()

    normalized = message.strip()
    lowered = normalized.lower()

    # Use high contrast styling if enabled
    if theme_settings.high_contrast:
        if normalized.startswith("⚠") or "warning" in lowered or "failed" in lowered:
            prefix = "[bold yellow]![/]"
            body_style = "bold yellow"
            normalized = normalized.removeprefix("⚠️").removeprefix("⚠").strip()
        elif "thinking" in lowered:
            prefix = "[bold cyan]◌[/]"
            body_style = "bold cyan"
        else:
            prefix = "[bold blue]•[/]"
            body_style = "bold"
    else:
        if normalized.startswith("⚠") or "warning" in lowered or "failed" in lowered:
            prefix = "[yellow]![/]"
            body_style = "yellow"
            normalized = normalized.removeprefix("⚠️").removeprefix("⚠").strip()
        elif "thinking" in lowered:
            prefix = "[cyan]◌[/]"
            body_style = "dim italic"
        else:
            prefix = "[blue]•[/]"
            body_style = "dim"

    console.print(f"{prefix} [{body_style}]{normalized}[/]")


def render_tool_preview(
    console: Console,
    preview_text: str,
    *,
    total_lines: int,
    preview_lines: int,
    hotkey: str,
    contains_rich_markup: bool = False,
) -> None:
    """Render compact preview lines with an expandable gutter.

    Args:
        console: Rich Console instance
        preview_text: Text to render (may contain Rich markup)
        total_lines: Total number of lines in the full output
        preview_lines: Number of lines shown in this preview
        hotkey: Hotkey string for expanding the preview
        contains_rich_markup: If True, preview_text contains Rich markup and should
                             be rendered directly. If False, wrap with [dim] tags.
    """
    from rich import box
    from rich.panel import Panel
    from rich.text import Text

    if not preview_text.strip():
        return

    # If it contains rich markup, we must construct a Text object using from_markup
    # Otherwise we just use a dimmed Text object
    if contains_rich_markup:
        content = Text.from_markup(preview_text)
    else:
        content = Text(preview_text, style="dim")

    footer = None
    if total_lines > preview_lines:
        remaining = total_lines - preview_lines
        suffix = "s" if remaining != 1 else ""
        footer = f"[dim italic]... {remaining} more line{suffix} • {hotkey} at prompt or /expand[/]"

    panel = Panel(
        content,
        border_style="dim",
        box=box.MINIMAL,
        padding=(0, 2),
        subtitle=footer,
        subtitle_align="left",
    )
    console.print(panel)


# Tools whose names map to valid Pygments lexer identifiers.
_SYNTAX_TOOL_WHITELIST: frozenset[str] = frozenset(
    {
        "python",
        "javascript",
        "typescript",
        "json",
        "yaml",
        "bash",
        "shell",
        "sql",
        "html",
        "css",
        "xml",
        "markdown",
        "toml",
        "go",
        "rust",
        "java",
        "cpp",
        "c",
    }
)


def expand_tool_output(
    console: Console,
    tool_name: str,
    content: str,
    *,
    pause_fn=None,
    resume_fn=None,
    max_chars: int = 10000,
) -> None:
    """Render full tool output in a syntax-highlighted panel.

    Shared implementation used by both FormatterRenderer and LiveDisplayRenderer.

    Args:
        console: Rich Console to render to
        tool_name: Name of the tool (used for panel title)
        content: Full tool output text
        pause_fn: Optional callable to pause the renderer before printing
        resume_fn: Optional callable to resume the renderer after printing
        max_chars: Truncate content beyond this many characters
    """
    if pause_fn is not None:
        pause_fn()

    if len(content) > max_chars:
        console.print(f"[dim yellow]⚠ Output is {len(content)} chars, showing first {max_chars}[/]")
        content = content[:max_chars]

    # Derive a lexer only from the whitelisted set; fall back to plain text.
    last_segment = tool_name.split("_")[-1] if "_" in tool_name else tool_name
    lexer = last_segment if last_segment in _SYNTAX_TOOL_WHITELIST else "text"
    display_name = format_tool_display_name(tool_name)

    try:
        syntax = Syntax(content, lexer, theme="monokai", line_numbers=True, word_wrap=True)
        console.print(
            Panel(
                syntax,
                title=f"[bold]{display_name}[/] - Full Output",
                border_style="blue",
            )
        )
    except Exception:
        console.print(
            Panel(
                content,
                title=f"[bold]{display_name}[/] - Full Output",
                border_style="blue",
            )
        )

    if resume_fn is not None:
        resume_fn()


# Tool metadata formatting helpers (Phase 1: Unified Registry Display)

_ACCESS_MODE_COLORS = {
    "readonly": "green",
    "write": "yellow",
    "execute": "red",
    "network": "blue",
    "mixed": "magenta",
}

_COST_TIER_SYMBOLS = {
    "free": "",
    "low": "$",
    "medium": "$$",
    "high": "$$$",
}

_EXECUTION_CATEGORY_ICONS = {
    "read_only": "🔍",
    "write": "📝",
    "compute": "⚙️",
    "network": "🌐",
    "execute": "⚡",
    "mixed": "🔀",
}

# Explicit display labels. "mixed" is relabeled to "read+write" because the raw
# word "Mixed" was widely misread as a half-failed / mixed-success status (it is
# only a capability hint: the tool does both reads and writes). The success/fail
# signal is the green/red [DONE] color, not this badge.
_EXECUTION_CATEGORY_LABELS = {
    "mixed": "read+write",
}


def format_access_mode_badge(access_mode: str) -> str:
    """Format access mode as a colored badge.

    Args:
        access_mode: Access mode value (readonly, write, execute, network, mixed)

    Returns:
        Rich markup for colored badge
    """
    color = _ACCESS_MODE_COLORS.get(str(access_mode).lower(), "dim")
    mode_upper = str(access_mode).upper() if access_mode else "UNKNOWN"
    return f"[{color}]{mode_upper}[/]"


def format_cost_tier_indicator(cost_tier: str) -> str:
    """Format cost tier as dollar sign indicator.

    Args:
        cost_tier: Cost tier value (free, low, medium, high)

    Returns:
        Dollar sign indicator (empty for free, $ for low, etc.)
    """
    return _COST_TIER_SYMBOLS.get(str(cost_tier).lower(), "")


def format_execution_category_hint(execution_category: str) -> str:
    """Format execution category with icon hint.

    Args:
        execution_category: Execution category (read_only, write, compute, network, execute, mixed)

    Returns:
        Icon + category name
    """
    icon = _EXECUTION_CATEGORY_ICONS.get(str(execution_category).lower(), "•")
    cat_key = str(execution_category).lower()
    cat_label = _EXECUTION_CATEGORY_LABELS.get(
        cat_key, str(execution_category).replace("_", " ").title()
    )
    return f"{icon} {cat_label}"


def format_tool_metadata_badges(
    category: str = "",
    access_mode: str = "",
    cost_tier: str = "",
    execution_category: str = "",
) -> str:
    """Format complete tool metadata badges for display.

    Args:
        category: Tool category
        access_mode: Access mode (readonly, write, execute, network, mixed)
        cost_tier: Cost tier (free, low, medium, high)
        execution_category: Execution category

    Returns:
        Rich markup with all applicable badges
    """
    badges = []

    # Category badge (dimmed, no special styling)
    if category:
        badges.append(f"[dim dim]{category}[/]")

    # Access mode badge (colored)
    if access_mode and access_mode != "readonly":
        badges.append(format_access_mode_badge(access_mode))

    # Cost tier indicator
    if cost_tier:
        cost_indicator = format_cost_tier_indicator(cost_tier)
        if cost_indicator:
            badges.append(f"[yellow]{cost_indicator}[/]")

    # Execution category hint
    if execution_category:
        badges.append(f"[dim]{format_execution_category_hint(execution_category)}[/]")

    return " ".join(badges) if badges else ""


def get_tool_metadata_for_display(tool_name: str, arguments: dict | None = None) -> dict:
    """Get tool metadata from unified registry for display.

    Args:
        tool_name: Name of the tool
        arguments: This invocation's arguments. When provided, the static
            access mode is narrowed to the invocation's effective mode (e.g.
            a ``code grep`` renders READ ONLY instead of the tool's static
            MIXED envelope).

    Returns:
        Dict with category, access_mode, cost_tier, execution_category
    """
    display: dict
    try:
        from victor.tools.metadata_registry import ToolMetadataRegistry

        registry = ToolMetadataRegistry.get_instance()
        metadata = registry.get_metadata(tool_name)

        if metadata:
            display = {
                "category": metadata.category or "",
                "access_mode": (metadata.access_mode.value if metadata.access_mode else "readonly"),
                "cost_tier": (
                    metadata.cost_tier.value
                    if hasattr(metadata, "cost_tier") and metadata.cost_tier
                    else "free"
                ),
                "execution_category": (
                    metadata.execution_category.value
                    if metadata.execution_category
                    else "read_only"
                ),
            }
        else:
            display = None  # type: ignore[assignment]
    except Exception:
        display = None  # type: ignore[assignment]

    if display is None:
        # Fallback defaults
        display = {
            "category": "",
            "access_mode": "readonly",
            "cost_tier": "free",
            "execution_category": "read_only",
        }

    if arguments is not None:
        try:
            from victor.tools.base import AccessMode
            from victor.tools.effective_access import resolve_effective_access

            if resolve_effective_access(tool_name, arguments) is AccessMode.READONLY:
                display["access_mode"] = AccessMode.READONLY.value
                display["execution_category"] = "read_only"
        except Exception:
            pass

    return display

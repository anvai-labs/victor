"""Caller-declared completion-critical tool exemptions (generic).

Some consumers model completion as a state transition performed by one of
their own tools (a proposal, submit, commit, or publish call). If the
framework's loop safeguards suppress that call — budget-exhaustion skips,
failed-signature and semantic-dedup skips, post-warning signature blocks —
the run can never finish: the caller's terminal call never executes no
matter how the model retries.

Consumers declare their completion-critical tool names (settings
``tools.completion_tools``, or the equivalent flat setting). The
declaration is generic: the framework never learns what a declared tool
does, only that its batch must be allowed to execute. With no declared
names every safeguard behaves exactly as before (flag-graduation policy,
ADR-012).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

from victor.tools.core_tool_aliases import normalize_model_tool_name
from victor.tools.tool_names import get_canonical_name

__all__ = [
    "CompletionToolPolicy",
    "normalize_completion_tool_names",
    "resolve_completion_tool_names",
]


def normalize_completion_tool_names(names: Iterable[Any]) -> Tuple[str, ...]:
    """Normalize a caller declaration to canonical tool names, dropping junk.

    Non-string entries and blank strings are silently ignored (a JSON
    round-trip must not crash loop safeguards), duplicates collapse, and
    names canonicalize through the shared tool-naming registry so a caller
    may declare either canonical or alias spellings.
    """
    normalized: "dict[str, None]" = {}
    if not isinstance(names, (list, tuple, set, frozenset)):
        # A malformed declaration (or a test double) must not crash the
        # loop safeguards that consult it.
        return ()
    for name in names:
        if not isinstance(name, str):
            continue
        stripped = name.strip()
        if not stripped:
            continue
        canonical = get_canonical_name(stripped) or stripped
        canonical = normalize_model_tool_name(canonical) or canonical
        normalized[canonical] = None
        lowered = canonical.lower()
        if lowered != canonical:
            normalized[lowered] = None
    return tuple(normalized)


def resolve_completion_tool_names(settings: Any) -> Tuple[str, ...]:
    """Resolve the completion-tools declaration from settings.

    Reads the nested ``tools.completion_tools`` group first, then a legacy
    flat ``completion_tools`` attribute. Missing or empty declarations
    resolve to an empty tuple (default-off parity).
    """
    if settings is None:
        return ()
    tools = getattr(settings, "tools", None)
    if tools is not None:
        declared = getattr(tools, "completion_tools", None)
        if declared:
            return normalize_completion_tool_names(declared)
    declared = getattr(settings, "completion_tools", None)
    if declared:
        return normalize_completion_tool_names(declared)
    return ()


class CompletionToolPolicy:
    """Immutable view of the caller's completion-critical tool declaration.

    Empty by construction when the caller declares nothing, in which case
    every query returns False and callers keep their legacy code path.
    """

    def __init__(self, tool_names: Iterable[Any] = ()) -> None:
        self._names = frozenset(normalize_completion_tool_names(tool_names))

    @property
    def enabled(self) -> bool:
        """True when the caller declared at least one completion tool."""
        return bool(self._names)

    @property
    def declared_names(self) -> frozenset:
        """Canonical (plus lowercase) spellings of the declared tools."""
        return self._names

    def is_completion_critical(self, tool_name: Any) -> bool:
        """True when ``tool_name`` is a caller-declared completion tool."""
        if not self._names or not isinstance(tool_name, str):
            return False
        name = tool_name.strip()
        return name in self._names or name.lower() in self._names

    def exempt_block_reason(
        self,
        tool_name: Any,
        block_reason: Optional[str],
    ) -> Optional[str]:
        """Clear the block reason for completion-critical tools, pass the rest.

        Use as a wrapper around any block-checker callable's result so a
        declared completion tool is never refused by that checker while
        every other tool keeps the original verdict.
        """
        if block_reason is not None and self.is_completion_critical(tool_name):
            return None
        return block_reason

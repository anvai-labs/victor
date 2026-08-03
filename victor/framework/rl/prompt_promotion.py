# Copyright 2026 Vijaykumar Singh <singhvjd@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Codegen core for promoting an evolved prompt candidate into source.

The promotion mechanism (FEP-0025 §"Promotion into source") rewrites a section
assignment in ``victor/agent/prompt_section_texts.py`` with a candidate's text.
This module holds the *pure* transformation — no DB, no filesystem, no argparse
— so the delicate parts (turning rendered completion markers back into
``{PLACEHOLDER}`` f-string interpolations, and swapping one triple-quoted
assignment without disturbing the rest of the file) are testable library code
rather than untested one-off script logic that writes a source-of-truth file.

``scripts/prompt_candidates.py`` (and any future ``victor prompts promote``
command) supplies the file I/O, the benchmark gate, and the diff/PR workflow
around :func:`build_promoted_source`.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


def marker_placeholders() -> Dict[str, str]:
    """Rendered marker value -> the name the source interpolates it under.

    Longest first: substituting a value that is a prefix of another would
    corrupt the longer one.
    """
    from victor.core import completion_markers as cm

    names = ("FILE_DONE_MARKER", "TASK_DONE_MARKER", "SUMMARY_MARKER", "BLOCKED_MARKER")
    pairs = {getattr(cm, name): name for name in names if hasattr(cm, name)}
    return dict(sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True))


def retemplatize(text: str) -> Tuple[str, Optional[str]]:
    """Turn rendered marker values back into ``{PLACEHOLDER}`` interpolations.

    ``COMPLETION_GUIDANCE`` is the only f-string in the section module, and it is
    also the section evolution targets most. A candidate's stored text is the
    *rendered* output, so writing it back verbatim would hardcode
    ``VICTOR_FILE_DONE::`` into the source and silently end
    ``completion_markers.py``'s role as the single definition of those tokens —
    renaming a marker would then change the detector and leave the prompt telling
    the model to emit the old one.

    Returns ``(templatized, error)``; a non-None error means do not write.
    """
    placeholders = marker_placeholders()
    if not any(value in text for value in placeholders):
        return text, "candidate contains none of the completion markers"

    # Literal braces would be interpolation sites once this becomes an f-string.
    stray = [ch for ch in "{}" if ch in text]
    if stray:
        return text, (
            f"candidate contains a literal {stray[0]!r}, which an f-string would read as "
            "an interpolation; promote this section by hand"
        )

    templatized = text
    for value, name in placeholders.items():
        templatized = templatized.replace(value, "{" + name + "}")

    rendered = templatized.format(**{name: value for value, name in placeholders.items()})
    # Round trip or refuse. A near-miss here means the substitution was lossy,
    # and the failure would only surface at runtime as a prompt that no longer
    # matches the detector.
    if rendered != text:
        return text, "re-templatizing did not round-trip; refusing to write"
    return templatized, None


def is_fstring_section(source: str, section: str) -> bool:
    """True when ``section`` is defined as an f-string assignment in ``source``."""
    return bool(re.search(rf'^{re.escape(section)} = f"""', source, re.MULTILINE))


def replace_section(source: str, section: str, body: str, is_fstring: bool) -> Optional[str]:
    """Swap one ``SECTION = \"\"\"...\"\"\".strip()`` assignment for a new body.

    Returns the updated source, or None when no such assignment is found.
    """
    pattern = re.compile(
        rf'^{re.escape(section)} = (f?)"""\n.*?\n""".strip\(\)$',
        re.DOTALL | re.MULTILINE,
    )
    match = pattern.search(source)
    if match is None:
        return None
    prefix = "f" if is_fstring else ""
    return (
        source[: match.start()]
        + f'{section} = {prefix}"""\n{body}\n""".strip()'
        + source[match.end() :]
    )


def build_promoted_source(
    source: str,
    section_name: str,
    candidate_text: str,
    provenance: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(updated_source, error)`` for promoting one candidate.

    Pure: performs no I/O. ``updated_source`` is None exactly when ``error`` is
    set, so a caller writes the file only on ``error is None``. Steps:

    1. If the section is an f-string, re-templatize the candidate's markers
       (refusing on any lossy or ambiguous substitution).
    2. Swap the section's assignment body for the promoted text.
    3. Prepend a ``provenance`` comment line above the assignment.
    """
    is_fstring = is_fstring_section(source, section_name)

    body = candidate_text
    if is_fstring:
        body, error = retemplatize(candidate_text)
        if error is not None:
            return None, error

    updated = replace_section(source, section_name, body, is_fstring)
    if updated is None:
        return None, (
            f'could not locate a \'{section_name} = """...""".strip()\' assignment; '
            "promote this section by hand"
        )

    updated = updated.replace(f"{section_name} = ", f"{provenance}\n{section_name} = ", 1)
    return updated, None

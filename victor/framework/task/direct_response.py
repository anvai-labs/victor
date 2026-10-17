"""Shared direct-response heuristics for crisp answer-only prompts.

This module centralizes the lightweight semantics needed to decide when a
prompt should be answered directly instead of entering the full agentic loop.
It intentionally focuses on direct-response behavior, not broader task typing
or budgeting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

_ACTION_WORDS = re.compile(
    r"\b("
    r"fix|create|write|edit|refactor|add|implement|update|delete|remove|"
    r"change|modify|build|deploy|run|test|debug|install|configure|setup|migrate"
    r")\b",
    re.IGNORECASE,
)

_CODEBASE_CONTEXT = re.compile(
    r"("
    r"`[^`]+`|"
    r"\b[a-z0-9_-]+\.[a-z0-9]{1,8}\b|"
    r"\b("
    r"api|code|codebase|database|db|directory|directories|"
    r"endpoint|endpoints|file|files|folder|folders|"
    r"import|imports|path|paths|repo|repository|"
    r"schema|schemas|sqlite|table|tables|tool|tools|workspace"
    r")\b"
    r")",
    re.IGNORECASE,
)

_CODE_ENTITY_CONTEXT = re.compile(
    r"\b("
    r"(this|that|the|our|my)\s+(class|classes|function|functions|method|methods|module|modules)|"
    r"[a-z_][a-z0-9_-]*\s+(class|function|method|module)"
    r")\b",
    re.IGNORECASE,
)

_EXACT_RESPONSE_PREFIX = re.compile(
    r"^\s*("
    r"(reply|respond|answer)\s+(with|using|exactly|only)\b|"
    r"(reply|respond|answer)\b.*\b(exactly|only)\b|"
    r"(say|print|output)\b"
    r")",
    re.IGNORECASE,
)

_EXACT_QUOTED_LITERAL = re.compile(
    r"^\s*(?:reply|respond|answer)\s+(?:with\s+)?(?:exactly|only)\b.*?[`'\"](?P<literal>[^`'\"]+)[`'\"]\s*$|"
    r"^\s*(?:say|print|output)\b\s*[`'\"](?P<literal_alt>[^`'\"]+)[`'\"]\s*$",
    re.IGNORECASE,
)

_EXACT_BARE_LITERAL = re.compile(
    r"^\s*(?:reply|respond|answer)\s+(?:with\s+)?(?:exactly|only)\s+"
    r"(?:(?:one|single)\s+\w+\s*:\s*)?(?P<literal>[A-Za-z0-9_.:/-]+)\s*$|"
    r"^\s*(?:say|print|output)\s+(?P<literal_alt>[A-Za-z0-9_.:/-]+)\s*$",
    re.IGNORECASE,
)

_GENERAL_QUERY_PREFIX = re.compile(
    r"^\s*("
    r"what\s+(is|are|does)|"
    r"who\s+(is|are)|"
    r"when\s+(is|does)|"
    r"where\s+(is|are|does)|"
    r"why\s+(is|does|do)|"
    r"how\s+(is|does|do|can)|"
    r"explain|describe|define|tell\s+me\s+about|summarize"
    r")\b",
    re.IGNORECASE,
)

# Conversational greetings and social phrases that need no tool usage.
# Pattern: short messages (≤ 60 chars when stripped) that are purely social.
_GREETING_PATTERN = re.compile(
    r"^\s*(hi|hey|hello|howdy|greetings|yo|sup|hiya|good\s+(morning|afternoon|evening|day)"
    r"|thanks|thank\s+you|cheers|bye|goodbye|ciao|later)"
    r"(\s+(there|everyone|all|folks|friend))?"  # allow "hi there", "hello everyone"
    r"\W*$",
    re.IGNORECASE,
)

_EXPLANATORY_QUERY_PREFIX = re.compile(
    r"^\s*(" r"why\s+(is|does|do)|" r"how\s+(is|does|do|can)" r")\b",
    re.IGNORECASE,
)

# Only complete trailing constraints are ignored for intent classification.
# Do not remove arbitrary negated phrases: repository references, conditional
# clauses and requested effects must still keep a task on the full runtime path.
# This never rewrites the prompt sent to the provider or grants tool permission.
_ANSWER_CONSTRAINT_SUFFIX = re.compile(
    r"(?<=[?!.;])\s*(?:"
    r"(?:do\s+not|don't)\s+use\s+tools|"
    r"(?:reply|respond|answer)\s+(?:with\s+)?only\s+the\s+(?:marker|answer)"
    r")\s*[.!;]*\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DirectResponseClassification:
    """Classification for prompts that should be answered directly."""

    is_direct_response: bool
    is_exact_response: bool
    confidence: float
    reason: str = ""


@dataclass
class DirectResponseOutputState:
    """Prompt-aware output state for direct-response normalization.

    This state object keeps exact-response stream buffering and final-response
    normalization in one reusable place so framework/service callers can share
    the same contract without UI-local heuristics.
    """

    prompt: str
    classification: DirectResponseClassification = field(init=False)
    literal: str | None = field(init=False)
    _buffered_content: list[str] = field(default_factory=list, init=False)
    _last_metadata: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.classification = classify_direct_response_prompt(self.prompt)
        self.literal = (
            extract_exact_response_literal(self.prompt)
            if self.classification.is_exact_response
            else None
        )

    @property
    def buffers_stream_content(self) -> bool:
        """Return whether content should be held until stream completion."""

        return self.classification.is_exact_response

    def consume_stream_content(
        self,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Consume stream content and return any immediately visible text."""

        if not content:
            return ""

        if isinstance(metadata, Mapping):
            self._last_metadata = dict(metadata)

        if not self.buffers_stream_content:
            return content

        self._buffered_content.append(content)
        return ""

    def flush_stream_content(self) -> tuple[str, dict[str, Any]]:
        """Return the normalized final stream content and last seen metadata."""

        if not self.buffers_stream_content:
            return "", dict(self._last_metadata)

        content = "".join(self._buffered_content)
        return self.normalize_final_response(content), dict(self._last_metadata)

    def normalize_final_response(self, response: str) -> str:
        """Normalize final response text against the original prompt contract."""

        return normalize_direct_response_output(self.prompt, response)


def has_codebase_context(message: str) -> bool:
    """Return whether the prompt explicitly points at repo/code/tool context."""

    text = message or ""
    return bool(_CODEBASE_CONTEXT.search(text) or _CODE_ENTITY_CONTEXT.search(text))


def classify_direct_response_prompt(message: str) -> DirectResponseClassification:
    """Classify whether a prompt should bypass agentic looping.

    Exact-answer prompts like ``Reply with exactly READY`` are the strongest
    signal. Broader general queries are only treated as direct-response prompts
    when they do not point at codebase/workspace/database context and do not
    request execution-oriented work.
    """

    text = (message or "").strip()
    while constraint := _ANSWER_CONSTRAINT_SUFFIX.search(text):
        text = text[: constraint.start()].rstrip()
    if not text:
        return DirectResponseClassification(False, False, 0.0, "empty")

    if _ACTION_WORDS.search(text):
        return DirectResponseClassification(False, False, 0.0, "action_request")

    # Short conversational greetings / social phrases need no tools.
    if len(text) <= 60 and _GREETING_PATTERN.match(text):
        return DirectResponseClassification(True, False, 0.95, "greeting")

    codebase_context = has_codebase_context(text)
    if _EXACT_RESPONSE_PREFIX.search(text) and not codebase_context:
        return DirectResponseClassification(True, True, 0.99, "exact_response")

    if codebase_context:
        return DirectResponseClassification(False, False, 0.0, "codebase_context")

    lower = text.lower()
    if _EXPLANATORY_QUERY_PREFIX.search(lower):
        return DirectResponseClassification(False, False, 0.0, "explanatory_query")
    if len(text) <= 160 and text.endswith("?"):
        return DirectResponseClassification(True, False, 0.92, "short_question")
    if _GENERAL_QUERY_PREFIX.search(lower):
        return DirectResponseClassification(True, False, 0.9, "general_query")

    return DirectResponseClassification(False, False, 0.0, "non_direct_response")


def extract_exact_response_literal(message: str) -> str | None:
    """Extract the literal expected by an exact-response prompt."""

    text = (message or "").strip()
    if not text:
        return None

    for pattern in (_EXACT_QUOTED_LITERAL, _EXACT_BARE_LITERAL):
        match = pattern.search(text)
        if not match:
            continue
        literal = match.group("literal") or match.group("literal_alt")
        if literal:
            return literal.strip()
    return None


def normalize_direct_response_output(prompt: str, response: str) -> str:
    """Collapse exact-response outputs down to the requested literal when present."""

    classification = classify_direct_response_prompt(prompt)
    if not classification.is_exact_response:
        return response

    literal = extract_exact_response_literal(prompt)
    if not literal:
        return response

    literal_pattern = re.compile(rf"(?<!\w){re.escape(literal)}(?!\w)")
    if literal_pattern.search(response or ""):
        return literal

    return response


def is_direct_response_prompt(message: str) -> bool:
    """Convenience boolean wrapper around direct-response classification."""

    return classify_direct_response_prompt(message).is_direct_response

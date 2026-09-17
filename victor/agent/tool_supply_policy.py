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

"""Pure tool-supply policy (ADR-019 orchestrator decomposition, increment 2).

The Tool-Necessity Gate (HDPO-inspired Q&A bypass) and the context-window tool
budgeter, lifted verbatim out of ``AgentOrchestrator``. Both are pure over their
inputs once their single orchestrator dependency is injected as a callable —
``edge_check`` (the DecisionService edge-model consult) for :func:`classify_tool_supply`,
and ``estimate_tokens`` (the provider-tier token estimator) for
:func:`demote_tools_to_fit`. This carries no orchestrator coupling and is directly
unit-testable. Behaviour is preserved exactly: keyword sets, branch order, length
gate, and STUB-demotion logic all match the original methods.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

logger = logging.getLogger(__name__)

#: Keywords that strongly suggest tool usage is required.
TOOL_SIGNAL_KEYWORDS = frozenset(
    {
        "file",
        "code",
        "search",
        "edit",
        "write",
        "read",
        "create",
        "delete",
        "run",
        "test",
        "debug",
        "fix",
        "build",
        "deploy",
        "docker",
        "git",
        "commit",
        "branch",
        "install",
        "refactor",
        "implement",
        "function",
        "class",
        "import",
        "variable",
        "error",
        "bug",
        "compile",
        "execute",
        "database",
        "query",
        "api",
        "endpoint",
        "config",
        "log",
        "trace",
        "mkdir",
        "move",
        "rename",
        "grep",
        "find",
        "ls",
        "bash",
        "shell",
        "pip",
        "npm",
        "make",
        "curl",
        "fetch",
        "download",
        "upload",
    }
)

#: Patterns that indicate pure Q&A (no tools needed).
QA_SIGNAL_PATTERNS = (
    "what is",
    "what are",
    "what does",
    "what do",
    "how does",
    "how do",
    "how is",
    "how are",
    "why does",
    "why do",
    "why is",
    "why are",
    "can you explain",
    "explain",
    "describe",
    "tell me about",
    "what's the difference",
    "thanks",
    "thank you",
    "hello",
    "hi ",
    "good morning",
    "good evening",
)

#: Bare continuation/affirmation user turns (< 15 chars, no tool keyword) would trip
#: the length gate below and drop the working tool set; keep the read-only core so
#: the model can still reason over in-progress work (classify_tool_supply: read_core).
CONTINUATION_TOKENS = frozenset(
    {
        "continue",
        "proceed",
        "go",
        "go on",
        "keep going",
        "next",
        "more",
        "again",
        "yes",
        "y",
        "ok",
        "okay",
        "sure",
        "do it",
        "apply",
        "apply it",
    }
)


def classify_tool_supply(
    context_msg: str,
    *,
    edge_check: Callable[[str, float], bool],
) -> str:
    """Decide tool supply for a (possibly conversational) turn (tool-supply P3).

    Uses a fast heuristic first (keyword scan), then optionally consults the edge
    model via ``edge_check`` for borderline Q&A. Returns one of:

    - ``"skip"``      — trivially-safe conversational turn (short greeting): no tools.
    - ``"read_core"`` — borderline Q&A: provide ONLY a minimal read-only core so the
      model can look something up if it turns out it needs to, rather than removing
      the entire tool set.
    - ``"tools"``     — proceed with normal tool selection.

    Args:
        context_msg: The user turn text.
        edge_check: ``(context_msg, heuristic_conf) -> bool`` — True when the edge
            model judges tools unnecessary (the DecisionService consult).
    """
    msg_lower = context_msg.lower().strip()

    # Bare continuation/affirmation of an in-progress task ("continue", "proceed",
    # "go", "yes", "apply it", ...). Checked BEFORE the length short-circuit.
    if msg_lower in CONTINUATION_TOKENS:
        return "read_core"

    # Very short messages are almost always greetings/Q&A — the only hard no-tools path.
    if len(msg_lower) < 15:
        # Unless they look like commands: "fix it", "run tests", etc.
        if any(kw in msg_lower for kw in ("fix", "run", "edit", "create", "delete")):
            return "tools"
        return "skip"

    # Heuristic: count tool-signal keywords vs Q&A patterns
    words = set(msg_lower.split())
    tool_signals = len(words & TOOL_SIGNAL_KEYWORDS)
    qa_match = any(msg_lower.startswith(pat) for pat in QA_SIGNAL_PATTERNS)

    # High-confidence heuristic paths
    if tool_signals >= 2:
        return "tools"  # Clearly needs tools
    if qa_match and tool_signals == 0:
        # Consult edge model for borderline Q&A (might still need tools). Even when it
        # says "skip", give the read-only core rather than nothing.
        skip = edge_check(context_msg, 0.85)
        return "read_core" if skip else "tools"

    # Ambiguous: 0-1 tool signals — default to providing tools
    if tool_signals == 0 and qa_match:
        skip = edge_check(context_msg, 0.6)
        return "read_core" if skip else "tools"

    return "tools"  # Default: provide tools


def should_skip_tools_via_edge(
    context_msg: str,
    heuristic_conf: float,
    *,
    container: Any = None,
) -> bool:
    """Ask the optional decision service whether a turn needs tools.

    The policy keeps the edge-model integration out of ``AgentOrchestrator``.
    It is deliberately fail-open: an unavailable decision service, an
    unsupported result, or any error leaves tools available.
    """
    try:
        from victor.agent.decisions.schemas import DecisionType
        from victor.agent.services.protocols.decision_service import get_decision_service

        service = get_decision_service(container)
        if service is None:
            return heuristic_conf >= 0.7

        decision = service.decide_sync(
            DecisionType.TOOL_NECESSITY,
            context={"message_excerpt": context_msg[:300]},
            heuristic_result={"requires_tools": heuristic_conf < 0.7},
            heuristic_confidence=heuristic_conf,
        )
        if decision.source == "heuristic" or decision.result is None:
            return heuristic_conf >= 0.7

        requires = decision.result.get("requires_tools", True)
        confidence = decision.result.get("confidence", 0.5)
        if not requires and confidence >= 0.6:
            logger.debug(
                "TOOL_NECESSITY: skipping tools for Q&A turn " "(confidence=%.2f, source=%s)",
                confidence,
                decision.source,
            )
            return True
        return False
    except Exception:
        logger.debug("Tool necessity check failed, defaulting to provide tools")
        return False


def demote_tools_to_fit(
    tools: List[Any],
    max_tokens: int,
    context_window: int,
    *,
    estimate_tokens: Callable[..., int],
    provider_category: Optional[str] = None,
    stub_estimate_tokens: Optional[Callable[..., int]] = None,
) -> List[Any]:
    """Demote or drop low-priority tools until within budget.

    Args:
        tools: List of tools to filter.
        max_tokens: Maximum tool tokens allowed (25% of context window).
        context_window: Context window size.
        estimate_tokens: ``(tool[, provider_category]) -> int`` token estimator.
        provider_category: Provider category for tier selection.
        stub_estimate_tokens: estimator for the temporary STUB probe; defaults
            to ``estimate_tokens``. Callers whose estimator caches by
            (tool, tier) must pass a cache-bypassing variant — the probe's
            temporary ``_schema_level`` patch makes its estimate
            unrepresentative of the real tier (adversarial-review finding).

    Returns:
        Filtered list of tools that fit within budget.
    """
    from victor.tools.enums import Priority, SchemaLevel

    # Sort by priority (CRITICAL first)
    sorted_tools = sorted(
        tools,
        key=lambda t: (t.priority.value if hasattr(t, "priority") else 99, t.name),
    )

    result: List[Any] = []
    current_tokens = 0

    for tool in sorted_tools:
        # Estimate current token cost using provider-specific tiers
        tool_cost = estimate_tokens(tool, provider_category)

        if current_tokens + tool_cost <= max_tokens:
            # Tool fits within budget
            result.append(tool)
            current_tokens += tool_cost
        elif hasattr(tool, "priority") and tool.priority == Priority.CRITICAL:
            # Critical tools MUST fit - demote to STUB
            try:
                # Temporarily override schema level to STUB
                original_schema = getattr(tool, "_schema_level", None)
                tool._schema_level = SchemaLevel.STUB
                stub_fn = stub_estimate_tokens or estimate_tokens
                stub_cost = stub_fn(tool)
                tool._schema_level = original_schema  # Restore

                if current_tokens + stub_cost <= max_tokens:
                    result.append(tool)
                    current_tokens += stub_cost
                    logger.debug(f"Demoted critical tool {tool.name} to STUB to fit budget")
                else:
                    logger.warning(
                        f"Critical tool {tool.name} ({stub_cost} tokens) exceeds budget "
                        f"even as STUB. Dropping tool."
                    )
            except Exception as e:
                logger.warning(f"Error demoting tool {tool.name}: {e}")
        else:
            # Skip non-critical tool
            logger.debug(
                f"Skipping {tool.name} (priority: {tool.priority if hasattr(tool, 'priority') else 'unknown'}) "
                f"to fit within {max_tokens} token budget"
            )

    logger.info(
        f"Demoted tools to fit context window: {len(tools)} → {len(result)} tools, "
        f"{current_tokens} tokens (budget: {max_tokens}, context: {context_window})"
    )

    return result

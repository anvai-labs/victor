# Copyright 2025 Vijaykumar Singh <vijay@anvaiops.com>
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

"""Built-in governance policies.

Three policies cover the most common governance needs and mirror Omnigent's
builtins:

* :class:`CostBudgetPolicy` — cap session spend, ask at soft thresholds.
* :class:`AskOnToolsPolicy` — require approval before high-impact tools.
* :class:`MaxToolCallsPolicy` — cap tool calls per session (loop/cost guard).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional, Pattern, Set

from victor.framework.policies.base import Policy
from victor.framework.policies.types import Phase, PolicyEvent, PolicyVerdict

logger = logging.getLogger(__name__)

# Message phases (gated at the turn boundary, not tool-scoped).
_MESSAGE_PHASES: Set[Phase] = {Phase.REQUEST, Phase.RESPONSE}


class CostBudgetPolicy(Policy):
    """Gate tool calls on cumulative session cost.

    Semantics follow Omnigent's "downgrade gate": once the hard ``max_cost_usd``
    is reached, tool calls are DENIED *while the active model is expensive*. The
    user can switch to a cheaper model (via ``/model``) to keep working — the
    gate only blocks expensive spend. Soft ``ask_thresholds_usd`` each trigger a
    one-time ASK as cost crosses them.

    If cost is unavailable (``context.cost_usd == 0.0`` because pricing isn't
    configured), the gate simply never trips (fail-open) — matching Omnigent's
    "no pricing -> gate never trips" behaviour.
    """

    name = "cost_budget"

    def __init__(
        self,
        max_cost_usd: Optional[float] = None,
        *,
        ask_thresholds_usd: Optional[Iterable[float]] = None,
        expensive_models: Optional[Iterable[str]] = None,
    ) -> None:
        """Initialize the cost budget.

        Args:
            max_cost_usd: Hard cap; None disables the hard gate.
            ask_thresholds_usd: Soft thresholds that each trigger one ASK.
            expensive_models: Substrings identifying expensive models the hard
                gate applies to (e.g. ["opus", "fable"]). Empty/None means the
                gate applies to every model.
        """
        self._max = max_cost_usd
        self._ask_thresholds: List[float] = sorted(ask_thresholds_usd or [])
        self._expensive = [m.lower() for m in (expensive_models or [])]
        self._asked: Set[float] = set()

    def phases(self) -> Set[Phase]:
        """Cost is evaluated before each tool call."""
        return {Phase.TOOL_CALL}

    def _on_expensive_model(self, model: Optional[str]) -> bool:
        if not self._expensive:
            return True
        model_l = (model or "").lower()
        return any(tok in model_l for tok in self._expensive)

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Deny over the hard cap (on expensive models); ask at thresholds."""
        cost = event.context.cost_usd or 0.0

        if (
            self._max is not None
            and cost >= self._max
            and self._on_expensive_model(event.context.model)
        ):
            return PolicyVerdict.deny(
                f"Session cost ${cost:.2f} has reached the budget of ${self._max:.2f}. "
                f"Switch to a cheaper model to continue.",
                policy_name=self.name,
            )

        for threshold in self._ask_thresholds:
            if cost >= threshold and threshold not in self._asked:
                self._asked.add(threshold)
                return PolicyVerdict.ask(
                    f"Session cost ${cost:.2f} crossed the ${threshold:.2f} checkpoint.",
                    policy_name=self.name,
                )

        return PolicyVerdict.allow()


class AskOnToolsPolicy(Policy):
    """Require human approval before running a configured set of tools.

    Useful for high-impact tools (shell, code execution, deletes). Tool names
    are matched exactly against ``tool_names``.
    """

    name = "ask_on_tools"

    def __init__(self, tool_names: Optional[Iterable[str]] = None) -> None:
        """Initialize with the set of tools that require approval."""
        self._tools: Set[str] = set(tool_names or [])

    def phases(self) -> Set[Phase]:
        """Approval is requested before the tool runs."""
        return {Phase.TOOL_CALL}

    def applies_to(self, tool_name: str) -> bool:
        """Only the configured tools trigger this policy."""
        return tool_name in self._tools

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Always ask for the configured tools."""
        return PolicyVerdict.ask(
            f"Approval required before running '{event.tool_name}'.",
            policy_name=self.name,
        )


class DenyToolsPolicy(Policy):
    """Hard-block a denylist of tools (no approval bypass).

    Unlike :class:`AskOnToolsPolicy`, listed tools are DENIED outright — useful
    for read-only sessions (e.g. block ``run_command``/``write_file``). Tool
    names are matched exactly.
    """

    name = "deny_tools"

    def __init__(self, tool_names: Optional[Iterable[str]] = None) -> None:
        """Initialize with the set of tools to block."""
        self._tools: Set[str] = set(tool_names or [])

    def phases(self) -> Set[Phase]:
        """Blocking happens before the tool runs."""
        return {Phase.TOOL_CALL}

    def applies_to(self, tool_name: str) -> bool:
        """Only the configured (denied) tools trigger this policy."""
        return tool_name in self._tools

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Deny the configured tools."""
        return PolicyVerdict.deny(
            f"Tool '{event.tool_name}' is blocked by policy.",
            policy_name=self.name,
        )


class AllowToolsPolicy(Policy):
    """Restrict execution to an allowlist — every other tool is DENIED.

    When configured with a non-empty allowlist, any tool *not* in the list is
    blocked. Tool names are matched exactly. An empty allowlist disables the
    policy (the builder only constructs it when the list is non-empty).
    """

    name = "allow_tools"

    def __init__(self, tool_names: Optional[Iterable[str]] = None) -> None:
        """Initialize with the set of permitted tools."""
        self._tools: Set[str] = set(tool_names or [])

    def phases(self) -> Set[Phase]:
        """Restriction is enforced before the tool runs."""
        return {Phase.TOOL_CALL}

    def applies_to(self, tool_name: str) -> bool:
        """Applies to any tool NOT on the allowlist (those are denied)."""
        return bool(self._tools) and tool_name not in self._tools

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Deny tools outside the allowlist."""
        return PolicyVerdict.deny(
            f"Tool '{event.tool_name}' is not on the allowed-tools list.",
            policy_name=self.name,
        )


class MaxToolCallsPolicy(Policy):
    """Cap the number of tool calls per session (loop / runaway-cost guard).

    The counter is per policy instance, which is per session (the middleware
    chain is built per orchestrator/session).
    """

    name = "max_tool_calls"

    def __init__(self, limit: Optional[int] = None) -> None:
        """Initialize with the maximum number of tool calls (None disables)."""
        self._limit = limit
        self._count = 0

    def phases(self) -> Set[Phase]:
        """Counts each tool call before execution."""
        return {Phase.TOOL_CALL}

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Deny once the call count exceeds the limit."""
        if self._limit is None:
            return PolicyVerdict.allow()
        self._count += 1
        if self._count > self._limit:
            return PolicyVerdict.deny(
                f"Exceeded the maximum of {self._limit} tool calls for this session.",
                policy_name=self.name,
            )
        return PolicyVerdict.allow()


def _compile_patterns(patterns: Optional[Iterable[str]]) -> List[Pattern[str]]:
    """Compile regex patterns, skipping (with a warning) any that don't compile."""
    compiled: List[Pattern[str]] = []
    for raw in patterns or []:
        try:
            compiled.append(re.compile(raw))
        except re.error as exc:  # pragma: no cover - defensive: bad config
            logger.warning("Skipping invalid policy regex %r: %s", raw, exc)
    return compiled


class RedactContentPolicy(Policy):
    """Redact regex matches from message text (REQUEST and/or RESPONSE).

    Replaces every match of any configured pattern with ``placeholder`` and
    ALLOWs the (modified) message. Useful for stripping secrets/PII from the
    user message before it reaches the LLM, or from the assistant output before
    it is shown/stored. ALLOW-with-redaction never blocks the turn.
    """

    name = "redact_content"

    def __init__(
        self,
        patterns: Optional[Iterable[str]] = None,
        *,
        placeholder: str = "[REDACTED]",
        phases: Optional[Iterable[Phase]] = None,
    ) -> None:
        """Initialize with redaction patterns and the phases to apply them in."""
        self._patterns = _compile_patterns(patterns)
        self._placeholder = placeholder
        self._phases: Set[Phase] = set(phases) if phases is not None else set(_MESSAGE_PHASES)

    def phases(self) -> Set[Phase]:
        """Apply in the configured message phases (default REQUEST + RESPONSE)."""
        return self._phases

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Redact matches; ALLOW unchanged when nothing matches."""
        text = event.content
        if not text or not self._patterns:
            return PolicyVerdict.allow()
        redacted = text
        for pattern in self._patterns:
            redacted = pattern.sub(self._placeholder, redacted)
        if redacted == text:
            return PolicyVerdict.allow()
        return PolicyVerdict.allow(policy_name=self.name, modified_content=redacted)


class BlockPatternPolicy(Policy):
    """DENY a message whose text matches any configured pattern.

    A hard content gate for message phases — e.g. block prompt-injection markers
    in the user message (REQUEST) or forbidden content in the assistant output
    (RESPONSE). Matching is a regex ``search`` (anywhere in the text).
    """

    name = "block_pattern"

    def __init__(
        self,
        patterns: Optional[Iterable[str]] = None,
        *,
        phases: Optional[Iterable[Phase]] = None,
        reason: Optional[str] = None,
    ) -> None:
        """Initialize with block patterns and the phases to enforce them in."""
        self._patterns = _compile_patterns(patterns)
        self._phases: Set[Phase] = set(phases) if phases is not None else {Phase.REQUEST}
        self._reason = reason

    def phases(self) -> Set[Phase]:
        """Enforce in the configured message phases (default REQUEST only)."""
        return self._phases

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """DENY when the text matches any pattern; ALLOW otherwise."""
        text = event.content or ""
        for pattern in self._patterns:
            if pattern.search(text):
                return PolicyVerdict.deny(
                    self._reason or "Message blocked by content policy.",
                    policy_name=self.name,
                )
        return PolicyVerdict.allow()


__all__ = [
    "CostBudgetPolicy",
    "AskOnToolsPolicy",
    "DenyToolsPolicy",
    "AllowToolsPolicy",
    "MaxToolCallsPolicy",
    "RedactContentPolicy",
    "BlockPatternPolicy",
]

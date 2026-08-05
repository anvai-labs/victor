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

"""Policy composition engine.

Runs an ordered list of :class:`~victor.framework.policies.base.Policy`
instances against a :class:`PolicyEvent` and composes a single verdict:

* the **first DENY short-circuits** (later policies do not run);
* any **ASK** (with no DENY) collapses to a single approval request;
* otherwise **ALLOW**.

The policy list is expected to be pre-ordered strictest-first
(session -> agent -> server), matching Omnigent's trust model: the most local
authority gets the final say first. Argument and result modifications are
threaded between policies, so a later policy sees earlier modifications.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable, Dict, List, Optional, Sequence

from victor.framework.policies.base import Policy
from victor.framework.policies.types import (
    UNSET,
    PolicyAction,
    PolicyEvent,
    PolicyVerdict,
)

logger = logging.getLogger(__name__)

# A thin observability hook: (topic, payload) -> None. Kept as a plain callable
# to avoid coupling the engine to the event bus implementation.
EventEmitter = Callable[[str, Dict[str, Any]], None]


class PolicyEngine:
    """Composes multiple policies into one verdict per event."""

    def __init__(
        self,
        policies: Sequence[Policy],
        *,
        event_emitter: Optional[EventEmitter] = None,
    ) -> None:
        """Initialize the engine.

        Args:
            policies: Ordered policies (strictest/most-local first).
            event_emitter: Optional ``(topic, payload)`` sink for DENY/ASK
                audit events. Failures in the emitter are swallowed.
        """
        self._policies: List[Policy] = list(policies)
        self._emit = event_emitter

    @property
    def policies(self) -> List[Policy]:
        """The configured policies, in evaluation order."""
        return list(self._policies)

    async def evaluate(self, event: PolicyEvent) -> PolicyVerdict:
        """Evaluate ``event`` against all applicable policies."""
        current_args: Dict[str, Any] = dict(event.arguments or {})
        args_modified = False
        current_result: Any = event.result
        result_modified = False
        current_content: Any = event.content
        content_modified = False

        ask_reasons: List[str] = []
        ask_policy = ""

        for policy in self._policies:
            if event.phase not in policy.phases():
                continue
            if not policy.applies_to(event.tool_name):
                continue

            sub_event = replace(
                event,
                arguments=current_args,
                result=current_result,
                content=current_content,
            )
            try:
                verdict = await policy.evaluate(sub_event)
            except Exception:  # pragma: no cover - defensive
                # A misbehaving policy must not crash the tool pipeline. Fail
                # open for that policy (skip it) but record the failure.
                logger.exception("Policy %s raised during evaluation; skipping", policy.name)
                continue

            if verdict.is_deny:
                self._emit_decision(event, verdict)
                return verdict

            if verdict.modified_arguments is not None:
                current_args = verdict.modified_arguments
                args_modified = True
            if verdict.modified_result is not UNSET:
                current_result = verdict.modified_result
                result_modified = True
            if verdict.modified_content is not UNSET:
                current_content = verdict.modified_content
                content_modified = True

            if verdict.is_ask:
                ask_reasons.append(verdict.reason)
                if not ask_policy:
                    ask_policy = verdict.policy_name

        final_args: Optional[Dict[str, Any]] = current_args if args_modified else None
        final_result: Any = current_result if result_modified else UNSET
        final_content: Any = current_content if content_modified else UNSET

        if ask_reasons:
            combined = PolicyVerdict(
                action=PolicyAction.ASK,
                reason="; ".join(r for r in ask_reasons if r),
                policy_name=ask_policy,
                modified_arguments=final_args,
                modified_result=final_result,
                modified_content=final_content,
            )
            self._emit_decision(event, combined)
            return combined

        return PolicyVerdict(
            action=PolicyAction.ALLOW,
            modified_arguments=final_args,
            modified_result=final_result,
            modified_content=final_content,
        )

    def _emit_decision(self, event: PolicyEvent, verdict: PolicyVerdict) -> None:
        """Emit a DENY/ASK audit event (best-effort)."""
        if self._emit is None:
            return
        try:
            self._emit(
                "policy.decision",
                {
                    "phase": event.phase.value,
                    "tool_name": event.tool_name,
                    "decision": verdict.action.value,
                    "policy": verdict.policy_name,
                    "reason": verdict.reason,
                    "session_id": event.context.session_id,
                },
            )
        except Exception:  # pragma: no cover - observability must not break flow
            logger.debug("Policy event emitter failed", exc_info=True)


__all__ = ["PolicyEngine", "EventEmitter"]

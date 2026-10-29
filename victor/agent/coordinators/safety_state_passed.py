"""State-passed safety coordinator (SPA-4).

Wraps SafetyCoordinator with the state-passed pattern:
- Input: ContextSnapshot (for reading tool calls and settings)
- Output: CoordinatorResult with safety check transitions

The SafetyCoordinator already has zero orchestrator dependencies,
making it an ideal candidate for state-passed migration.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from victor_contracts.safety import (
    SafetyAction,
    SafetyCheckResult,
    SafetyCoordinator,
    SafetyRule,
)
from victor.agent.coordinators.state_context import (
    CoordinatorResult,
    ContextSnapshot,
    TransitionBatch,
)

logger = logging.getLogger(__name__)


class SafetyStatePassedCoordinator:
    """State-passed safety coordinator.

    Takes a ContextSnapshot and proposed tool call, returns a
    CoordinatorResult indicating whether the operation is safe
    and what transitions to apply (warnings, blocks, state updates).

    Usage:
        snapshot = create_snapshot(orchestrator)
        coord = SafetyStatePassedCoordinator(rules=my_rules)
        result = await coord.check(snapshot, "git", ["push", "--force"])
        if not result.should_continue:
            # Operation blocked by safety rules
            print(result.reasoning)
    """

    def __init__(
        self,
        rules: Optional[List[SafetyRule]] = None,
        strict_mode: bool = False,
    ) -> None:
        self._inner = SafetyCoordinator(
            strict_mode=strict_mode,
            enable_default_rules=True,
        )
        if rules:
            for rule in rules:
                self._inner.register_rule(rule)

    async def check(
        self,
        context: ContextSnapshot,
        tool_name: str,
        tool_args: List[str],
    ) -> CoordinatorResult:
        """Check safety of a proposed tool call.

        Args:
            context: Immutable snapshot (for reading conversation state)
            tool_name: Name of the tool being called
            tool_args: Arguments to the tool

        Returns:
            CoordinatorResult with:
            - should_continue=False if operation is blocked
            - transitions for warnings and safety state updates
        """
        check_result = self._inner.check_safety(tool_name, tool_args)
        return self._to_coordinator_result(check_result, tool_name)

    def _to_coordinator_result(
        self,
        check: SafetyCheckResult,
        tool_name: str,
    ) -> CoordinatorResult:
        """Convert SafetyCheckResult to CoordinatorResult."""
        batch = TransitionBatch()

        # Record safety check in conversation state
        batch.update_state(
            "last_safety_check",
            {
                "tool": tool_name,
                "is_safe": check.is_safe,
                "action": check.action.value,
                "warnings": check.warnings,
            },
            scope="conversation",
        )

        if check.action == SafetyAction.BLOCK:
            return CoordinatorResult(
                transitions=batch,
                reasoning=f"BLOCKED: {check.block_reason or 'Safety rule violation'}",
                confidence=1.0,
                should_continue=False,
                metadata=check.to_dict(),
            )

        if check.action == SafetyAction.WARN:
            # Record warnings but allow continuation
            if check.warnings:
                batch.update_state(
                    "safety_warnings",
                    check.warnings,
                    scope="conversation",
                )
            return CoordinatorResult(
                transitions=batch,
                reasoning=f"WARNING: {'; '.join(check.warnings)}",
                confidence=0.7,
                should_continue=True,
                metadata=check.to_dict(),
            )

        if check.action == SafetyAction.REQUIRE_CONFIRMATION:
            return CoordinatorResult(
                transitions=batch,
                reasoning=f"CONFIRMATION REQUIRED: {check.confirmation_prompt}",
                confidence=0.5,
                should_continue=False,  # Pause for confirmation
                metadata=check.to_dict(),
            )

        # ALLOW — safe operation
        return CoordinatorResult(
            transitions=batch,
            reasoning="Safe operation",
            confidence=1.0,
            should_continue=True,
            metadata=check.to_dict(),
        )

"""Tool approval gate for HITL (Human-in-the-Loop) tool execution control.

Uses DangerLevel classification to gate tool execution. Three modes:
- auto: all tools approved (default, backward compatible)
- dangerous: MEDIUM+ danger requires approval
- all: every tool requires approval

Pattern: Strategy -- pluggable approval handler injected at construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class ApprovalMode(str, Enum):
    """Tool approval modes for HITL control."""

    AUTO = "auto"
    DANGEROUS = "dangerous"
    ALL = "all"


@dataclass
class ApprovalDecision:
    """Result of a tool approval request."""

    approved: bool
    reason: str = ""
    modified_args: Optional[Dict[str, Any]] = None


class ToolApprovalGate:
    """Checks tool danger level and requests approval when needed.

    The approval_handler is injected -- it can be a CLI prompt, API callback,
    or auto-approver for testing. When no handler is provided, dangerous
    tools are auto-approved with a warning log.
    """

    def __init__(
        self,
        mode: ApprovalMode = ApprovalMode.AUTO,
        approval_handler: Optional[Callable] = None,
        timeout_seconds: float = 60.0,
    ):
        self._mode = mode
        self._handler = approval_handler
        self._timeout = timeout_seconds

    @property
    def mode(self) -> ApprovalMode:
        return self._mode

    def needs_approval(self, tool_name: str, danger_level: str) -> bool:
        """Check if a tool requires approval based on mode and danger level."""
        if self._mode == ApprovalMode.AUTO:
            return False
        if self._mode == ApprovalMode.ALL:
            return True
        # DANGEROUS mode: block medium, high, critical
        return danger_level in ("medium", "high", "critical")

    async def request_approval(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        danger_level: str,
        reasoning: str = "",
    ) -> ApprovalDecision:
        """Request approval for a tool call. Non-blocking for auto mode."""
        if not self.needs_approval(tool_name, danger_level):
            return ApprovalDecision(approved=True, reason="auto-approved")

        if self._handler is None:
            logger.warning(
                f"[hitl] Tool '{tool_name}' (danger={danger_level}) needs approval "
                f"but no handler configured -- auto-approving"
            )
            return ApprovalDecision(approved=True, reason="no handler configured")

        try:
            return await self._handler(tool_name, arguments, danger_level, reasoning)
        except Exception as e:
            logger.error(f"[hitl] Approval handler error: {e} -- auto-approving")
            return ApprovalDecision(approved=True, reason=f"handler error: {e}")

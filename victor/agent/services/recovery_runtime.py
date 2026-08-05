# Copyright 2026 Vijaykumar Singh <vijay@anvaiops.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Service-owned recovery runtime helper."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from victor.providers.base import StreamChunk


class RecoveryRuntime:
    """Bridge orchestrator recovery state into the canonical recovery service."""

    def __init__(self, runtime_host: Any) -> None:
        self._runtime = runtime_host

    def create_recovery_context(self, stream_ctx: Any) -> Any:
        """Create StreamingRecoveryContext from the current runtime state."""
        from victor.agent.services.recovery_service import StreamingRecoveryContext

        runtime = self._runtime
        current_session = getattr(runtime._streaming_controller, "current_session", None)
        now = time.time()
        elapsed_time = 0.0
        if current_session is not None:
            elapsed_time = now - current_session.start_time

        return StreamingRecoveryContext(
            iteration=stream_ctx.total_iterations,
            elapsed_time=elapsed_time,
            tool_calls_used=runtime.tool_calls_used,
            tool_budget=runtime.tool_budget,
            max_iterations=stream_ctx.max_total_iterations,
            session_start_time=(current_session.start_time if current_session is not None else now),
            last_quality_score=stream_ctx.last_quality_score,
            streaming_context=stream_ctx,
            provider_name=runtime.provider_name,
            model=runtime.model,
            temperature=runtime.temperature,
            unified_task_type=stream_ctx.unified_task_type,
            is_analysis_task=stream_ctx.is_analysis_task,
            is_action_task=stream_ctx.is_action_task,
        )

    async def handle_recovery_with_integration(
        self,
        stream_ctx: Any,
        full_content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        mentioned_tools: Optional[List[str]] = None,
    ) -> Any:
        """Build recovery context and delegate to RecoveryService."""
        runtime = self._runtime
        recovery_ctx = self.create_recovery_context(stream_ctx)
        return await runtime._recovery_service.handle_recovery_with_integration(
            recovery_ctx,
            full_content,
            tool_calls,
            mentioned_tools,
            message_adder=runtime.add_message,
        )

    def apply_recovery_action(
        self,
        recovery_action: Any,
        stream_ctx: Any,
    ) -> Optional[StreamChunk]:
        """Build recovery context and delegate action application to RecoveryService."""
        runtime = self._runtime
        recovery_ctx = self.create_recovery_context(stream_ctx)
        return runtime._recovery_service.apply_recovery_action(
            recovery_action,
            recovery_ctx,
            message_adder=runtime.add_message,
        )

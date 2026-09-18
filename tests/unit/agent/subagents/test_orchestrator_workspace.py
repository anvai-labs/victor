"""Public spawn carries the assigned workspace into the member configuration."""

from unittest.mock import AsyncMock, MagicMock, patch

from victor.agent.subagents.base import SubAgentResult
from victor.agent.subagents.orchestrator import SubAgentOrchestrator
from victor.core.shared_types import SubAgentRole


async def test_spawn_passes_working_directory(tmp_path):
    parent = MagicMock()
    parent.settings.subagent_default_tool_budget = 10
    parent.settings.subagent_default_context_limit = 1000
    with patch("victor.agent.subagents.orchestrator.SubAgent") as cls:
        cls.return_value.execute = AsyncMock(
            return_value=SubAgentResult(
                success=True,
                summary="ok",
                details={},
                tool_calls_used=0,
                context_size=0,
                duration_seconds=0,
            )
        )
        result = await SubAgentOrchestrator(parent).spawn(
            SubAgentRole.EXECUTOR,
            "write",
            working_directory=str(tmp_path),
            member_id="worker",
            parent_session_id="parent",
        )
    assert result.success
    assert cls.call_args.args[0].working_directory == str(tmp_path)
    assert cls.call_args.args[0].resolve_member_session_id() == "parent-worker"

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


async def test_explicit_member_gateway_needs_no_upstream_credential(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://127.0.0.1:18788/v1")
    monkeypatch.setenv("SANDHI_GATEWAY_VIRTUAL_KEY_ZAI", "vk-test")
    key = MagicMock(side_effect=AssertionError("must not resolve upstream key"))
    monkeypatch.setattr("victor.config.api_keys.get_api_key", key)
    with patch(
        "victor.providers.factory.ManagedProviderFactory.create", new_callable=AsyncMock
    ) as create:
        parent = SubAgentOrchestrator(SimpleNamespace(model="glm-5.3"))
        await parent._resolve_override_provider("zai", "glm-5.3")
        assert create.call_args.kwargs["gateway"] == {
            "url": "http://127.0.0.1:18788",
            "virtual_key": "vk-test",
        }
        assert create.call_args.kwargs["api_key"] == "vk-test"
        key.assert_not_called()


async def test_invalid_member_gateway_cannot_inherit_parent(monkeypatch):
    import pytest
    from types import SimpleNamespace

    monkeypatch.setenv("SANDHI_GATEWAY_URL", "http://127.0.0.1:18788")
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY_ZAI", raising=False)
    monkeypatch.delenv("SANDHI_GATEWAY_VIRTUAL_KEY", raising=False)
    with pytest.raises(ValueError, match="gateway resolution failed"):
        await SubAgentOrchestrator(SimpleNamespace(model="glm-5.3"))._resolve_override_provider(
            "zai", None
        )
